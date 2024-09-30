from flask import Flask, request, jsonify
import chess
import chess.pgn
import torch
import torch.nn as nn
import argparse
import logging
import random
import traceback
from flask_cors import CORS
from collections import deque
import chess.polyglot

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)  # Enable CORS for all routes and origins, and allow credentials

# Global variables to store the model, tokenizer, current game state, and opening book
model = None
tokenizer = None
board = None
move_history = deque()
current_position = -1
opening_book = None

# Set AI color (e.g., chess.BLACK or chess.WHITE)
ai_color = chess.BLACK

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(8 * 8 * 12 + 1 + 4 + 64 + 2, 1024)  # Updated input size
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 4096)  # Output for 64*64 possible moves (simplified)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(model_path, book_path):
    """
    Load the ChessNet model and opening book from the specified paths.

    Args:
        model_path (str): Path to the trained model.
        book_path (str): Path to the opening book file.

    Global variables:
        model: The loaded ChessNet model.
        opening_book: The loaded opening book reader.
    """
    global model, opening_book
    try:
        model = ChessNet()
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set the model to evaluation mode
        logger.info("Model loaded successfully")

        opening_book = chess.polyglot.open_reader(book_path)
        logger.info("Opening book loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model or opening book: {str(e)}")
        raise

def board_to_tensor(board):
    """
    Convert a chess board state to a tensor representation for input to the neural network.

    This function takes a chess.Board object and converts it into a tensor that represents
    the board state, including piece positions, turn information, castling rights,
    en passant possibilities, and move counters.

    Args:
        board (chess.Board): The current chess board state.

    Returns:
        torch.Tensor: A tensor representation of the board state with shape (1, 773).

    The tensor is structured as follows:
    1. Piece positions (8x8x12 = 768 elements):
       - 12 channels, one for each piece type and color
       - 1 indicates the presence of a piece, 0 indicates absence
    2. Turn information (1 element):
       - 1.0 for White's turn, 0.0 for Black's turn
    3. Castling rights (4 elements):
       - One element each for White kingside, White queenside, Black kingside, Black queenside
       - 1.0 if the castling right is available, 0.0 if not
    4. En passant square (64 elements):
       - One-hot encoding of the en passant square if available, all zeros if not
    5. Move counters (2 elements):
       - Halfmove clock (normalized to [0, 1])
       - Fullmove number (normalized assuming max 500 moves)

    The resulting tensor is flattened and unsqueezed to create a batch dimension of 1.
    """
    piece_map = board.piece_map()
    tensor = torch.zeros((8, 8, 12), dtype=torch.float32)
    piece_types = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                   'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        tensor[row, col, piece_types[piece.symbol()]] = 1
    
    # Add turn information
    turn = torch.tensor([1.0 if board.turn == chess.WHITE else 0.0], dtype=torch.float32)
    
    # Add castling rights (4 bits)
    castling_rights = torch.tensor([
        float(board.has_kingside_castling_rights(chess.WHITE)),
        float(board.has_queenside_castling_rights(chess.WHITE)),
        float(board.has_kingside_castling_rights(chess.BLACK)),
        float(board.has_queenside_castling_rights(chess.BLACK))
    ], dtype=torch.float32)
    
    # Add en passant square information (1 byte)
    en_passant = torch.zeros(64, dtype=torch.float32)
    if board.ep_square is not None:
        en_passant[board.ep_square] = 1.0
    
    # Add move counters
    halfmove_clock = torch.tensor([board.halfmove_clock / 100.0], dtype=torch.float32)  # Normalize to [0, 1]
    fullmove_number = torch.tensor([board.fullmove_number / 500.0], dtype=torch.float32)  # Normalize assuming max 500 moves
    
    return torch.cat((
        tensor.flatten(),
        turn,
        castling_rights,
        en_passant,
        halfmove_clock,
        fullmove_number
    )).unsqueeze(0)

def label_to_move(label):
    from_square = label // 64
    to_square = label % 64
    return chess.Move(from_square, to_square)

def simple_evaluate(board):
    if board.is_checkmate():
        return -10000 if board.turn == ai_color else 10000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            score += value if piece.color == ai_color else -value

            # Evaluate piece mobility
            mobility = len(list(board.attacks(square)))
            mobility_factor = 10
            if piece.color == ai_color:
                score += mobility * mobility_factor
            else:
                score -= mobility * mobility_factor

    # Evaluate king safety
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    
    white_king_attackers = len(board.attackers(chess.BLACK, white_king_square))
    black_king_attackers = len(board.attackers(chess.WHITE, black_king_square))
    
    king_safety_factor = 50
    if ai_color == chess.WHITE:
        score -= white_king_attackers * king_safety_factor
        score += black_king_attackers * king_safety_factor
    else:
        score += white_king_attackers * king_safety_factor
        score -= black_king_attackers * king_safety_factor

    # Evaluate center control
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    center_control_factor = 30
    for square in center_squares:
        if board.piece_at(square):
            if board.piece_at(square).color == ai_color:
                score += center_control_factor
            else:
                score -= center_control_factor
        else:
            white_attackers = len(board.attackers(chess.WHITE, square))
            black_attackers = len(board.attackers(chess.BLACK, square))
            if ai_color == chess.WHITE:
                score += (white_attackers - black_attackers) * center_control_factor
            else:
                score += (black_attackers - white_attackers) * center_control_factor

    return score

def move_ordering(board, move):
    """
    Assign a score to moves for move ordering in the minimax algorithm.
    Higher score means higher priority.
    """
    score = 0
    if board.is_capture(move):
        captured_piece = board.piece_at(move.to_square)
        if captured_piece:
            capture_value = piece_values[captured_piece.piece_type]
            score += capture_value
    if move.promotion:
        score += 1000  # High priority for promotions
    return score

def minimax(board, depth, alpha, beta, maximizing_player):
    """
    Minimax algorithm with alpha-beta pruning for move evaluation.

    Args:
        board (chess.Board): The current board state.
        depth (int): The current depth in the search tree.
        alpha (float): The alpha value for alpha-beta pruning.
        beta (float): The beta value for alpha-beta pruning.
        maximizing_player (bool): True if the current player is maximizing, False otherwise.

    Returns:
        tuple: The evaluation score of the best move and the best move itself.
    """
    if depth == 0 or board.is_game_over():
        return simple_evaluate(board), None

    best_move = None
    if maximizing_player:
        max_eval = float('-inf')
        legal_moves = list(board.legal_moves)
        legal_moves.sort(key=lambda move: move_ordering(board, move), reverse=True)
        for move in legal_moves:
            board.push(move)
            eval, _ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        legal_moves = list(board.legal_moves)
        legal_moves.sort(key=lambda move: move_ordering(board, move))
        for move in legal_moves:
            board.push(move)
            eval, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

def generate_move(num_moves=10, depth=4):
    """
    Generate chess moves using the opening book, the loaded ChessNet model, or minimax.

    Args:
        num_moves (int): Number of move candidates to generate. Default is 10.
        depth (int): Depth for the minimax search. Default is 4.

    Returns:
        list: A list containing the best move generated.
        str: The source of the generated move (book, minimax, or random).

    Global variables:
        model: The loaded ChessNet model.
        board: The current chess board state.
        opening_book: The opening book reader.
    """
    ensure_board_initialized()
    
    global model, board, opening_book
    
    try:
        # First, try to get a move from the opening book
        book_move = opening_book.get(board)
        if book_move:
            logger.info(f"Move found in opening book: {book_move.move}")
            return [board.san(book_move.move)], "book"

        # Use minimax to evaluate the best move
        maximizing_player = (board.turn == ai_color)
        _, best_move = minimax(board, depth, float('-inf'), float('inf'), maximizing_player)
        
        if best_move:
            logger.info(f"Generated move: {best_move.uci()}")
            return [board.san(best_move)], "minimax"
        
        logger.info("No good moves found, falling back to random move")
        legal_moves = list(board.legal_moves)
        if legal_moves:
            random_move = random.choice(legal_moves)
            return [board.san(random_move)], "random"
        else:
            logger.error("No legal moves available")
            return [], "random"
    except Exception as e:
        logger.error(f"Error in generate_move: {str(e)}")
        return [], "random"

# Add this at the beginning of the file, after the global variable declarations
def ensure_board_initialized():
    global board
    if board is None:
        board = chess.Board()
        logger.info("Board initialized")

@app.route('/init', methods=['POST'])
def init_game():
    """
    Initialize a new chess game.

    Returns:
        dict: A JSON response indicating the status of the initialization.

    Global variables:
        board: The chess board to be initialized.
        move_history: The history of moves to be reset.
        current_position: The current position in the move history to be reset.
    """
    global board, move_history, current_position
    try:
        board = chess.Board()
        move_history = deque()
        current_position = -1
        logger.info(f"/init : {board.fen()}")
        return jsonify({"status": "ok", "message": "New game initialized"})
    except Exception as e:
        logger.error(f"Error in init_game: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/move', methods=['POST'])
def make_move():
    """
    Apply a move to the current chess board.

    Expected JSON payload:
        {
            "move": "string"  # The move in UCI notation
        }

    Returns:
        dict: A JSON response indicating the status of the move application.

    Global variables:
        board: The current chess board state.
        move_history: The history of moves made in the game.
        current_position: The current position in the move history.
    """
    ensure_board_initialized()
    
    logger.info(f"Received /move request: {request.json}")
    logger.info(f"Request headers: {request.headers}")
    logger.info(f"Request data: {request.data}")
    if not request.is_json:
        logger.error("Invalid JSON received")
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400
    
    move = request.json.get('move')
    promotion = request.json.get('promotion')
    if not move:
        logger.error("Move not provided in request")
        return jsonify({"status": "error", "message": "Move not provided"}), 400
    
    global board, move_history, current_position

    logger.info(f"/move {move} before : {board.fen()}")
    try:
        move_obj = chess.Move.from_uci(move)
        
        # Check if it's a pawn promotion move
        if board.piece_at(move_obj.from_square) and board.piece_at(move_obj.from_square).piece_type == chess.PAWN and chess.square_rank(move_obj.to_square) in [0, 7]:
            if not promotion:
                return jsonify({"status": "promotion_required", "message": "Pawn promotion piece required"}), 400
            move_obj.promotion = chess.Piece.from_symbol(promotion).piece_type

        if move_obj in board.legal_moves:
            board.push(move_obj)
            current_position += 1
            if current_position < len(move_history):
                # If we're not at the end of the history, truncate the future moves
                move_history = deque(list(move_history)[:current_position + 1])
            move_history.append(move_obj)
            logger.info(f"/move after : {board.fen()}")
            return jsonify({"status": "ok", "message": f"Move {move} applied", "fen": board.fen()})
        else:
            logger.error(f"Illegal move: {move}")
            return jsonify({"status": "error", "message": "Illegal move"}), 400
    except ValueError as e:
        logger.error(f"/move Error applying move: {str(e)}")
        return jsonify({"status": "error", "message": f"Invalid move: {str(e)}"}), 400

@app.route('/get_move', methods=['GET'])
def get_ai_move():
    """
    Generate and apply an AI move to the current chess board.

    Returns:
        dict: A JSON response containing the AI's move and the new board state.

    Global variables:
        board: The current chess board state.
    """
    ensure_board_initialized()
    
    global board
    logger.info("Received /get_move request")
    try:
        if board.is_game_over():
            result = board.result()
            logger.info(f"Game over. Result: {result}")
            return jsonify({"status": "game_over", "result": result})
        
        generated_moves, source = generate_move()
        logger.info(f"/get_move Generated moves: {generated_moves}")
        
        if generated_moves:
            move_san = generated_moves[0]
            move = board.parse_san(move_san)
            board.push(move)
            logger.info(f"AI move applied: {move_san}")
            return jsonify({
                "status": "ok",
                "move": move.uci(),
                "source": source,
                "new_fen": board.fen(en_passant='fen')
            })
        
        logger.info(f"/get_move Falling back to random move")
        legal_moves = list(board.legal_moves)
        if legal_moves:
            random_move = random.choice(legal_moves)
            board.push(random_move)
            logger.info(f"/get_move Random move: {random_move.uci()}")
            return jsonify({
                "status": "ok",
                "move": random_move.uci(),
                "source": "random",
                "new_fen": board.fen(en_passant='fen')
            })
        else:
            logger.error("No valid moves available")
            return jsonify({"status": "error", "message": "No valid moves available"}), 500
    except Exception as e:
        logger.error(f"Error in get_ai_move: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/board', methods=['GET'])
def get_board():
    """
    Get the current state of the chess board.

    Returns:
        dict: A JSON response containing the current board state in FEN notation.

    Global variables:
        board: The current chess board state.
    """
    global board
    try:
        return jsonify({"status": "ok", "fen": board.fen(en_passant='fen')})
    except Exception as e:
        logger.error(f"Error in get_board: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 Not Found errors.

    Args:
        error: The error object.

    Returns:
        dict: A JSON response indicating a 404 error.
    """
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404

@app.route('/undo', methods=['POST'])
def undo_move():
    global board, move_history, current_position
    if current_position >= 0:
        board.pop()
        current_position -= 1
        return jsonify({"status": "ok", "message": "Move undone", "fen": board.fen()})
    else:
        return jsonify({"status": "error", "message": "No moves to undo"}), 400

@app.route('/redo', methods=['POST'])
def redo_move():
    global board, move_history, current_position
    if current_position < len(move_history) - 1:
        move = move_history[current_position + 1]
        board.push(move)
        current_position += 1
        return jsonify({"status": "ok", "message": "Move redone", "fen": board.fen()})
    else:
        return jsonify({"status": "error", "message": "No moves to redo"}), 400

@app.errorhandler(Exception)
def handle_exception(e):
    """
    Handle unhandled exceptions.

    Args:
        e: The exception object.

    Returns:
        dict: A JSON response indicating an internal server error.
    """
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return jsonify({"status": "error", "message": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess server using ChessNet model and opening book")
    parser.add_argument("--model", default="./model/chess_model.pth", help="Path to the trained model (default: ./model/chess_model.pth)")
    parser.add_argument("--book", default="./opening_books/Titans.bin", help="Path to the opening book file (default: ./opening_books/Titans.bin)")
    parser.add_argument("--port", type=int, default=9999, help="Port to run the server on (default: 9999)")
    
    args = parser.parse_args()

    load_model(args.model, args.book)

    app.run(debug=True, port=args.port , host='0.0.0.0')
