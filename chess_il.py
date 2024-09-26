import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# Neural network to predict the best move from the board state
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(8 * 8 * 12 + 1, 1024)  # +1 for the turn information
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 4096)  # Output for 64*64 possible moves (simplified)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Function to convert the board state into a numerical representation
def board_to_tensor(board):
    piece_map = board.piece_map()
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    piece_types = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                   'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        tensor[row, col, piece_types[piece.symbol()]] = 1
    
    # Add turn information
    turn = np.array([1.0 if board.turn == chess.WHITE else 0.0], dtype=np.float32)
    
    return torch.tensor(np.concatenate((tensor.flatten(), turn)))

# Function to encode move as a label (64*64 possible moves)
def move_to_label(move):
    return move.from_square * 64 + move.to_square

# Function to decode label back to a move
def label_to_move(label):
    from_square = label // 64
    to_square = label % 64
    return chess.Move(from_square, to_square)

# Function to read PGN files from a directory
def read_pgn_files(directory):
    games = []
    for filename in os.listdir(directory):
        if filename.endswith('.pgn'):
            filepath = os.path.join(directory, filename)
            print(f"Reading PGN file: {filename}")
            with open(filepath) as pgn_file:
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    games.append(game)
    return games

# Generate dataset using PGN games
def generate_data(pgn_directory, num_games=1000):
    data = []
    games = read_pgn_files(pgn_directory)
    print(f"Found {len(games)} games in the PGN directory.")
    
    if not games:
        raise ValueError(f"No PGN games found in directory: {pgn_directory}")
    
    total_moves = 0
    for i, game in enumerate(games[:num_games]):
        board = game.board()
        for move in game.mainline_moves():
            total_moves += 1
            if total_moves % 1000 == 0:  # Print progress every 1000 moves
                print(f"Processed {i+1} games, {total_moves} total moves")
            data.append((board_to_tensor(board), move_to_label(move)))
            board.push(move)
    
    print(f"Processed {min(num_games, len(games))} games. Total positions: {len(data)}")
    return data

# Train the neural network using PGN data
def train_model(data, model, epochs=10, batch_size=64, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            boards, labels = zip(*batch)
            boards = torch.stack(boards)
            labels = torch.tensor(labels)
            
            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(data)}")

# Test the model by making a prediction
def test_model(board, model):
    board_tensor = board_to_tensor(board).unsqueeze(0)
    with torch.no_grad():
        output = model(board_tensor)
    
    # Get top 10 moves
    top_moves = torch.topk(output, 10).indices[0]
    
    # Find the first legal move
    for move_label in top_moves:
        move = label_to_move(move_label.item())
        if move in board.legal_moves:
            return move, True  # True indicates the move was predicted
    
    # If no legal moves found in top 10, return a random legal move
    return random.choice(list(board.legal_moves)), False  # False indicates a random move

# Function to print the board with enumeration
def print_board_with_enumeration(board):
    board_str = str(board)
    rows = board_str.split('\n')
    
    # Add column labels (a-h)
    col_labels = '   a b c d e f g h'
    print(col_labels)
    
    # Add row numbers and board contents
    for i, row in enumerate(rows):
        print(f"{8-i}  {row}  {8-i}")
    
    # Repeat column labels at the bottom
    print(col_labels)

# Function to play against the model
def play_against_model(model):
    board = chess.Board()
    
    while not board.is_game_over():
        print_board_with_enumeration(board)
        print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
        
        if board.turn == chess.WHITE:
            # Human's turn (White)
            while True:
                try:
                    move_uci = input("Enter your move (in UCI format, e.g., 'e2e4'): ")
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move. Try again.")
                except ValueError:
                    print("Invalid input. Please use UCI format (e.g., 'e2e4').")
        else:
            # Model's turn (Black)
            move, is_predicted = test_model(board, model)
            if is_predicted:
                move_type = 'predicted'
            else:
                move_type = 'random'
            print(f"Model's move ({move_type}): {move}")
            board.push(move)
    
    print_board_with_enumeration(board)
    print("Game over. Result:", board.result())

# Main function
if __name__ == "__main__":
    # Initialize the neural network model
    model = ChessNet()

    # Step 1: Generate data using PGN files
    pgn_directory = "pgn"  # Assuming PGN files are stored in a 'pgn' directory
    print("Generating data from PGN files...")
    try:
        if not os.path.exists(pgn_directory):
            raise ValueError(f"Directory not found: {pgn_directory}")
        data = generate_data(pgn_directory, num_games=100)  # You can increase this for better results
        print(f"Generated {len(data)} training examples")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure that PGN files are present in the 'pgn' directory.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)
    
    if not data:
        print("No data was generated. Exiting.")
        exit(1)
    
    # Step 2: Train the model on the generated data
    print("Training model...")
    train_model(data, model, epochs=5)

    # Step 3: Choose between testing the model or playing against it
    choice = input("Enter 1 to test the model on the initial position, or 2 to play against the model: ")
    
    if choice == '1':
        # Test the model
        print("Testing the model on the initial position...")
        board = chess.Board()
        print("Initial board:")
        print_board_with_enumeration(board)
        print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
        
        predicted_move, is_predicted = test_model(board, model)
        print(f"Predicted move: {predicted_move}")
        if is_predicted:
            print("This move was predicted by the model.")
        else:
            print("This move was randomly selected from legal moves.")
        
        if predicted_move in board.legal_moves:
            print("The move is legal.")
        else:
            print("Warning: The move is not legal!")
            print("Legal moves:", list(board.legal_moves))
    elif choice == '2':
        # Play against the model
        print("Starting a game against the model. You'll play as White.")
        play_against_model(model)
    else:
        print("Invalid choice. Exiting.")
