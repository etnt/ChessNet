import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import logging
import os
import argparse
import json

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
def board_to_tensor(board, device):
    piece_map = board.piece_map()
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    piece_types = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                   'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        tensor[row, col, piece_types[piece.symbol()]] = 1
    
    # Add turn information
    turn = np.array([1.0 if board.turn == chess.WHITE else 0.0], dtype=np.float32)
    
    return torch.tensor(np.concatenate((tensor.flatten(), turn)), device=device)

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
def generate_data(pgn_directory, device, max_games=None):
    data = []
    games = read_pgn_files(pgn_directory)
    print(f"Found {len(games)} games in the PGN directory.")
    
    if not games:
        raise ValueError(f"No PGN games found in directory: {pgn_directory}")
    
    total_moves = 0
    num_games = len(games) if max_games is None else min(max_games, len(games))
    for i, game in enumerate(games[:num_games]):
        board = game.board()
        for move in game.mainline_moves():
            total_moves += 1
            if total_moves % 1000 == 0:  # Print progress every 1000 moves
                print(f"Processed {i+1} games, {total_moves} total moves")
            data.append((board_to_tensor(board, device), move_to_label(move)))
            board.push(move)
    
    print(f"Processed {num_games} games. Total positions: {len(data)}")
    return data

# Train the neural network using PGN data
def train_model(data, model, device, epochs=10, batch_size=64, learning_rate=0.001):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            boards, labels = zip(*batch)
            boards = torch.stack(boards).to(device)
            labels = torch.tensor(labels, device=device)
            
            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(data)}")

# Test the model by making a prediction
def test_model(board, model, device):
    model.to(device)
    board_tensor = board_to_tensor(board, device).unsqueeze(0)
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
def play_against_model(model, device):
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
            move, is_predicted = test_model(board, model, device)
            if is_predicted:
                move_type = 'predicted'
            else:
                move_type = 'random'
            print(f"Model's move ({move_type}): {move}")
            board.push(move)
    
    print_board_with_enumeration(board)
    print("Game over. Result:", board.result())

# Function to save the model in a Hugging Face compatible format
def save_model(model, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save the model architecture and weights
    model_path = os.path.join(model_dir, "chess_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model.__class__.__name__,
    }, model_path)
    
    # Save a config file (optional, but useful for Hugging Face)
    config_path = os.path.join(model_dir, "config.json")
    config = {
        "model_type": "ChessNet",
        "input_size": 8 * 8 * 12 + 1,
        "hidden_size1": 1024,
        "hidden_size2": 512,
        "output_size": 4096,
    }
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    print(f"Model saved in {model_dir}")

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Chess Imitation Learning")
    parser.add_argument("--max-games", type=int, default=None, help="Maximum number of games to process (default: all available)")
    parser.add_argument("--model-dir", type=str, default="./model", help="Directory to save the trained model (default: ./model)")
    parser.add_argument("--play", action="store_true", help="Play against the model after training")
    parser.add_argument("--pgn-dir", type=str, default="./pgn", help="Directory containing PGN files (default: ./pgn)")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "mps", "auto"], help="Device to use for training (default: auto)")
    return parser.parse_args()

# Function to select the appropriate device
def select_device(device_preference):
    if device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Main function
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Select the appropriate device
    device = select_device(args.device)
    print(f"Using device: {device}")

    # Initialize the neural network model
    model = ChessNet()

    # Step 1: Generate data using PGN files
    print(f"Generating data from PGN files in {args.pgn_dir}...")
    try:
        if not os.path.exists(args.pgn_dir):
            raise ValueError(f"Directory not found: {args.pgn_dir}")
        data = generate_data(args.pgn_dir, device, max_games=args.max_games)
        print(f"Generated {len(data)} training examples")
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Please ensure that PGN files are present in the '{args.pgn_dir}' directory.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)
    
    if not data:
        print("No data was generated. Exiting.")
        exit(1)
    
    # Step 2: Train the model on the generated data
    print("Training model...")
    train_model(data, model, device, epochs=5)

    # Step 3: Save the trained model
    print("Saving model...")
    save_model(model, args.model_dir)

    # Step 4: Either play against the model or test it on the initial position
    if args.play:
        print("Starting a game against the model. You'll play as White.")
        play_against_model(model, device)
    else:
        # Test the model
        print("Testing the model on the initial position...")
        board = chess.Board()
        print("Initial board:")
        print_board_with_enumeration(board)
        print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
        
        predicted_move, is_predicted = test_model(board, model, device)
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
