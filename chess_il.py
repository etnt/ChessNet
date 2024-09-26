import chess
import chess.engine
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------
# Breakdown of the Code
#
# ChessNet Class: This is a simple fully connected neural network that maps
# a board state (flattened into a 1D vector) to a move. The move is represented
# by a 64 × 64 matrix, where each possible move is an index
#  (from square * 64 + to square).
#
# Board Representation (board_to_tensor): We represent the chess board as
# an 8x8x12 tensor, where each slice along the third dimension represents
# a piece type (6 for white and 6 for black). This tensor is then flattened
# nto a 1D vector as input to the neural network.
#
# Move Encoding (move_to_label): Each move is represented as a label
# (integer) by encoding the from_square and to_square on the board into
# a single number.
#
# Data Generation: We generate training data by playing random moves
# using Stockfish. Stockfish suggests the best move, and we save the
# board state and the corresponding best move label.
#
# Training Loop: We train the network using CrossEntropyLoss.
# The network learns to predict the best move label given a board state.
#
# Testing: We test the trained model by predicting the best move on
# a random board. The model outputs a label, which is converted back
# into a chess move.
# ---------------------------------------------------------------------

# Initialize Stockfish engine
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # Adjust the path for your system
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

def get_best_move(board):
    # Set the board position
    result = engine.play(board, chess.engine.Limit(time=0.5))
    best_move = result.move

    return best_move

# Neural network to predict the best move from the board state
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(8 * 8 * 12, 1024)
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
    return torch.tensor(tensor).view(1, -1)  # Flatten to a vector

# Function to encode move as a label (64*64 possible moves)
def move_to_label(move):
    return move.from_square * 64 + move.to_square

# Function to decode label back to a move
def label_to_move(label):
    from_square = label // 64
    to_square = label % 64
    return from_square, to_square

# Generate dataset using Stockfish
def generate_data(num_games=1000):
    data = []
    for i in range(num_games):
        board = chess.Board()
        num_moves = 1
        while not board.is_game_over():
            print(f"Number of Games: {i+1}/{num_games} , number of moves: {num_moves}", end='\r')
            #sys.stdout.flush()
            best_move = get_best_move(board)  # Assuming this function gets the best move from Stockfish
            logging.info(f"Board FEN: {board.fen()}")
            logging.info(f"Best move: {best_move}")
            if best_move not in board.legal_moves:
                logging.error(f"Illegal move detected: {best_move}")
                break
            board.push(best_move)
            data.append((board_to_tensor(board), move_to_label(best_move)))
            num_moves += 1
    return data

# Train the neural network using Stockfish data
def train_model(data, model, epochs=10, batch_size=64, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            boards, labels = zip(*batch)
            boards = torch.cat(boards)
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
    board_tensor = board_to_tensor(board)
    with torch.no_grad():
        output = model(board_tensor)
    _, predicted_label = torch.max(output, 1)
    from_square, to_square = label_to_move(predicted_label.item())
    return chess.Move(from_square, to_square)

# Main function
if __name__ == "__main__":
    # Initialize the neural network model
    model = ChessNet()

    # Step 1: Generate data using Stockfish
    print("Generating data using Stockfish...")
    data = generate_data(num_games=10)  # You can increase this for better results
    
    # Step 2: Train the model on the generated data
    print("Training model...")
    train_model(data, model, epochs=5)

    # Step 3: Test the model
    print("Testing the model on a random position...")
    board = chess.Board()
    print("Initial board:")
    print(board)
    
    predicted_move = test_model(board, model)
    print(f"Predicted move: {predicted_move}")
    
    engine.quit()  # Close Stockfish engine
