# Chess Imitation Learning with PGN Data and PyTorch

## Overview

This project implements **imitation learning** for chess using **PGN (Portable Game Notation)** files, **PyTorch**, and **python-chess**. The goal is to train a neural network that predicts chess moves by learning from real games stored in PGN format.

The project generates a dataset of chess board positions and corresponding moves from PGN files, then trains a neural network to learn to predict moves from given positions. After training, the model can predict moves on unseen chess positions, imitating the play style of the games it was trained on.

## Features

- **PGN Integration**: The project uses PGN files to generate training data from real chess games.
- **Neural Network**: A simple fully connected neural network is built using PyTorch to predict moves.
- **Board Representation**: Chess board states are converted into a numerical tensor representation that serves as input to the neural network.
- **Move Prediction**: The neural network is trained to predict moves for any given board state, based on the data generated from PGN files.
  
## Project Structure

```
├── chess_il.py     # Main Python script for training and testing the model
├── README.md       # Project documentation (this file)
├── requirements.txt # List of required Python libraries
└── pgn/            # Directory to store PGN files (you need to create this)
```

## Prerequisites

### Python Libraries

You need the following Python libraries:

- **python-chess**: For interacting with the chess board and parsing PGN files.
- **PyTorch**: For building and training the neural network.

You can install these libraries using the provided `requirements.txt` file.

### PGN Files

You'll need to provide your own PGN files for training. Place these files in a directory named `pgn` in the project root.

### Neural Network Architecture

The neural network (`ChessNet`) is a simple fully connected feedforward network with the following architecture:
- Input: Board state (8x8x12 tensor flattened into a vector).
- Hidden layers: Two fully connected layers with ReLU activations.
- Output: A vector representing 4096 possible chess moves (64x64 grid for from/to squares).

## How to Run the Project

### Step 1: Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/your-username/Chess-RL.git
cd Chess-RL
```

### Step 2: Install Dependencies

Make sure you have all required dependencies installed. You can install them using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 3: Prepare PGN Files

Create a directory named `pgn` in the project root and place your PGN files there:

```bash
mkdir pgn
# Copy your PGN files into the pgn directory
```

### Step 4: Run the Script

Run the main script to generate data from PGN files, train the model, and test its predictions:

```bash
python chess_il.py
```

### Step 5: Testing the Model

Once the training is complete, the model will predict a move for an initial board state (as a simple test).

## How the Code Works

1. **Data Generation**: 
    - PGN files are read from the `pgn` directory.
    - Chess positions and corresponding moves are extracted from these games and stored as training data.

2. **Model Training**:
    - The training data is fed to a neural network (implemented using PyTorch) which learns to predict moves from given board states.
    - Training is done using a supervised learning approach where the loss function is the cross-entropy between the predicted move and the actual move played in the game.

3. **Move Prediction**:
    - After training, the model can predict a move for a given board position by outputting the most probable move (encoded as from-square and to-square).

## Example Output

During training, you will see output like this:
```
Generating data from PGN files...
Training model...
Epoch [1/5], Loss: 0.4567
Epoch [2/5], Loss: 0.3189
...
Testing the model on a random position...
Predicted move: e2e4
```

The predicted move is displayed after testing the model on an initial chess position.

## Next Steps

- **Improve the Dataset**: You can use more PGN files or games from specific players or tournaments to tailor the model's play style.
- **Optimize the Neural Network**: Try using more advanced architectures like convolutional neural networks (CNNs) or transformers for better performance.
- **Reinforcement Learning**: Once the model is competent at making reasonable moves, you could expand the project to include reinforcement learning where the model learns through self-play.

## License

This project is licensed under the MPL-2.0 License. See the `LICENSE` file for details.

## Acknowledgments

- **PyTorch**: For enabling the neural network framework.
- **python-chess**: For providing the chess board manipulation utilities and PGN parsing capabilities.

---

### Notes:
- The current model architecture is quite simple. For better performance, consider implementing more sophisticated models (e.g., AlphaZero-style convolutional networks or transformer models).
- The quality of the model's predictions will depend on the quality and quantity of the PGN data used for training. Using games from strong players or from specific openings can help tailor the model's play style.
