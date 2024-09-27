import pygame
import chess
import requests
import sys
from requests.exceptions import RequestException

API_URL = "http://localhost:9999"  # Update this with your server's URL

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH = 600
HEIGHT = 700  # Increased height to accommodate buttons and status message area
SQUARE_SIZE = WIDTH // 8
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Game vs AI")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)
HIGHLIGHT_COLOR = (255, 255, 0, 128)  # Semi-transparent yellow

# Load chess piece images
pieces = {}
for color in ['w', 'b']:
    for piece in ['p', 'r', 'n', 'b', 'q', 'k']:
        image = pygame.image.load(f"chess_pieces/{color}{piece}.png")
        pieces[piece.upper() if color == 'w' else piece.lower()] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))

def draw_board(board, selected_square):
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else GRAY
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            
            # Highlight the selected square
            if selected_square is not None and selected_square == chess.square(col, 7 - row):
                highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                highlight_surface.fill(HIGHLIGHT_COLOR)
                screen.blit(highlight_surface, (col * SQUARE_SIZE, row * SQUARE_SIZE))
            
            piece = board.piece_at(chess.square(col, 7 - row))
            if piece:
                piece_image = pieces.get(piece.symbol())
                if piece_image:
                    screen.blit(piece_image, (col * SQUARE_SIZE, row * SQUARE_SIZE))
                else:
                    print(f"Warning: Image for piece '{piece.symbol()}' not found")

def get_square_from_mouse(pos):
    x, y = pos
    return chess.square(x // SQUARE_SIZE, 7 - (y // SQUARE_SIZE))

def draw_button(screen, text, position, width, height, color, text_color):
    pygame.draw.rect(screen, color, (*position, width, height))
    font = pygame.font.Font(None, 36)
    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=(position[0] + width // 2, position[1] + height // 2))
    screen.blit(text_surface, text_rect)

def promote_pawn():
    promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    promotion_images = [pieces[chess.PIECE_SYMBOLS[piece].upper()] for piece in promotion_pieces]
    
    promotion_rects = []
    for i, image in enumerate(promotion_images):
        rect = image.get_rect()
        rect.centerx = WIDTH // 2
        rect.centery = HEIGHT // 2 - 75 + i * 50
        promotion_rects.append(rect)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                for i, rect in enumerate(promotion_rects):
                    if rect.collidepoint(pos):
                        return promotion_pieces[i]
        
        screen.fill(WHITE)
        for i, (image, rect) in enumerate(zip(promotion_images, promotion_rects)):
            screen.blit(image, rect)
        pygame.display.flip()

def handle_events(board, selected_square, status_message, undo_button, redo_button, new_game_button):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            
            if undo_button.collidepoint(pos):
                status_message = handle_undo(board, status_message)
                selected_square = None
            
            elif redo_button.collidepoint(pos):
                status_message = handle_redo(board, status_message)
                selected_square = None
            
            elif new_game_button.collidepoint(pos):
                status_message = start_new_game(board, status_message)
                selected_square = None
            
            else:
                selected_square, status_message = handle_move(board, selected_square, pos, status_message)
    
    return selected_square, status_message

def handle_undo(board, status_message):
    try:
        response = requests.post(f"{API_URL}/undo")
        response.raise_for_status()
        response_data = response.json()
        if response_data["status"] == "ok":
            board.set_fen(response_data["fen"])
            status_message = "Move undone"
        else:
            status_message = response_data.get("message", "Error undoing move")
    except RequestException as e:
        status_message = f"Connection error: {str(e)}"
    return status_message

def handle_redo(board, status_message):
    try:
        response = requests.post(f"{API_URL}/redo")
        response.raise_for_status()
        response_data = response.json()
        if response_data["status"] == "ok":
            board.set_fen(response_data["fen"])
            status_message = "Move redone"
        else:
            status_message = response_data.get("message", "Error redoing move")
    except RequestException as e:
        status_message = f"Connection error: {str(e)}"
    return status_message

def handle_move(board, selected_square, pos, status_message):
    clicked_square = get_square_from_mouse(pos)
    
    if selected_square is None:
        piece = board.piece_at(clicked_square)
        if piece and piece.color == board.turn:
            selected_square = clicked_square
            status_message = f"Selected {chess.SQUARE_NAMES[selected_square]}. Choose destination."
        else:
            status_message = "Select a valid piece to move."
    else:
        move = chess.Move(selected_square, clicked_square)
        promotion = None
        if board.piece_at(selected_square) == chess.Piece(chess.PAWN, board.turn) and chess.square_rank(clicked_square) in [0, 7]:
            promotion = promote_pawn()
            if promotion:
                move = chess.Move(selected_square, clicked_square, promotion=promotion)
        
        if move in board.legal_moves:
            status_message = send_move_to_server(board, move, promotion, status_message)
            selected_square = None
        else:
            status_message = f"Invalid move: {chess.SQUARE_NAMES[selected_square]} to {chess.SQUARE_NAMES[clicked_square]}. Select a piece to move."
            selected_square = None  # Reset selected square on invalid move
    
    return selected_square, status_message

def send_move_to_server(board, move, promotion, status_message):
    try:
        response = requests.post(f"{API_URL}/move", json={"move": move.uci(), "promotion": chess.PIECE_SYMBOLS[promotion] if promotion else None}, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        response_data = response.json()
        if response_data["status"] == "ok":
            board.push(move)
            status_message = "AI is thinking..."
            status_message = get_ai_move(board, status_message)
        else:
            status_message = f"Error applying move: {response_data.get('message', 'Unknown error')}"
    except RequestException as e:
        status_message = f"Connection error: {str(e)}"
    return status_message

def get_ai_move(board, status_message):
    try:
        response = requests.get(f"{API_URL}/get_move")
        response.raise_for_status()
        response_data = response.json()
        predicted = "predicted" if response_data.get("predicted", False) else "random"
        if response_data["status"] == "ok":
            ai_move = chess.Move.from_uci(response_data["move"])
            board.push(ai_move)
            status_message = f"AI {predicted} move: {ai_move.uci()}. Your turn."
        elif response_data["status"] == "game_over":
            status_message = f"Game over. Result: {response_data['result']}"
        else:
            status_message = f"Error getting AI move: {response_data.get('message', 'Unknown error')}"
    except RequestException as e:
        status_message = f"Connection error: {str(e)}"
    return status_message

def start_new_game(board, status_message):
    try:
        response = requests.post(f"{API_URL}/init")
        response.raise_for_status()
        if response.json()["status"] == "ok":
            board.reset()
            status_message = "New game started. Your turn."
        else:
            status_message = "Error starting new game"
    except RequestException as e:
        status_message = f"Connection error: {str(e)}"
    return status_message

def update_ui(board, selected_square, status_message, undo_button, redo_button, new_game_button, button_width, button_height):
    screen.fill(WHITE)
    draw_board(board, selected_square)
    
    # Draw undo, redo, and new game buttons
    draw_button(screen, "Undo", (undo_button.x, undo_button.y), button_width, button_height, LIGHT_BLUE, BLACK)
    draw_button(screen, "Redo", (redo_button.x, redo_button.y), button_width, button_height, LIGHT_BLUE, BLACK)
    draw_button(screen, "New", (new_game_button.x, new_game_button.y), button_width, button_height, LIGHT_BLUE, BLACK)
    
    # Draw status message area
    status_area_height = 50
    pygame.draw.rect(screen, LIGHT_BLUE, (0, HEIGHT - status_area_height, WIDTH, status_area_height))
    
    # Draw status message
    font = pygame.font.Font(None, 24)
    text = font.render(status_message, True, BLACK)
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT - status_area_height // 2))
    screen.blit(text, text_rect)

    pygame.display.flip()

def main():
    board = chess.Board()
    selected_square = None
    status_message = "Your turn"

    # Define button dimensions and positions
    button_width = 100
    button_height = 40
    button_y = HEIGHT - 100  # Position buttons above the status message area
    undo_button = pygame.Rect(10, button_y, button_width, button_height)
    redo_button = pygame.Rect(WIDTH - button_width - 10, button_y, button_width, button_height)
    new_game_button = pygame.Rect(WIDTH // 2 - button_width // 2, button_y, button_width, button_height)

    while True:
        selected_square, status_message = handle_events(board, selected_square, status_message, undo_button, redo_button, new_game_button)
        update_ui(board, selected_square, status_message, undo_button, redo_button, new_game_button, button_width, button_height)

if __name__ == "__main__":
    main()
