import math
import random
import sys
import numpy as np
from copy import deepcopy
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB

# --- Global variables ---
ROW_COUNT = 6
COLUMN_COUNT = 7
EMPTY, OUR_AI, OPP_AI = 0, 1, 2

def create_board(rows, cols):
    global ROW_COUNT, COLUMN_COUNT
    ROW_COUNT, COLUMN_COUNT = rows, cols
    return [[EMPTY] * cols for _ in range(rows)]

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return 0 <= col < COLUMN_COUNT and board[0][col] == EMPTY

def get_next_open_row(board, col):
    if not (0 <= col < COLUMN_COUNT):
        return None
    for r in range(ROW_COUNT - 1, -1, -1):
        if board[r][col] == EMPTY:
            return r
    return None

# --- Version 1 & 3 feature: Emoji colored board with blue squares ---
def print_board(board):
    blue_square = "\U0001F7E6"  # Blue square for empty cell
    print("\n   " + "   ".join(str(i+1) for i in range(COLUMN_COUNT)))
    print("  +" + "---+" * COLUMN_COUNT)
    for r in range(ROW_COUNT):
        line = ""
        for c in range(COLUMN_COUNT):
            cell = board[r][c]
            if cell == EMPTY:
                line += f"{blue_square} | "
            elif cell == OUR_AI:
                line += "🔴 | "  
            else:
                line += "🟡 | "  
        print(" | " + line[:-2])
        print("  +" + "---+" * COLUMN_COUNT)

def winning_move(board, piece):
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            if all(board[r][c+i] == piece for i in range(4)):
                return True
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if all(board[r+i][c] == piece for i in range(4)):
                return True
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            if all(board[r+i][c+i] == piece for i in range(4)):
                return True
    for r in range(3, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            if all(board[r-i][c+i] == piece for i in range(4)):
                return True
    return False

def get_valid_locations(board):
    return [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]

# --- Feature Engineering from versions 2,3,4 ---
def extract_features(board):
    features = []
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            window = board[r][c:c+4]
            features.append(window.count(OUR_AI))
            features.append(window.count(OPP_AI))
    # Padding to fixed length 30 for consistent ML model input
    return features + [0] * (30 - len(features))

# --- Version 4 improvement: Dynamic training data based on board size ---
def create_training_data():
    win_board_our = [[EMPTY]*COLUMN_COUNT for _ in range(ROW_COUNT)]
    win_board_opp = [[EMPTY]*COLUMN_COUNT for _ in range(ROW_COUNT)]
    for i in range(4):
        win_board_our[i][0] = OUR_AI
        win_board_opp[i][1] = OPP_AI
    X_train = [extract_features(win_board_our), extract_features(win_board_opp)]
    y_train_nn = [1.0, 0.0]
    y_train_nb = [1, 0]
    return X_train, y_train_nn, y_train_nb

def train_models():
    X_train, y_train_nn, y_train_nb = create_training_data()
    # --- Version 3 & 4 usage of Neural Network ---
    mlp = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000)
    mlp.fit(X_train, y_train_nn) # type: ignore
    # --- Version 2 & 4 usage of Naive Bayes ---
    gnb = GaussianNB()
    gnb.fit(X_train, y_train_nb) # type: ignore
    return mlp, gnb

def evaluate_window(window, piece):
    score = 0
    opp_piece = OPP_AI if piece == OUR_AI else OUR_AI
    if window.count(piece) == 4:
        score += 100000
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 500
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 50
    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 900
    elif window.count(opp_piece) == 2 and window.count(EMPTY) == 2:
        score -= 100
    return score

# --- Enhanced heuristic incorporating positional weighting (from version 1,3) ---
def score_position(board, piece, mlp, gnb):
    score = 0
    center_col_idx = COLUMN_COUNT // 2
    center_col = [board[r][center_col_idx] for r in range(ROW_COUNT)]
    score += center_col.count(piece) * 15
    for offset in [-1, 1]:
        adj_idx = center_col_idx + offset
        if 0 <= adj_idx < COLUMN_COUNT:
            adj_col_vals = [board[r][adj_idx] for r in range(ROW_COUNT)]
            score += adj_col_vals.count(piece) * 8
    for r in range(ROW_COUNT):
        row = board[r]
        for c in range(COLUMN_COUNT - 3):
            score += evaluate_window(row[c:c+4], piece)
    for c in range(COLUMN_COUNT):
        col = [board[r][c] for r in range(ROW_COUNT)]
        for r in range(ROW_COUNT - 3):
            score += evaluate_window(col[r:r+4], piece)
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            score += evaluate_window([board[r+i][c+i] for i in range(4)], piece)
    for r in range(3, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            score += evaluate_window([board[r-i][c+i] for i in range(4)], piece)

    # --- Ensemble components: neural network and naive bayes model predictions ---
    feats = np.array(extract_features(board)).reshape(1, -1)
    nn_score = mlp.predict(feats)[0] * 1000
    nb_proba = gnb.predict_proba(feats)[0][1] * 1000
    # Example opponent probability adjustment placeholder
    opp_prob_adj = sum(0.5 for c in range(COLUMN_COUNT)) * 50
    combined = 0.5 * score + 0.3 * nn_score + 0.15 * nb_proba - 0.05 * opp_prob_adj
    return combined

def is_terminal_node(board):
    return winning_move(board, OUR_AI) or winning_move(board, OPP_AI) or len(get_valid_locations(board)) == 0

# --- Minimax with alpha-beta pruning including ensemble evaluation ---
def minimax(board, depth, alpha, beta, maximizingPlayer, mlp, gnb):
    valid_locations = get_valid_locations(board)
    terminal = is_terminal_node(board)
    if depth == 0 or terminal:
        if terminal:
            if winning_move(board, OUR_AI):
                return None, 10**9 + depth
            elif winning_move(board, OPP_AI):
                return None, -10**9 - depth
            else:
                return None, 0
        else:
            return None, score_position(board, OUR_AI, mlp, gnb)
    center = COLUMN_COUNT // 2
    ordered = sorted(valid_locations, key=lambda x: abs(x - center))
    if maximizingPlayer:
        value = -math.inf
        best_col = random.choice(ordered)
        for col in ordered:
            row = get_next_open_row(board, col)
            temp = deepcopy(board)
            drop_piece(temp, row, col, OUR_AI)
            new_score = minimax(temp, depth - 1, alpha, beta, False, mlp, gnb)[1]
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value
    else:
        value = math.inf
        best_col = random.choice(ordered)
        for col in ordered:
            row = get_next_open_row(board, col)
            temp = deepcopy(board)
            drop_piece(temp, row, col, OPP_AI)
            new_score = minimax(temp, depth - 1, alpha, beta, True, mlp, gnb)[1]
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value

def get_best_move(board, mlp, gnb):
    col, _ = minimax(board, 7, -math.inf, math.inf, True, mlp, gnb)
    return col if col is not None else random.choice(get_valid_locations(board))

# --- User board size selection ---
def get_board_size():
    print("Select board size:")
    print("1) 6x6")
    print("2) 6x7")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return 6, 6
        elif choice == "2":
            return 6, 7
        print("Invalid input. Please enter 1 or 2.")

def print_header():
    print("\nConnect 4 AI Ensemble - Our AI (🔴) vs Opponent (🟡)\n")

# --- Main game loop incorporating all version features ---
if __name__ == "__main__":
    print_header()
    rows, cols = get_board_size()
    board = create_board(rows, cols)  # Dynamic board size (Version 4 feature)
    print_board(board)
    mlp, gnb = train_models()         # Train ML models dynamically (Version 4)
    starter = input("Who plays first? (Our AI / Opponent): ").strip().lower()
    turn = OUR_AI if starter.startswith("our") else OPP_AI
    move_count = 0

    while True:
        valid_locations = get_valid_locations(board)
        if not valid_locations:
            print("Game drawn: board full.")
            break
        move_count += 1
        print(f"Move #{move_count}")
        if turn == OUR_AI:
            move = get_best_move(board, mlp, gnb)
            row = get_next_open_row(board, move)
            drop_piece(board, row, move, OUR_AI)
            print(f"Our AI plays column {move + 1}")
            print_board(board)
            if winning_move(board, OUR_AI):
                print("Our AI WINS!")
                break
        else:
            print(f"Valid columns: {[c + 1 for c in valid_locations]}")
            while True:
                try:
                    move_input = input(f"Enter opponent move (1-{cols}): ").strip()
                    move = int(move_input) - 1
                    if move not in valid_locations:
                        print("Invalid move, try again.")
                        continue
                    row = get_next_open_row(board, move)
                    drop_piece(board, row, move, OPP_AI)
                    print(f"Opponent plays column {move + 1}")
                    print_board(board)
                    if winning_move(board, OPP_AI):
                        print("Opponent WINS!")
                        break
                    break
                except ValueError:
                    print("Please enter a valid number.")
                except KeyboardInterrupt:
                    print("\nGame interrupted.")
                    sys.exit(0)
            if winning_move(board, OPP_AI):
                break
        turn = OPP_AI if turn == OUR_AI else OUR_AI
    print("\nGame over. Thanks for playing!")
