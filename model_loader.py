import os
import chess
import torch
import torch.nn as nn
import numpy as np

class ChessResBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ChessMimicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_init = nn.Conv2d(17, 32, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        
        self.res_blocks = nn.Sequential(
            ChessResBlock(32),
            ChessResBlock(32)
        )
        
        self.conv_flat = nn.Conv2d(32, 32, kernel_size=1)
        self.bn_flat = nn.BatchNorm2d(32)
        
        self.fc1 = nn.Linear(32 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 4096)

    def forward(self, x):
        x = self.relu(self.bn_init(self.conv_init(x)))
        x = self.res_blocks(x)
        x = self.relu(self.bn_flat(self.conv_flat(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Converts a python-chess Board state into a 17x8x8 PyTorch float tensor.
    """
    tensor = np.zeros((17, 8, 8), dtype=np.float32)
    
    # Map piece types to plane offsets
    piece_to_plane = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            plane = piece_to_plane[piece.piece_type]
            if piece.color == chess.BLACK:
                plane += 6
            
            # Map ranks: Rank 8 (top, 7 in chess) is index 0 in tensor row
            # Rank 1 (bottom, 0 in chess) is index 7 in tensor row
            file_idx = chess.square_file(sq)
            rank_idx = chess.square_rank(sq)
            row = 7 - rank_idx
            col = file_idx
            tensor[plane, row, col] = 1.0
            
    # Plane 12: Turn indicator (all 1s if White's turn, 0s if Black's turn)
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0
        
    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0
        
    # Return as a PyTorch batch of size 1 (1, 17, 8, 8)
    return torch.tensor(tensor).unsqueeze(0)

def get_board_material(board: chess.Board) -> int:
    """
    Computes absolute piece balance: White is positive, Black is negative.
    """
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    val = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            score = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                val += score
            else:
                val -= score
    return val

def check_blunder(board: chess.Board, move: chess.Move) -> bool:
    """
    Checks if a move walks straight into checkmate or hands the opponent 
    an immediate material advantage of more than 1.5 pawns (e.g. hanging a piece).
    """
    active_color = board.turn
    
    # Material before the move
    mat_before = get_board_material(board)
    if active_color == chess.BLACK:
        mat_before = -mat_before
        
    # Simulate move
    board.push(move)
    
    if board.is_checkmate():
        # Checking opponent is a winning move, not a blunder!
        board.pop()
        return False
        
    opp_legal = list(board.legal_moves)
    if not opp_legal:
        board.pop()
        return False
        
    worst_net_loss = 0
    
    for opp_move in opp_legal:
        board.push(opp_move)
        
        # Check if we get checkmated in response
        if board.is_checkmate():
            board.pop()
            board.pop()
            return True # Walking straight into checkmate
            
        mat_after = get_board_material(board)
        if active_color == chess.BLACK:
            mat_after = -mat_after
            
        net_change = mat_after - mat_before
        
        if net_change < worst_net_loss:
            worst_net_loss = net_change
            
        board.pop()
        
    board.pop()
    
    # If opponent can capture a piece worth > 1.5 pawns for free in their next turn
    if worst_net_loss < -150:
        return True
        
    return False

def load_chess_model(model_path: str, device: str = 'cpu') -> ChessMimicModel:
    """
    Loads weights into the ChessMimicModel from a .pth file.
    """
    model = ChessMimicModel()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def evaluate_moves(model: ChessMimicModel, board: chess.Board, temperature: float = 0.0, device: str = 'cpu'):
    """
    Evaluates all legal moves for the current board state using the model.
    Returns:
        best_move: chess.Move
        move_evals: list of dicts with keys (uci, probability, confidence, logit)
        all_logits_map: dict of all 4096 logits for heatmap visualization
    """
    # 1. Convert board to tensor
    inp = board_to_tensor(board).to(device)
    
    # 2. Run model forward pass
    with torch.no_grad():
        logits = model(inp).squeeze(0).cpu().numpy() # shape: (4096,)
    
    # Create complete logit map for heatmap display
    heatmap_data = {}
    for from_sq in range(64):
        heatmap_data[from_sq] = {}
        for to_sq in range(64):
            idx = from_sq * 64 + to_sq
            heatmap_data[from_sq][to_sq] = float(logits[idx])
            
    # 3. Filter legal moves and run Blunder Guard Check
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, [], heatmap_data
        
    move_logits = []
    blunder_flags = []
    
    for move in legal_moves:
        idx = move.from_square * 64 + move.to_square
        logit = float(logits[idx])
        
        # Check blunder guard
        is_blunder = check_blunder(board, move)
        if is_blunder:
            logit -= 100.0  # Heavier penalty to override neural logits
            
        move_logits.append(logit)
        blunder_flags.append(is_blunder)
        
    move_logits = np.array(move_logits, dtype=np.float32)
    
    # 4. Softmax calculation over legal moves
    # Subtract max for numerical stability
    exp_logits = np.exp(move_logits - np.max(move_logits))
    probs = exp_logits / np.sum(exp_logits)
    
    # Organize evaluated moves
    move_evals = []
    for move, logit, prob, bl_flag in zip(legal_moves, move_logits, probs, blunder_flags):
        move_evals.append({
            'uci': move.uci(),
            'from_square': move.from_square,
            'to_square': move.to_square,
            'logit': float(logit),
            'probability': float(prob),
            'is_blunder': bl_flag
        })
        
    # Sort moves by probability descending
    move_evals.sort(key=lambda x: x['probability'], reverse=True)
    
    # 5. Select move based on temperature
    if temperature <= 0.01:
        # Greedy choice: highest probability move
        best_move_dict = move_evals[0]
    else:
        # Sample based on temperature scaled probabilities
        scaled_logits = move_logits / temperature
        exp_scaled = np.exp(scaled_logits - np.max(scaled_logits))
        scaled_probs = exp_scaled / np.sum(exp_scaled)
        
        chosen_idx = np.random.choice(len(legal_moves), p=scaled_probs)
        best_move_dict = next(m for m in move_evals if m['uci'] == legal_moves[chosen_idx].uci())
        
    best_move = chess.Move.from_uci(best_move_dict['uci'])
    
    return best_move, move_evals, heatmap_data
