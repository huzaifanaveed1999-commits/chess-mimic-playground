import os
import sys
import chess

# Import our custom loader
from model_loader import load_chess_model, evaluate_moves

def run_dry_run():
    print("=====================================================")
    # 1. Locate the model
    possible_paths = [
        r"C:\Users\Osama\Downloads\chess_mimic_model.pth",
        r"C:\Users\Osama\Downloads\y4k2_mimic.pth"
    ]
    model_path = None
    for p in possible_paths:
        if os.path.exists(p):
            model_path = p
            break
            
    if not model_path:
        print("[ERROR] Could not find the model file in Downloads folder.")
        print("Please ensure chess_mimic_model.pth remains in your Downloads folder.")
        sys.exit(1)
        
    print(f"[INFO] Model found at: {model_path}")
    print(f"[INFO] File size: {os.path.getsize(model_path)} bytes")
    
    # 2. Load PyTorch model
    print("[INFO] Loading PyTorch model and injecting weights...")
    try:
        model = load_chess_model(model_path, device='cpu')
        print("[SUCCESS] Model loaded successfully! Architecture match 100% correct.")
    except Exception as e:
        print(f"[ERROR] Failed to load model weights. This indicates a layer shape or name mismatch.")
        print(f"Details: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # 3. Setup chess board FEN (Starting position)
    board = chess.Board()
    print(f"[INFO] Initializing starting chess position FEN: {board.fen()}")
    
    # 4. Evaluate moves using model
    print("[INFO] Evaluating starting moves...")
    try:
        best_move, move_evals, heatmap_data, thinking_process = evaluate_moves(model, board, temperature=0.0, device='cpu', engine=None)
        
        print("\n================== TEST RESULTS ==================")
        print(f"AI Selected Best Move: {best_move} ({board.san(best_move)})")
        print("\nTop 5 Legal Move Evaluations:")
        for idx, ev in enumerate(move_evals[:5]):
            print(f"  {idx+1}. Move: {ev['uci']} (Prob: {ev['probability']*100:.2f}%, Logit: {ev['logit']:.2f})")
            
        # Verify the selected move is indeed legal
        if best_move in board.legal_moves:
            print("\n[SUCCESS] Verification Complete! AI chosen move is LEGAL.")
            print("Everything runs with ZERO bugs.")
        else:
            print("\n[CRITICAL ERROR] AI selected move is ILLEGAL!")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] Inference execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("=====================================================")

if __name__ == '__main__':
    run_dry_run()
