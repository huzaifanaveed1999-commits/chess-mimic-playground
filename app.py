import os
import shutil
import chess
from flask import Flask, request, jsonify, render_template
import torch

# Import our custom model loader
from model_loader import load_chess_model, evaluate_moves, ChessMimicModel

app = Flask(__name__)

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Application state
ACTIVE_MODEL = None
ACTIVE_MODEL_NAME = "None"
DEVICE = 'cpu'

def init_default_model():
    """
    Intelligently checks if the user's downloaded y4k2_mimic.pth exists,
    copies it to the local models folder, and loads it as active.
    """
    global ACTIVE_MODEL, ACTIVE_MODEL_NAME
    default_download_path = r"C:\Users\Osama\Downloads\y4k2_mimic.pth"
    local_path = os.path.join(MODELS_DIR, "y4k2_mimic.pth")
    
    # 1. Copy file if it exists in Downloads and not locally
    if os.path.exists(default_download_path):
        if not os.path.exists(local_path):
            try:
                shutil.copy(default_download_path, local_path)
                print(f"Copied {default_download_path} to {local_path}")
            except Exception as e:
                print(f"Error copying default model: {e}")
                
    # 2. Try loading local model
    if os.path.exists(local_path):
        try:
            ACTIVE_MODEL = load_chess_model(local_path, device=DEVICE)
            ACTIVE_MODEL_NAME = "y4k2_mimic.pth"
            print("Successfully loaded default model: y4k2_mimic.pth")
            return
        except Exception as e:
            print(f"Error loading local model: {e}")
            
    print("No default model loaded. Waiting for user upload...")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    global ACTIVE_MODEL_NAME, ACTIVE_MODEL
    has_model = ACTIVE_MODEL is not None
    model_size_mb = 0
    param_count = 0
    
    if has_model:
        local_path = os.path.join(MODELS_DIR, ACTIVE_MODEL_NAME)
        if os.path.exists(local_path):
            model_size_mb = round(os.path.getsize(local_path) / (1024 * 1024), 2)
        
        # Calculate parameter count
        param_count = sum(p.numel() for p in ACTIVE_MODEL.parameters())
        
    return jsonify({
        'has_model': has_model,
        'model_name': ACTIVE_MODEL_NAME,
        'model_size_mb': model_size_mb,
        'parameter_count': param_count
    })

@app.route('/api/upload', methods=['POST'])
def upload_model():
    global ACTIVE_MODEL, ACTIVE_MODEL_NAME
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if not file.filename.endswith('.pth'):
        return jsonify({'error': 'Invalid file type. Only PyTorch (.pth) models are allowed.'}), 400
        
    try:
        # Save uploaded file
        filename = file.filename
        dest_path = os.path.join(MODELS_DIR, filename)
        file.save(dest_path)
        
        # Verify it loads correctly
        loaded_model = load_chess_model(dest_path, device=DEVICE)
        
        # Update active model
        ACTIVE_MODEL = loaded_model
        ACTIVE_MODEL_NAME = filename
        
        param_count = sum(p.numel() for p in ACTIVE_MODEL.parameters())
        file_size_mb = round(os.path.getsize(dest_path) / (1024 * 1024), 2)
        
        return jsonify({
            'success': True,
            'message': f"Model '{filename}' uploaded and loaded successfully!",
            'model_name': filename,
            'model_size_mb': file_size_mb,
            'parameter_count': param_count
        })
    except Exception as e:
        # Cleanup file if loading failed
        if 'dest_path' in locals() and os.path.exists(dest_path):
            os.remove(dest_path)
        return jsonify({'error': f"Failed to load the uploaded model. Please check the model structure. Error: {str(e)}"}), 500

@app.route('/api/play', methods=['POST'])
def play():
    global ACTIVE_MODEL
    
    # Ensure a model is loaded
    if ACTIVE_MODEL is None:
        return jsonify({'error': 'No AI model is loaded. Please upload a .pth model file.'}), 400
        
    data = request.get_json() or {}
    fen = data.get('fen', chess.STARTING_FEN)
    temperature = float(data.get('temperature', 0.0))
    
    try:
        board = chess.Board(fen)
    except Exception as e:
        return jsonify({'error': f"Invalid FEN string: {str(e)}"}), 400
        
    if board.is_game_over():
        return jsonify({
            'game_over': True,
            'result': board.result(),
            'message': "The game is already over."
        })
        
    try:
        best_move, move_evals, heatmap_data = evaluate_moves(
            ACTIVE_MODEL, board, temperature=temperature, device=DEVICE
        )
        
        if best_move is None:
            return jsonify({
                'game_over': True,
                'result': board.result(),
                'message': "No legal moves available."
            })
            
        # Return move and detailed evaluations
        return jsonify({
            'success': True,
            'best_move': best_move.uci(),
            'san': board.san(best_move), # Standard algebraic notation (e.g. e4)
            'from_square': best_move.from_square,
            'to_square': best_move.to_square,
            'move_evals': move_evals, # Top legal moves with probabilities
            'heatmap': heatmap_data # 4096-logit map
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Inference execution failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Initialize the default model before launching
    init_default_model()
    # Run server locally
    app.run(host='127.0.0.1', port=5000, debug=True)
