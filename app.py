import os
import shutil
import chess
from flask import Flask, request, jsonify, render_template
import torch
import atexit
import chess.engine

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
STOCKFISH_ENGINE = None

def ensure_stockfish_installed():
    """
    Checks if Stockfish binary is present locally or in the system PATH.
    If not, downloads the official pre-compiled AVX2 binary from GitHub
    (Windows zip or Linux tar.xz) and extracts it locally.
    """
    import platform
    engines_dir = os.path.join(BASE_DIR, 'engines')
    os.makedirs(engines_dir, exist_ok=True)
    
    is_windows = platform.system() == 'Windows'
    binary_name = 'stockfish.exe' if is_windows else 'stockfish'
    stockfish_path = os.path.join(engines_dir, binary_name)
    
    # 1. First check system-wide path (e.g. if installed via apt-get on Linux)
    system_stockfish = shutil.which("stockfish")
    if not system_stockfish and not is_windows:
        for loc in ["/usr/games/stockfish", "/usr/bin/stockfish", "/usr/local/bin/stockfish"]:
            if os.path.exists(loc):
                system_stockfish = loc
                break
    if system_stockfish:
        print(f"Stockfish engine found globally in system PATH/directories: {system_stockfish}")
        return True
        
    # 2. Check local engines directory
    if os.path.exists(stockfish_path):
        print(f"Stockfish engine found locally at: {stockfish_path}")
        if not is_windows:
            os.chmod(stockfish_path, 0o755)
        return True
        
    print("Stockfish engine not found locally or globally. Initiating automatic download...")
    import urllib.request
    
    if is_windows:
        zip_url = "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64-avx2.zip"
        temp_file = os.path.join(engines_dir, "stockfish_temp.zip")
    else:
        zip_url = "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar.xz"
        temp_file = os.path.join(engines_dir, "stockfish_temp.tar.xz")
        
    try:
        print(f"Downloading Stockfish from {zip_url} ...")
        req = urllib.request.Request(
            zip_url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(temp_file, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            
        print("Download complete. Extracting files...")
        
        if is_windows:
            import zipfile
            temp_extract_dir = os.path.join(engines_dir, "temp_extract")
            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)
                
            found_exe = None
            for root, dirs, files in os.walk(temp_extract_dir):
                for file in files:
                    if file.lower().endswith('.exe') and 'stockfish' in file.lower():
                        found_exe = os.path.join(root, file)
                        break
                if found_exe:
                    break
                    
            if found_exe:
                shutil.copy(found_exe, stockfish_path)
                print(f"Stockfish engine successfully installed at: {stockfish_path}")
                success = True
            else:
                print("[ERROR] Could not find stockfish.exe within the downloaded zip.")
                success = False
                
            if os.path.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir)
        else:
            import tarfile
            temp_extract_dir = os.path.join(engines_dir, "temp_extract")
            with tarfile.open(temp_file, "r:xz") as tar_ref:
                tar_ref.extractall(temp_extract_dir)
                
            found_bin = None
            for root, dirs, files in os.walk(temp_extract_dir):
                for file in files:
                    # Linux binary usually doesn't have an extension
                    if 'stockfish' in file.lower() and not file.lower().endswith(('.txt', '.md', '.pdf')):
                        found_bin = os.path.join(root, file)
                        break
                if found_bin:
                    break
                    
            if found_bin:
                shutil.copy(found_bin, stockfish_path)
                os.chmod(stockfish_path, 0o755)
                print(f"Stockfish engine successfully installed at: {stockfish_path}")
                success = True
            else:
                print("[ERROR] Could not find stockfish binary within the downloaded tarball.")
                success = False
                
            if os.path.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir)
                
    except Exception as e:
        print(f"[ERROR] Failed to download or install Stockfish: {e}")
        success = False
        
    finally:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass
                
    return success

def init_stockfish_engine():
    """
    Triggers installation and opens a persistent UCI subprocess connection to Stockfish.
    """
    global STOCKFISH_ENGINE
    if not ensure_stockfish_installed():
        print("[WARNING] Stockfish downloader failed. Running in Pure Policy mode (tactical lookahead disabled).")
        return None
        
    try:
        import platform
        is_windows = platform.system() == 'Windows'
        system_stockfish = shutil.which("stockfish")
        if not system_stockfish and not is_windows:
            for loc in ["/usr/games/stockfish", "/usr/bin/stockfish", "/usr/local/bin/stockfish"]:
                if os.path.exists(loc):
                    system_stockfish = loc
                    break
        if system_stockfish:
            stockfish_executable = system_stockfish
        else:
            binary_name = 'stockfish.exe' if is_windows else 'stockfish'
            stockfish_executable = os.path.join(BASE_DIR, 'engines', binary_name)
            
        STOCKFISH_ENGINE = chess.engine.SimpleEngine.popen_uci(stockfish_executable)
        print(f"Stockfish Chess Engine successfully initialized as a persistent background process from: {stockfish_executable}")
        return STOCKFISH_ENGINE
    except Exception as e:
        print(f"[ERROR] Failed to start Stockfish subprocess: {e}")
        STOCKFISH_ENGINE = None
        return None

@atexit.register
def shutdown_engine():
    """
    Ensures that the Stockfish background subprocess is closed cleanly when Flask exits.
    """
    global STOCKFISH_ENGINE
    if STOCKFISH_ENGINE is not None:
        print("Closing Stockfish background process...")
        try:
            STOCKFISH_ENGINE.close()
        except Exception:
            pass

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

@app.after_request
def add_header(response):
    """
    Forces the browser to completely disable caching. This ensures that any template 
    or static file updates are instantly pulled without requiring aggressive manual reloads.
    """
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/api/status', methods=['GET'])
def get_status():
    global ACTIVE_MODEL_NAME, ACTIVE_MODEL, STOCKFISH_ENGINE
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
        'parameter_count': param_count,
        'stockfish_active': STOCKFISH_ENGINE is not None
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
    global ACTIVE_MODEL, STOCKFISH_ENGINE
    
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
        best_move, move_evals, heatmap_data, thinking_process = evaluate_moves(
            ACTIVE_MODEL, board, temperature=temperature, device=DEVICE, engine=STOCKFISH_ENGINE
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
            'heatmap': heatmap_data, # 4096-logit map
            'thinking_process': thinking_process # Rollout evaluation data
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Inference execution failed: {str(e)}"}), 500

if __name__ == '__main__':
    import os
    # Initialize the default model before launching
    init_default_model()
    # Initialize Stockfish Persistent Engine
    init_stockfish_engine()
    
    # Cloud dynamic port binding (Hugging Face Spaces uses port 7860 by default)
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    
    app.run(host=host, port=port, debug=False)
