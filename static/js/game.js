// High-resolution pixel-perfect Chess.com Neo theme pieces loaded directly from the official Chess.com CDN
const PIECE_SVGS = {
    w: {
        p: '<img class="chess-piece white" src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wp.png" alt="wp" />',
        n: '<img class="chess-piece white" src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wn.png" alt="wn" />',
        b: '<img class="chess-piece white" src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wb.png" alt="wb" />',
        r: '<img class="chess-piece white" src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wr.png" alt="wr" />',
        q: '<img class="chess-piece white" src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wq.png" alt="wq" />',
        k: '<img class="chess-piece white" src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wk.png" alt="wk" />'
    },
    b: {
        p: '<img class="chess-piece black" src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bp.png" alt="bp" />',
        n: '<img class="chess-piece black" src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bn.png" alt="bn" />',
        b: '<img class="chess-piece black" src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bb.png" alt="bb" />',
        r: '<img class="chess-piece black" src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/br.png" alt="br" />',
        q: '<img class="chess-piece black" src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bq.png" alt="bq" />',
        k: '<img class="chess-piece black" src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bk.png" alt="bk" />'
    }
};

// Application State
let game = new Chess();
let selectedSquare = null;
let boardFlipped = false; // true if playing black (AI is white)
let gameMode = 'player-vs-ai-black'; // 'player-vs-ai-black', 'player-vs-ai-white', 'ai-vs-ai'
let aiControlMode = 'auto'; // 'auto' (AI moves automatically), 'suggest' (AI suggests, user clicks to play)
let temperature = 0.0;
let showHeatmap = true;
let aiThinking = false;
let currentHeatmapData = null;
let userPrefersThoughtsOpen = false; // Persistent user choice to show/hide the drawer
let analysisMode = false; // true if player is analyzing board without AI moving
let currentReplayIndex = -1; // -1 means active live game, otherwise index of reviewed move

// DOM Elements
const boardGrid = document.getElementById('chessboard');
const heatmapOverlay = document.getElementById('heatmap-overlay');
const statusMessage = document.getElementById('game-status-message');
const statusIcon = document.getElementById('game-status-icon');
const probList = document.getElementById('probability-list');
const moveCounter = document.getElementById('move-counter');
const moveHistoryList = document.getElementById('move-history-list');
const modelSelector = document.getElementById('model-selector');
const modelSizeText = document.getElementById('model-size-text');
const modelParamsText = document.getElementById('model-params-text');
const tempSlider = document.getElementById('temp-slider');
const tempVal = document.getElementById('temp-val');
const heatmapToggle = document.getElementById('heatmap-toggle');

// Modal Elements
const modalOverlay = document.getElementById('game-over-modal');
const modalTitle = document.getElementById('modal-title');
const modalDesc = document.getElementById('modal-desc');
const btnModalRestart = document.getElementById('btn-modal-restart');
const btnModalCopyFen = document.getElementById('btn-modal-copy-fen');
const btnModalCopyPgn = document.getElementById('btn-modal-copy-pgn');
const btnModalAnalyze = document.getElementById('btn-modal-analyze');
const btnModalDownloadPgn = document.getElementById('btn-modal-download-pgn');

// Action Buttons
const btnUndo = document.getElementById('btn-undo');
const btnReset = document.getElementById('btn-reset');
const btnCopyFen = document.getElementById('btn-copy-fen');
const dropzone = document.getElementById('upload-dropzone');
const fileInput = document.getElementById('model-file-input');

// Thinking Process Panel DOM Elements
const btnToggleThoughts = document.getElementById('btn-toggle-thoughts');
const btnCloseDrawer = document.getElementById('btn-close-drawer');
const thinkingDrawer = document.getElementById('thinking-drawer');
const thinkingInitialEval = document.getElementById('thinking-initial-eval');
const thinkingTableBody = document.getElementById('thinking-table-body');

// Initialize the Application
window.addEventListener('DOMContentLoaded', () => {
    buildBoardSquares();
    fetchModelsList();
    setupEventListeners();
    startNewGame();
});

// Build standard 8x8 squares inside the container
function buildBoardSquares() {
    boardGrid.innerHTML = '';
    heatmapOverlay.innerHTML = '';
    
    // Rows 0 to 7. If flipped, we reverse drawing order
    const rowOrder = boardFlipped ? [0, 1, 2, 3, 4, 5, 6, 7] : [7, 6, 5, 4, 3, 2, 1, 0];
    const colOrder = boardFlipped ? [7, 6, 5, 4, 3, 2, 1, 0] : [0, 1, 2, 3, 4, 5, 6, 7];
    
    rowOrder.forEach(r => {
        colOrder.forEach(c => {
            const sqIndex = r * 8 + c;
            const isLight = (r + c) % 2 !== 0; // standard algebraic lighting
            
            // Grid cell
            const sqEl = document.createElement('div');
            sqEl.className = `square ${isLight ? 'light' : 'dark'}`;
            sqEl.dataset.square = sqIndex;
            sqEl.id = `sq-${sqIndex}`;
            
            // Event listener for moves
            sqEl.addEventListener('click', () => handleSquareClick(sqIndex));
            boardGrid.appendChild(sqEl);
            
            // Heatmap overlay cell
            const heatEl = document.createElement('div');
            heatEl.className = 'heatmap-cell';
            heatEl.id = `heat-${sqIndex}`;
            heatmapOverlay.appendChild(heatEl);
        });
    });
    
    updateBoardLabels();
}

// Draw proper algebraic rank and file headers based on flip
function updateBoardLabels() {
    const rankLabelEl = document.querySelector('.board-labels.ranks');
    const fileLabelEl = document.querySelector('.board-labels.files');
    
    const ranks = ['1', '2', '3', '4', '5', '6', '7', '8'];
    const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    
    if (!boardFlipped) {
        ranks.reverse();
    } else {
        files.reverse();
    }
    
    rankLabelEl.innerHTML = ranks.map(r => `<span>${r}</span>`).join('');
    fileLabelEl.innerHTML = files.map(f => `<span>${f}</span>`).join('');
}

// Start a fresh match
function startNewGame() {
    game = new Chess();
    selectedSquare = null;
    currentHeatmapData = null;
    analysisMode = false;
    currentReplayIndex = -1;
    clearHighlights();
    updateBoardPieces();
    updateMoveHistory();
    updateStatusMessage();
    clearAIAnalysis();
    clearThinkingProcess();
    
    // Hide game-over modal
    modalOverlay.style.display = 'none';
    
    // If game mode is AI vs AI or AI plays White, trigger AI immediately
    checkAITurn();
}

// Render actual pieces on the board based on chess.js state
function updateBoardPieces(customGame) {
    if (!customGame) customGame = game;
    // Clear all pieces
    document.querySelectorAll('.square').forEach(sq => {
        const piece = sq.querySelector('.chess-piece');
        if (piece) piece.remove();
    });
    
    // Loop through chess.js board representation
    const boardState = customGame.board(); // 8x8 nested array of pieces
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const piece = boardState[r][c];
            if (piece) {
                // python-chess rank mapping: chess.js board is row 0 = rank 8, row 7 = rank 1
                const chessSquareIdx = (7 - r) * 8 + c;
                const sqEl = document.getElementById(`sq-${chessSquareIdx}`);
                if (sqEl) {
                    const svgHtml = PIECE_SVGS[piece.color][piece.type];
                    sqEl.insertAdjacentHTML('beforeend', svgHtml);
                }
            }
        }
    }
}

// Handle clicking on squares
function handleSquareClick(sqIndex) {
    if (aiThinking) return;
    
    // If game is over and NOT in analysisMode, block moves
    if (game.game_over() && !analysisMode && currentReplayIndex === -1) return;
    
    // Determine active game state (either replayed past state or live state)
    let activeGame = game;
    if (currentReplayIndex !== -1) {
        activeGame = new Chess();
        const history = game.history({ verbose: true });
        for (let i = 0; i <= currentReplayIndex; i++) {
            activeGame.move(history[i]);
        }
    }
    
    // Check if player turn matches current board turn
    const turn = activeGame.turn();
    const isWhiteTurn = turn === 'w';
    
    if (!analysisMode && currentReplayIndex === -1) {
        if (gameMode === 'ai-vs-ai') return;
        if (gameMode === 'player-vs-ai-black' && !isWhiteTurn) return;
        if (gameMode === 'player-vs-ai-white' && isWhiteTurn) return;
    }
    
    const clickedPiece = activeGame.get(squareIdxToUci(sqIndex));
    
    // 1. If clicking our own piece, select it and show hints
    if (clickedPiece && clickedPiece.color === turn) {
        selectedSquare = sqIndex;
        clearHighlights();
        
        const sqEl = document.getElementById(`sq-${sqIndex}`);
        sqEl.classList.add('selected');
        
        // Show legal hints based on active replayed game
        showLegalMoveHintsForGame(sqIndex, activeGame);
        
        // Highlight destination probabilities in heatmap specifically for this piece (only if not replaying)
        if (currentHeatmapData && showHeatmap && currentReplayIndex === -1) {
            renderHeatmapForSelectedPiece(sqIndex);
        }
        return;
    }
    
    // 2. If a piece is already selected, try to execute move to target square
    if (selectedSquare !== null) {
        const fromUci = squareIdxToUci(selectedSquare);
        const toUci = squareIdxToUci(sqIndex);
        
        // Check if move is legal on the active board
        const moves = activeGame.moves({ square: fromUci, verbose: true });
        const legalMove = moves.find(m => m.to === toUci);
        
        if (legalMove) {
            // If we were replaying a past position, truncate the main game history to this point!
            if (currentReplayIndex !== -1) {
                game = activeGame; // Truncate and commit history up to reviewed move
                currentReplayIndex = -1;
                analysisMode = true; // Engage self-analysis on the new branch
            }
            
            // Executing standard move
            let moveObj = {
                from: fromUci,
                to: toUci
            };
            
            // Auto-promote to Queen for simplicity and bug prevention
            if (legalMove.flags.includes('p')) {
                moveObj.promotion = 'q';
            }
            
            game.move(moveObj);
            
            // Reset selection, redraw board, update logs
            selectedSquare = null;
            clearHighlights();
            updateBoardPieces();
            updateMoveHistory();
            updateStatusMessage();
            
            // Check if game is over or hand over to AI
            if (game.game_over()) {
                if (!analysisMode) {
                    handleGameOver();
                } else {
                    statusMessage.textContent = "Checkmate or Draw reached in Self Analysis.";
                    statusIcon.setAttribute('data-lucide', 'award');
                    lucide.createIcons();
                }
            } else {
                if (!analysisMode) {
                    checkAITurn();
                }
            }
        } else {
            // Clicked an invalid/empty square, deselect
            selectedSquare = null;
            clearHighlights();
            if (currentHeatmapData && showHeatmap && currentReplayIndex === -1) {
                renderHeatmapCumulative();
            }
        }
    }
}

// Draw subtle glowing dots on legal squares for a specific game instance
function showLegalMoveHintsForGame(sqIndex, customGame) {
    if (!customGame) customGame = game;
    const fromUci = squareIdxToUci(sqIndex);
    const legalMoves = customGame.moves({ square: fromUci, verbose: true });
    
    legalMoves.forEach(m => {
        const targetIdx = uciToSquareIdx(m.to);
        const targetSq = document.getElementById(`sq-${targetIdx}`);
        if (targetSq) {
            const hasOpponent = targetSq.querySelector('.chess-piece');
            if (hasOpponent) {
                // Glow outline for captures
                const capEl = document.createElement('div');
                capEl.className = 'move-hint-capture';
                targetSq.appendChild(capEl);
            } else {
                // Smooth central dot
                const dotEl = document.createElement('div');
                dotEl.className = 'move-hint-dot';
                targetSq.appendChild(dotEl);
            }
        }
    });
}

// Clear temporary highlight rings, hints, selection borders
function clearHighlights() {
    document.querySelectorAll('.square').forEach(sq => {
        sq.classList.remove('selected');
        const hint = sq.querySelector('.move-hint-dot');
        if (hint) hint.remove();
        const cap = sq.querySelector('.move-hint-capture');
        if (cap) cap.remove();
    });
}

// Trigger AI step if it matches current turn conditions
function checkAITurn() {
    if (analysisMode || game.game_over()) return;
    
    const turn = game.turn();
    const isAITurn = 
        gameMode === 'ai-vs-ai' || 
        (gameMode === 'player-vs-ai-black' && turn === 'b') || 
        (gameMode === 'player-vs-ai-white' && turn === 'w');
        
    if (isAITurn) {
        triggerAIMove();
    }
}

// Connect to Flask play endpoint
function triggerAIMove() {
    aiThinking = true;
    updateStatusMessage();
    
    // Add pulsing border to active player avatar
    document.getElementById('player-white-info').classList.toggle('active', game.turn() === 'w');
    document.getElementById('player-black-info').classList.toggle('active', game.turn() === 'b');
    
    const fen = game.fen();
    
    fetch('/api/play', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            fen: fen,
            temperature: temperature
        })
    })
    .then(res => {
        if (!res.ok) {
            return res.json().then(err => { throw new Error(err.error || 'Server error') });
        }
        return res.json();
    })
    .then(data => {
        if (data.game_over) {
            handleGameOver();
            return;
        }
        
        currentHeatmapData = data.heatmap;
        
        // Render detailed AI predictions dashboard
        renderAIAnalysis(data.move_evals);
        
        // Render Stockfish Hybrid rollout thinking process drawer
        renderAIThinkingProcess(data.thinking_process);
        
        // Render Heatmap glows
        if (showHeatmap) {
            renderHeatmapCumulative();
        }
        
        // Decide how to proceed based on AI Control Mode
        if (aiControlMode === 'auto') {
            // Auto Play Mode: Execute the AI's best move automatically
            const fromSquare = data.best_move.substring(0, 2);
            const toSquare = data.best_move.substring(2, 4);
            const promotion = data.best_move.length > 4 ? data.best_move.charAt(4) : undefined;
            
            let moveObj = {
                from: fromSquare,
                to: toSquare
            };
            if (promotion) moveObj.promotion = promotion;
            
            let moveResult = game.move(moveObj);
            
            if (moveResult === null) {
                console.error("AI returned illegal move:", data.best_move);
                aiThinking = false;
                statusMessage.textContent = "AI predicted an illegal move: " + data.best_move;
                return;
            }
            
            // Redraw board, updates
            updateBoardPieces();
            updateMoveHistory();
            
            // Faint flash highlight on the source/destination squares played by AI
            animateAIMoveSquareGlow(data.from_square, data.to_square);
            
            aiThinking = false;
            updateStatusMessage();
            
            // Check game status or trigger next AI turn
            if (game.game_over()) {
                handleGameOver();
            } else {
                // Keep playing if AI vs AI
                if (gameMode === 'ai-vs-ai') {
                    setTimeout(checkAITurn, 800); // 800ms delay for human visual pacing
                } else {
                    checkAITurn();
                }
            }
        } else {
            // Suggest & Select Mode: Do NOT make the move, let the user select it!
            aiThinking = false;
            
            // Highlight the starting square of the AI's top suggestion
            animateAIMoveSquareGlow(data.from_square, data.from_square);
            
            // Update status message telling user to click one of the suggested moves on the left
            statusMessage.textContent = "AI suggestions loaded! Click any suggestion in the left panel to execute.";
            statusIcon.setAttribute('data-lucide', 'help-circle');
            lucide.createIcons();
        }
    })
    .catch(err => {
        console.error("AI Error:", err);
        aiThinking = false;
        statusMessage.textContent = `Error: ${err.message}. Make sure a PyTorch model is loaded.`;
        statusIcon.setAttribute('data-lucide', 'alert-circle');
        lucide.createIcons();
    });
}

// Light up the squares the AI just moved to/from
function animateAIMoveSquareGlow(fromIdx, toIdx) {
    const fromSq = document.getElementById(`sq-${fromIdx}`);
    const toSq = document.getElementById(`sq-${toIdx}`);
    
    if (fromSq && toSq) {
        fromSq.classList.add('selected');
        toSq.classList.add('selected');
        
        setTimeout(() => {
            fromSq.classList.remove('selected');
            toSq.classList.remove('selected');
        }, 1500);
    }
}

// Handle display of AI thought probability logs
function renderAIAnalysis(evals) {
    probList.innerHTML = '';
    
    if (!evals || evals.length === 0) {
        probList.innerHTML = '<div class="empty-list-placeholder">No evaluations found.</div>';
        return;
    }
    
    // Take top 5 legal moves
    const topEvals = evals.slice(0, 5);
    
    topEvals.forEach(ev => {
        const uci = ev.uci;
        const prob = (ev.probability * 100).toFixed(1);
        
        // Create representation in standard notation or clean squares
        const fromSq = uci.substring(0, 2);
        const toSq = uci.substring(2, 4);
        
        const row = document.createElement('div');
        row.className = `prob-row ${aiControlMode === 'suggest' ? 'clickable' : ''}`;
        
        if (aiControlMode === 'suggest') {
            row.setAttribute('onclick', `makeSuggestedMove('${uci}')`);
            row.setAttribute('title', `Play AI suggested move: ${fromSq} to ${toSq}`);
        }
        
        row.innerHTML = `
            <div class="prob-meta">
                <span class="prob-move">
                    <i data-lucide="corner-down-right" style="width:12px;height:12px;"></i>
                    ${fromSq} <span class="prob-move-san">to ${toSq}</span>
                </span>
                <span class="prob-pct">${prob}%</span>
            </div>
            <div class="prob-bar-container">
                <div class="prob-bar-fill" style="width: 0%;"></div>
            </div>
        `;
        probList.appendChild(row);
        
        // Trigger smooth fill animation
        setTimeout(() => {
            const fill = row.querySelector('.prob-bar-fill');
            if (fill) fill.style.width = `${prob}%`;
        }, 50);
    });
    
    lucide.createIcons();
}

// Global selection function for suggested moves (Suggest & Select Mode)
window.makeSuggestedMove = function(uci) {
    if (aiThinking || game.game_over()) return;
    
    const fromSquare = uci.substring(0, 2);
    const toSquare = uci.substring(2, 4);
    const promotion = uci.length > 4 ? uci.charAt(4) : undefined;
    
    let moveObj = {
        from: fromSquare,
        to: toSquare
    };
    if (promotion) moveObj.promotion = promotion;
    
    let moveResult = game.move(moveObj);
    
    if (moveResult === null) {
        console.error("Failed to execute clicked suggested move:", uci);
        return;
    }
    
    // Clear selection highlights and redraw pieces
    selectedSquare = null;
    clearHighlights();
    updateBoardPieces();
    updateMoveHistory();
    clearHeatmap();
    clearAIAnalysis();
    updateStatusMessage();
    
    // Animate move glow
    const fromIdx = uciToSquareIdx(fromSquare);
    const toIdx = uciToSquareIdx(toSquare);
    animateAIMoveSquareGlow(fromIdx, toIdx);
    
    // Hand turn check
    if (game.game_over()) {
        handleGameOver();
    } else {
        checkAITurn();
    }
};

// Render the 4-ply Stockfish hybrid thought visualizer
function renderAIThinkingProcess(data) {
    if (!thinkingTableBody || !thinkingInitialEval) return;
    
    if (!data || !data.active) {
        thinkingTableBody.innerHTML = `
            <tr>
                <td colspan="5" class="empty-table-placeholder">
                    Stockfish engine is currently offline or not initialized.
                </td>
            </tr>
        `;
        thinkingInitialEval.textContent = "0.00";
        return;
    }
    
    // Set initial position evaluation score
    thinkingInitialEval.textContent = data.initial_score;
    
    // Clear previous rows
    thinkingTableBody.innerHTML = '';
    
    // Iterate and populate candidate rows
    data.candidates.forEach(cand => {
        const uci = cand.uci;
        const san = cand.san;
        const probPct = (cand.nn_probability * 100).toFixed(1);
        const finalEval = cand.final_score;
        const evalDrop = cand.eval_drop;
        const isSelected = cand.is_selected;
        
        // Generate styled labels for the rollout move chain
        let rolloutHtml = '';
        if (cand.rollout_sans && cand.rollout_sans.length > 0) {
            cand.rollout_sans.forEach((m_san, idx) => {
                const stepNum = idx + 1;
                rolloutHtml += `<span class="rollout-tag rollout-header">Ply ${stepNum}</span>`;
                rolloutHtml += `<span class="rollout-tag">${m_san}</span>`;
            });
        } else {
            rolloutHtml = '<span style="color: var(--text-secondary);">No subsequent moves (Checkmate/Draw reached)</span>';
        }
        
        // Highlight negative drop (good!) or heavy drop (blunder!)
        const dropRaw = cand.eval_drop_raw;
        let dropStyle = '';
        if (dropRaw > 150) {
            dropStyle = 'style="color: var(--accent-coral); font-weight: 700;"'; // heavy blunder!
        } else if (dropRaw <= 10) {
            dropStyle = 'style="color: #10b981; font-weight: 700;"'; // optimal!
        }
        
        const row = document.createElement('tr');
        row.className = isSelected ? 'row-selected' : 'row-eliminated';
        
        row.innerHTML = `
            <td>
                <span class="prob-move" style="font-weight:800;">${san}</span>
                ${isSelected ? '<span class="badge selected" style="margin-left:6px;">Played</span>' : ''}
            </td>
            <td style="font-family: monospace; font-weight: 700; color: var(--accent-purple);">${probPct}%</td>
            <td>${rolloutHtml}</td>
            <td style="font-family: monospace; font-weight: 700;">${finalEval}</td>
            <td style="font-family: monospace;" ${dropStyle}>${evalDrop}</td>
        `;
        
        thinkingTableBody.appendChild(row);
    });
    
    // Automatically pop open the drawer ONLY if the user prefers it to be open
    if (userPrefersThoughtsOpen) {
        if (!thinkingDrawer.classList.contains('show')) {
            thinkingDrawer.classList.add('show');
        }
        if (btnToggleThoughts) {
            btnToggleThoughts.innerHTML = '<i data-lucide="eye-off"></i> Hide AI Thoughts';
            lucide.createIcons();
        }
    } else {
        // Keep it closed
        if (thinkingDrawer.classList.contains('show')) {
            thinkingDrawer.classList.remove('show');
        }
        if (btnToggleThoughts) {
            btnToggleThoughts.innerHTML = '<i data-lucide="brain"></i> Show AI Thoughts';
            lucide.createIcons();
        }
    }
}

// Clear the thinking process drawer back to empty state
function clearThinkingProcess() {
    if (thinkingTableBody) {
        thinkingTableBody.innerHTML = `
            <tr>
                <td colspan="5" class="empty-table-placeholder">No rollout data calculated. Play a move!</td>
            </tr>
        `;
    }
    if (thinkingInitialEval) {
        thinkingInitialEval.textContent = "0.00";
    }
}

// Clear evaluation logs
function clearAIAnalysis() {
    probList.innerHTML = '<div class="empty-list-placeholder">No active computations. Make a move!</div>';
}

// Render cumulative starting square confidence heatmap
function renderHeatmapCumulative() {
    if (!currentHeatmapData) return;
    
    // Clear previous cell attributes
    document.querySelectorAll('.heatmap-cell').forEach(cell => {
        cell.style.removeProperty('--prob-val');
        cell.classList.remove('target-active');
    });
    
    // For general display, we find the maximum logit across all 4096 values
    // to normalize the scale between 0.0 and 0.8 opacity.
    let maxLogit = -9999;
    let minLogit = 9999;
    
    // 1. Gather all starting square probabilities by summing their destination logits
    // or just taking their max logit.
    const startWeights = new Array(64).fill(0);
    for (let from_sq = 0; from_sq < 64; from_sq++) {
        let max_val = -9999;
        for (let to_sq = 0; to_sq < 64; to_sq++) {
            const v = currentHeatmapData[from_sq][to_sq];
            if (v > max_val) max_val = v;
        }
        startWeights[from_sq] = max_val;
    }
    
    const maxWeight = Math.max(...startWeights);
    const minWeight = Math.min(...startWeights);
    const weightRange = (maxWeight - minWeight) || 1.0;
    
    // 2. Draw heat mapping
    for (let sq = 0; sq < 64; sq++) {
        const normalized = (startWeights[sq] - minWeight) / weightRange;
        // Cap opacity at 0.6 for aesthetic visibility
        const opacity = Math.max(0, normalized * 0.6);
        const cell = document.getElementById(`heat-${sq}`);
        if (cell) {
            cell.style.setProperty('--prob-val', opacity);
        }
    }
}

// Render destination squares specifically for selected piece
function renderHeatmapForSelectedPiece(fromSq) {
    if (!currentHeatmapData) return;
    
    // Clear previous cell attributes
    document.querySelectorAll('.heatmap-cell').forEach(cell => {
        cell.style.removeProperty('--prob-val');
        cell.classList.add('target-active'); // Switch colors to cyan-blue
    });
    
    // Get all 64 target square logits for this specific starting square
    const targets = [];
    for (let toSq = 0; toSq < 64; toSq++) {
        targets.push(currentHeatmapData[fromSq][toSq]);
    }
    
    const maxVal = Math.max(...targets);
    const minVal = Math.min(...targets);
    const valRange = (maxVal - minVal) || 1.0;
    
    for (let toSq = 0; toSq < 64; toSq++) {
        const val = currentHeatmapData[fromSq][toSq];
        const normalized = (val - minVal) / valRange;
        const opacity = Math.max(0, normalized * 0.6);
        
        const cell = document.getElementById(`heat-${toSq}`);
        if (cell) {
            cell.style.setProperty('--prob-val', opacity);
        }
    }
}

// Wipe heatmap display clean
function clearHeatmap() {
    document.querySelectorAll('.heatmap-cell').forEach(cell => {
        cell.style.removeProperty('--prob-val');
        cell.classList.remove('target-active');
    });
}

// Update game status banner
function updateStatusMessage() {
    if (aiThinking) {
        statusMessage.textContent = "Mimic Neural Net is thinking... Running PyTorch GPU/CPU inference.";
        statusIcon.setAttribute('data-lucide', 'brain');
    } else if (analysisMode) {
        statusIcon.setAttribute('data-lucide', 'brain-circuit');
        if (game.game_over()) {
            statusMessage.textContent = "Match concluded in Self Analysis. Click 'Reset' to play a new game!";
        } else {
            const turn = game.turn();
            const turnName = turn === 'w' ? 'White' : 'Black';
            statusMessage.textContent = `Self Analysis Active: Play moves for both sides freely. Current turn: ${turnName}`;
        }
    } else if (game.game_over()) {
        statusIcon.setAttribute('data-lucide', 'trophy');
        if (game.in_checkmate()) {
            const winner = game.turn() === 'w' ? 'Black (Mimic AI)' : 'White (Player)';
            statusMessage.textContent = `Checkmate! ${winner} has won the match.`;
        } else if (game.in_draw()) {
            statusMessage.textContent = "Draw! The match ended in a draw (Stalemate/Repetition/50-move).";
        }
    } else {
        const turn = game.turn();
        const activeName = turn === 'w' ? 'White (Player)' : 'Black (Mimic AI)';
        statusMessage.textContent = `${activeName}'s turn. Make a move!`;
        statusIcon.setAttribute('data-lucide', 'swords');
    }
    lucide.createIcons();
}

// Handle Checkmate / Draw Modal
function handleGameOver() {
    updateStatusMessage();
    
    let desc = "";
    if (game.in_checkmate()) {
        const winner = game.turn() === 'w' ? 'Black (Mimic AI)' : 'White (Player)';
        desc = `Checkmate! ${winner} has achieved victory.`;
    } else {
        desc = "The game has ended in a draw (Stalemate, Insufficient Material, or Threefold Repetition).";
    }
    
    modalTitle.textContent = "Match Concluded";
    modalDesc.textContent = desc;
    modalOverlay.style.display = 'flex';
}

// Render Move History List in sidebar
function updateMoveHistory() {
    moveHistoryList.innerHTML = '';
    const moves = game.history({ verbose: true });
    
    moveCounter.textContent = `${moves.length} moves`;
    
    if (moves.length === 0) {
        moveHistoryList.innerHTML = '<div class="empty-history-placeholder">No moves recorded yet.</div>';
        return;
    }
    
    let rowsHtml = '';
    for (let i = 0; i < moves.length; i += 2) {
        const moveNum = Math.floor(i / 2) + 1;
        const whiteMove = moves[i].san;
        const blackMove = moves[i + 1] ? moves[i + 1].san : '';
        
        rowsHtml += `
            <div class="history-row">
                <span class="history-num">${moveNum}.</span>
                <span class="history-move" onclick="jumpToMove(${i})">${whiteMove}</span>
                <span class="history-move" onclick="jumpToMove(${i + 1})">${blackMove}</span>
            </div>
        `;
    }
    moveHistoryList.innerHTML = rowsHtml;
    
    // Auto-scroll to bottom of history
    moveHistoryList.scrollTop = moveHistoryList.scrollHeight;
}

// UI Settings change listeners
function setupEventListeners() {
    // Model Selector change event
    if (modelSelector) {
        modelSelector.addEventListener('change', () => {
            const selectedModelName = modelSelector.value;
            if (!selectedModelName) return;
            
            aiThinking = true;
            statusMessage.textContent = `Switching model engine to ${selectedModelName}...`;
            statusIcon.setAttribute('data-lucide', 'refresh-cw');
            lucide.createIcons();
            
            fetch('/api/select_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ model_name: selectedModelName })
            })
            .then(res => res.json())
            .then(data => {
                aiThinking = false;
                if (data.success) {
                    updateModelMetadataDisplay();
                    startNewGame();
                } else {
                    alert(data.error || 'Failed to switch model.');
                    fetchModelsList();
                }
            })
            .catch(err => {
                aiThinking = false;
                console.error("Model select error:", err);
                alert("Error communicating with server.");
                fetchModelsList();
            });
        });
    }

    // Execution Mode Buttons
    const btnExecAuto = document.getElementById('btn-exec-auto');
    const btnExecSuggest = document.getElementById('btn-exec-suggest');
    
    if (btnExecAuto && btnExecSuggest) {
        btnExecAuto.addEventListener('click', () => {
            if (aiThinking) return;
            aiControlMode = 'auto';
            btnExecAuto.classList.add('active');
            btnExecSuggest.classList.remove('active');
            
            // Refresh turn check in case AI was waiting in suggest mode
            checkAITurn();
        });
        
        btnExecSuggest.addEventListener('click', () => {
            if (aiThinking) return;
            aiControlMode = 'suggest';
            btnExecSuggest.classList.add('active');
            btnExecAuto.classList.remove('active');
        });
    }

    // Temperature Slider
    tempSlider.addEventListener('input', (e) => {
        temperature = parseFloat(e.target.value);
        tempVal.textContent = temperature.toFixed(2);
    });
    
    // Heatmap Toggle Switch
    heatmapToggle.addEventListener('change', (e) => {
        showHeatmap = e.target.checked;
        if (showHeatmap) {
            renderHeatmapCumulative();
        } else {
            clearHeatmap();
        }
    });
    
    // Mode Buttons
    document.querySelectorAll('.btn-mode').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const clickedBtn = e.currentTarget;
            document.querySelectorAll('.btn-mode').forEach(b => b.classList.remove('active'));
            clickedBtn.classList.add('active');
            
            gameMode = clickedBtn.dataset.mode;
            
            // Adjust board orientation based on who plays White
            const shouldFlip = gameMode === 'player-vs-ai-white';
            if (boardFlipped !== shouldFlip) {
                boardFlipped = shouldFlip;
                buildBoardSquares();
                updateBoardPieces();
                if (showHeatmap) renderHeatmapCumulative();
            }
            
            // Update labels in banner
            const pWhite = document.getElementById('player-white-info').querySelector('.player-name');
            const pBlack = document.getElementById('player-black-info').querySelector('.player-name');
            
            if (gameMode === 'player-vs-ai-black') {
                pWhite.textContent = "Player (White)";
                pBlack.textContent = "Mimic AI (Black)";
            } else if (gameMode === 'player-vs-ai-white') {
                pWhite.textContent = "Mimic AI (White)";
                pBlack.textContent = "Player (Black)";
            } else {
                pWhite.textContent = "AI White (Mimic)";
                pBlack.textContent = "AI Black (Mimic)";
            }
            
            startNewGame();
        });
    });
    
    // Toggle AI Thoughts Drawer
    if (btnToggleThoughts) {
        btnToggleThoughts.addEventListener('click', () => {
            thinkingDrawer.classList.toggle('show');
            const isOpen = thinkingDrawer.classList.contains('show');
            userPrefersThoughtsOpen = isOpen; // Record user preference
            btnToggleThoughts.innerHTML = isOpen 
                ? '<i data-lucide="eye-off"></i> Hide AI Thoughts'
                : '<i data-lucide="brain"></i> Show AI Thoughts';
            lucide.createIcons();
        });
    }
    
    if (btnCloseDrawer) {
        btnCloseDrawer.addEventListener('click', () => {
            thinkingDrawer.classList.remove('show');
            userPrefersThoughtsOpen = false; // Record user preference (closed)
            if (btnToggleThoughts) {
                btnToggleThoughts.innerHTML = '<i data-lucide="brain"></i> Show AI Thoughts';
                lucide.createIcons();
            }
        });
    }

    // Undo, Reset, Copy FEN
    btnUndo.addEventListener('click', () => {
        if (aiThinking) return;
        
        // If playing vs AI, undo twice to revert player and AI moves
        if (gameMode.startsWith('player-vs-ai')) {
            game.undo();
            game.undo();
        } else {
            game.undo();
        }
        
        selectedSquare = null;
        clearHighlights();
        updateBoardPieces();
        updateMoveHistory();
        updateStatusMessage();
        clearHeatmap();
        clearAIAnalysis();
        clearThinkingProcess();
    });
    
    btnReset.addEventListener('click', () => {
        if (aiThinking) return;
        if (confirm("Reset the match? This will wipe the move logs.")) {
            startNewGame();
        }
    });
    
    btnCopyFen.addEventListener('click', () => {
        const fen = game.fen();
        navigator.clipboard.writeText(fen).then(() => {
            const originalText = btnCopyFen.innerHTML;
            btnCopyFen.innerHTML = '<i data-lucide="check"></i> FEN Copied!';
            lucide.createIcons();
            setTimeout(() => {
                btnCopyFen.innerHTML = originalText;
                lucide.createIcons();
            }, 2000);
        });
    });
    
    if (btnModalRestart) {
        btnModalRestart.addEventListener('click', () => {
            startNewGame();
        });
    }
    
    if (btnModalAnalyze) {
        btnModalAnalyze.addEventListener('click', () => {
            modalOverlay.style.display = 'none';
            analysisMode = true;
            currentReplayIndex = -1;
            updateStatusMessage();
        });
    }
    
    if (btnModalCopyFen) {
        btnModalCopyFen.addEventListener('click', () => {
            const fen = game.fen();
            navigator.clipboard.writeText(fen).then(() => {
                const originalText = btnModalCopyFen.innerHTML;
                btnModalCopyFen.innerHTML = '<i data-lucide="check" style="width:12px;height:12px;margin-right:3px;display:inline-block;vertical-align:middle;"></i>Copied!';
                lucide.createIcons();
                setTimeout(() => {
                    btnModalCopyFen.innerHTML = originalText;
                    lucide.createIcons();
                }, 2000);
            });
        });
    }
    
    if (btnModalCopyPgn) {
        btnModalCopyPgn.addEventListener('click', () => {
            const pgn = game.pgn();
            navigator.clipboard.writeText(pgn).then(() => {
                const originalText = btnModalCopyPgn.innerHTML;
                btnModalCopyPgn.innerHTML = '<i data-lucide="check" style="width:12px;height:12px;margin-right:3px;display:inline-block;vertical-align:middle;"></i>Copied!';
                lucide.createIcons();
                setTimeout(() => {
                    btnModalCopyPgn.innerHTML = originalText;
                    lucide.createIcons();
                }, 2000);
            });
        });
    }
    
    if (btnModalDownloadPgn) {
        btnModalDownloadPgn.addEventListener('click', () => {
            const pgn = game.pgn();
            const blob = new Blob([pgn], { type: 'text/plain;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `mimic_game_${new Date().toISOString().slice(0, 10)}.pgn`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            const originalText = btnModalDownloadPgn.innerHTML;
            btnModalDownloadPgn.innerHTML = '<i data-lucide="check" style="width:12px;height:12px;margin-right:3px;display:inline-block;vertical-align:middle;"></i>Saved!';
            lucide.createIcons();
            setTimeout(() => {
                btnModalDownloadPgn.innerHTML = originalText;
                lucide.createIcons();
            }, 2000);
        });
    }
    
    // Model Upload Zone Click and Drag
    dropzone.addEventListener('click', () => {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0]);
        }
    });
    
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    
    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });
    
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
}

// Asynchronously upload weights to Flask and hot-swap
function handleFileUpload(file) {
    if (!file.name.endsWith('.pth')) {
        showUploadMsg('Only .pth (PyTorch weight files) are supported!', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    const progressContainer = document.getElementById('upload-progress-container');
    const progressFill = document.getElementById('upload-progress-fill');
    const progressText = document.getElementById('upload-progress-text');
    
    progressContainer.style.display = 'block';
    progressFill.style.width = '0%';
    progressText.textContent = 'Uploading 0%';
    showUploadMsg('Sending file to backend...', '');
    
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload', true);
    
    xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 100);
            progressFill.style.width = `${pct}%`;
            progressText.textContent = `Uploading ${pct}%`;
        }
    };
    
    xhr.onload = () => {
        progressContainer.style.display = 'none';
        
        try {
            const res = JSON.parse(xhr.responseText);
            if (xhr.status === 200) {
                showUploadMsg(res.message, 'success');
                // Refresh list and select the uploaded model
                fetchModelsList(res.model_name);
                
                // Play new match with new weights
                startNewGame();
            } else {
                showUploadMsg(res.error || 'Failed to upload model weights.', 'error');
            }
        } catch (e) {
            showUploadMsg('Invalid response from server.', 'error');
        }
    };
    
    xhr.onerror = () => {
        progressContainer.style.display = 'none';
        showUploadMsg('Connection error during upload.', 'error');
    };
    
    xhr.send(formData);
}

function showUploadMsg(text, type) {
    const msg = document.getElementById('upload-msg');
    msg.textContent = text;
    msg.className = `upload-message ${type}`;
}

// // Fetch list of available models from Flask API
function fetchModelsList(selectFilenameAfterLoad) {
    fetch('/api/models')
    .then(res => res.json())
    .then(data => {
        if (!modelSelector) return;
        modelSelector.innerHTML = '';
        if (data.models && data.models.length > 0) {
            data.models.forEach(model => {
                const opt = document.createElement('option');
                opt.value = model.filename;
                opt.textContent = model.display_name;
                opt.dataset.size = model.size_mb;
                opt.dataset.params = model.parameter_count;
                opt.dataset.uses_stockfish = model.uses_stockfish;
                modelSelector.appendChild(opt);
            });
            
            // Set selection
            const targetModel = selectFilenameAfterLoad || data.active_model;
            if (targetModel) {
                modelSelector.value = targetModel;
            }
            
            // Trigger UI update based on the currently selected option
            updateModelMetadataDisplay();
            
            document.getElementById('connection-status-dot').className = "pulse-dot active";
            document.getElementById('connection-status-text').textContent = "Model Engine Ready";
        } else {
            modelSelector.innerHTML = '<option value="">No models loaded</option>';
            modelSizeText.textContent = "0.0 MB";
            modelParamsText.textContent = "0M";
            document.getElementById('connection-status-dot').className = "pulse-dot";
            document.getElementById('connection-status-text').textContent = "No Models Loaded";
            statusMessage.textContent = "Please upload PyTorch model weights (.pth) to play.";
        }
    })
    .catch(err => {
        console.error("Models Fetch Error:", err);
        if (modelSelector) {
            modelSelector.innerHTML = '<option value="">Server Offline</option>';
        }
        document.getElementById('connection-status-dot').className = "pulse-dot";
        document.getElementById('connection-status-text').textContent = "Server Offline";
    });
}

function updateModelMetadataDisplay() {
    if (!modelSelector) return;
    const selectedOpt = modelSelector.options[modelSelector.selectedIndex];
    if (selectedOpt && selectedOpt.value) {
        const size = selectedOpt.dataset.size;
        const params = parseInt(selectedOpt.dataset.params || 0);
        modelSizeText.textContent = `${size} MB`;
        modelParamsText.textContent = formatParamCount(params);
        
        // Dynamic status
        document.getElementById('connection-status-dot').className = "pulse-dot active";
        document.getElementById('connection-status-text').textContent = "Model Ready";
        
        // If the model does not use Stockfish, display a note in the thinking drawer
        const usesStockfish = selectedOpt.dataset.uses_stockfish === 'true';
        if (!usesStockfish) {
            clearThinkingProcess();
            if (thinkingTableBody) {
                thinkingTableBody.innerHTML = `
                    <tr>
                        <td colspan="5" class="empty-table-placeholder" style="color: var(--accent-cyan); font-weight: 500;">
                            <i data-lucide="brain-circuit" style="vertical-align: middle; margin-right: 5px; width: 14px; height: 14px;"></i>
                            Pure Policy Engine Active: Stockfish evaluation is disabled for this model.
                        </td>
                    </tr>
                `;
                lucide.createIcons();
            }
        }
    }
}

// Formatting helpers
function formatParamCount(count) {
    if (!count) return "0M";
    if (count >= 1000000) {
        return (count / 1000000).toFixed(1) + "M";
    } else if (count >= 1000) {
        return (count / 1000).toFixed(0) + "K";
    }
    return count.toString();
}

// Board Index conversions (python-chess 0-63 to chess.js algebraic square mapping)
function squareIdxToUci(sqIdx) {
    const file = sqIdx % 8;
    const rank = Math.floor(sqIdx / 8);
    const fileChar = String.fromCharCode(97 + file); // 'a' starts at ascii 97
    const rankNum = rank + 1; // 1-indexed
    return `${fileChar}${rankNum}`;
}

function uciToSquareIdx(uci) {
    const file = uci.charCodeAt(0) - 97; // 'a' -> 0
    const rank = parseInt(uci.charAt(1)) - 1; // '1' -> 0
    return rank * 8 + file;
}

// Jump board to a specific move index in history for replaying
window.jumpToMove = function(index) {
    if (aiThinking) return;
    
    const moves = game.history({ verbose: true });
    if (index < 0 || index >= moves.length) return;
    
    currentReplayIndex = index;
    
    // Play moves up to index
    const tempGame = new Chess();
    for (let i = 0; i <= index; i++) {
        tempGame.move(moves[i]);
    }
    
    // Display this temporary board state
    updateBoardPieces(tempGame);
    
    // Highlight the active history row visually
    highlightHistoryRow(index);
    
    // Update status feed
    statusMessage.textContent = `Reviewing move ${index + 1}: ${moves[index].san}. Play a move here to start a new analysis branch!`;
    statusIcon.setAttribute('data-lucide', 'eye');
    lucide.createIcons();
};

// Highlight the selected row in move history and dim others
function highlightHistoryRow(index) {
    const rows = document.querySelectorAll('.history-row');
    rows.forEach((row, idx) => {
        const whiteMoveSpan = row.querySelector('.history-move:nth-of-type(1)');
        const blackMoveSpan = row.querySelector('.history-move:nth-of-type(2)');
        
        if (whiteMoveSpan) whiteMoveSpan.classList.remove('active-replay');
        if (blackMoveSpan) blackMoveSpan.classList.remove('active-replay');
        
        const whiteIdx = idx * 2;
        const blackIdx = idx * 2 + 1;
        
        if (whiteIdx === index && whiteMoveSpan) {
            whiteMoveSpan.classList.add('active-replay');
        } else if (blackIdx === index && blackMoveSpan) {
            blackMoveSpan.classList.add('active-replay');
        }
    });
}
