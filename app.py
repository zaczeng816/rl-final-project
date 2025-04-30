import uuid
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
import torch
import numpy as np
from model import ConnectNet
from connect_board import Board as cboard, encode_board, decode_board
from MCTS import UCT_search, get_policy
import yaml
import os

# Get configuration from environment variables with defaults
CONFIG_PATH = os.getenv('CONNECT4_CONFIG', 'configs/h6_w7_c4_small_600.yaml')
CHECKPOINT_PATH = os.getenv('CONNECT4_CHECKPOINT', 'model_ckpts/cc4_current_net__iter7.pth.tar')

# Validate paths
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint file not found: {CHECKPOINT_PATH}")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Load model and config
configs = yaml.safe_load(open(CONFIG_PATH, 'r'))
device = "cuda" if torch.cuda.is_available() else "cpu"
net = ConnectNet(
    num_cols=configs['board']['num_cols'], 
    num_rows=configs['board']['num_rows'], 
    num_blocks=configs['model']['num_blocks']
).to(device)
net.eval()
checkpoint = torch.load(CHECKPOINT_PATH)
net.load_state_dict(checkpoint)

class GameState(BaseModel):
    game_id: str
    board: list
    current_player: int
    moves_count: int
    game_over: bool
    winner: Optional[str] = None
    player_color: str  # 'black' or 'white'

class MoveRequest(BaseModel):
    column: int

class GameResponse(BaseModel):
    game_id: str
    board: list
    current_player: int
    game_over: bool
    winner: Optional[str] = None
    message: str
    player_color: str

class GameConfig(BaseModel):
    num_cols: int
    num_rows: int
    win_streak: int

class CreateGameRequest(BaseModel):
    player_color: str

def create_new_game(player_color: str) -> GameState:
    game_id = str(uuid.uuid4())
    board = cboard(
        num_cols=configs['board']['num_cols'],
        num_rows=configs['board']['num_rows'],
        win_streak=configs['board']['win_streak']
    )
    
    game_state = GameState(
        game_id=game_id,
        board=board.current_board.tolist(),
        current_player=0,  # Black starts first
        moves_count=0,
        game_over=False,
        player_color=player_color
    )
    
    # Store in Redis
    redis_client.set(f"game:{game_id}", json.dumps(game_state.dict()))
    return game_state

def get_game_state(game_id: str) -> GameState:
    game_data = redis_client.get(f"game:{game_id}")
    if not game_data:
        raise HTTPException(status_code=404, detail="Game not found")
    return GameState(**json.loads(game_data))

def update_game_state(game_state: GameState):
    redis_client.set(f"game:{game_state.game_id}", json.dumps(game_state.dict()))

def do_decode_n_move_pieces(board: cboard, move: int) -> cboard:
    """Make a move on the board and return the updated board."""
    try:
        board.drop_piece(move)
        return board
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

def make_ai_move(board: cboard) -> cboard:
    """Make an AI move and return the updated board."""
    root = UCT_search(board, configs['mcts']['num_simulations'], net, 0.1, device)
    policy = get_policy(root, 0.1)
    ai_move = np.random.choice(np.arange(board.num_cols), p=policy)
    return do_decode_n_move_pieces(board, ai_move)

@app.post("/games", response_model=GameResponse)
async def create_game(request: CreateGameRequest):
    if request.player_color not in ['black', 'white']:
        raise HTTPException(status_code=400, detail="Invalid player color. Must be 'black' or 'white'")
    
    game_state = create_new_game(request.player_color)
    
    # If player is white, make AI's first move
    if request.player_color == 'white':
        board = cboard(
            num_cols=configs['board']['num_cols'],
            num_rows=configs['board']['num_rows'],
            win_streak=configs['board']['win_streak']
        )
        board.current_board = np.array(game_state.board)
        board.player = game_state.current_player
        
        board = make_ai_move(board)
        game_state.board = board.current_board.tolist()
        game_state.current_player = 1 - game_state.current_player
        game_state.moves_count += 1
        update_game_state(game_state)
    
    return GameResponse(
        game_id=game_state.game_id,
        board=game_state.board,
        current_player=game_state.current_player,
        game_over=game_state.game_over,
        message="New game created",
        player_color=game_state.player_color
    )

@app.get("/games/{game_id}", response_model=GameResponse)
async def get_game(game_id: str):
    game_state = get_game_state(game_id)
    return GameResponse(
        game_id=game_state.game_id,
        board=game_state.board,
        current_player=game_state.current_player,
        game_over=game_state.game_over,
        winner=game_state.winner,
        message="Game state retrieved",
        player_color=game_state.player_color
    )

@app.post("/games/{game_id}/move", response_model=GameResponse)
async def make_move(game_id: str, move: MoveRequest):
    game_state = get_game_state(game_id)
    
    if game_state.game_over:
        return GameResponse(
            game_id=game_state.game_id,
            board=game_state.board,
            current_player=game_state.current_player,
            game_over=game_state.game_over,
            winner=game_state.winner,
            message="Game is already over",
            player_color=game_state.player_color
        )
    
    # Check if it's the player's turn
    is_player_turn = (
        (game_state.current_player == 0 and game_state.player_color == 'black') or
        (game_state.current_player == 1 and game_state.player_color == 'white')
    )
    
    if not is_player_turn:
        raise HTTPException(status_code=400, detail="Not your turn")
    
    # Create board from state
    board = cboard(
        num_cols=configs['board']['num_cols'],
        num_rows=configs['board']['num_rows'],
        win_streak=configs['board']['win_streak']
    )
    board.current_board = np.array(game_state.board)
    board.player = game_state.current_player
    
    # Validate move
    if move.column < 0 or move.column >= board.num_cols:
        raise HTTPException(status_code=400, detail="Invalid column")
    
    # Make player's move
    board = do_decode_n_move_pieces(board, move.column)
    
    # Check for winner after player's move
    winner = None
    game_over = False
    if board.check_winner():
        game_over = True
        winner = "black" if board.player == 0 else "white"
    elif not board.actions():
        game_over = True
    
    # Update game state
    game_state.board = board.current_board.tolist()
    game_state.current_player = 1 - game_state.current_player
    game_state.moves_count += 1
    game_state.game_over = game_over
    game_state.winner = winner
    
    update_game_state(game_state)
    
    return GameResponse(
        game_id=game_state.game_id,
        board=game_state.board,
        current_player=game_state.current_player,
        game_over=game_state.game_over,
        winner=game_state.winner,
        message="Move made successfully",
        player_color=game_state.player_color
    )

@app.post("/games/{game_id}/ai-move", response_model=GameResponse)
async def make_ai_move_endpoint(game_id: str):
    game_state = get_game_state(game_id)
    
    if game_state.game_over:
        return GameResponse(
            game_id=game_state.game_id,
            board=game_state.board,
            current_player=game_state.current_player,
            game_over=game_state.game_over,
            winner=game_state.winner,
            message="Game is already over",
            player_color=game_state.player_color
        )
    
    # Check if it's the AI's turn
    is_ai_turn = (
        (game_state.current_player == 0 and game_state.player_color == 'white') or
        (game_state.current_player == 1 and game_state.player_color == 'black')
    )
    
    if not is_ai_turn:
        raise HTTPException(status_code=400, detail="Not AI's turn")
    
    # Create board from state
    board = cboard(
        num_cols=configs['board']['num_cols'],
        num_rows=configs['board']['num_rows'],
        win_streak=configs['board']['win_streak']
    )
    board.current_board = np.array(game_state.board)
    board.player = game_state.current_player
    
    # Make AI move
    board = make_ai_move(board)
    
    # Check for winner after AI's move
    winner = None
    game_over = False
    if board.check_winner():
        game_over = True
        winner = "black" if board.player == 0 else "white"
    elif not board.actions():
        game_over = True
    
    # Update game state
    game_state.board = board.current_board.tolist()
    game_state.current_player = 1 - game_state.current_player
    game_state.moves_count += 1
    game_state.game_over = game_over
    game_state.winner = winner
    
    update_game_state(game_state)
    
    return GameResponse(
        game_id=game_state.game_id,
        board=game_state.board,
        current_player=game_state.current_player,
        game_over=game_state.game_over,
        winner=game_state.winner,
        message="AI move made successfully",
        player_color=game_state.player_color
    )

@app.get("/config", response_model=GameConfig)
async def get_config():
    return GameConfig(
        num_cols=configs['board']['num_cols'],
        num_rows=configs['board']['num_rows'],
        win_streak=configs['board']['win_streak']
    )