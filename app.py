import uuid
import json
from typing import Optional, List
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
import time
import random
import string

# Get configuration from environment variables with defaults
CONFIG_PATH = os.getenv('CONNECT4_CONFIG', 'configs/h6_w7_c4_small_600.yaml')
CHECKPOINT_PATH = os.getenv('CONNECT4_CHECKPOINT', 'model_ckpts/cc4_current_net__iter7.pth.tar')
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')

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
redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0)

# Load model and config
configs = yaml.safe_load(open(CONFIG_PATH, 'r'))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
net = ConnectNet(
    num_cols=configs['board']['num_cols'], 
    num_rows=configs['board']['num_rows'], 
    num_blocks=configs['model']['num_blocks']
).to(device)
net.eval()
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
net.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)

class GameState(BaseModel):
    game_id: str
    board: list
    current_player: int
    moves_count: int
    game_over: bool
    winner: Optional[str] = None
    player_color: str  # 'black' or 'white'
    created_at: int
    winning_positions: Optional[List[List[int]]] = None

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
    winning_positions: Optional[List[List[int]]] = None

class GameConfig(BaseModel):
    num_cols: int
    num_rows: int
    win_streak: int

class CreateGameRequest(BaseModel):
    player_color: str

class GameHistory(BaseModel):
    game_id: str
    created_at: int
    player_color: str
    winner: Optional[str]
    moves_count: int

def generate_game_id() -> str:
    timestamp = int(time.time() * 1000)
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{timestamp}-{random_str}"

def create_new_game(player_color: str) -> GameState:
    game_id = generate_game_id()
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
        player_color=player_color,
        created_at=int(time.time() * 1000),
        winning_positions=None
    )
    
    # Store in Redis
    redis_client.set(f"game:{game_id}", json.dumps(game_state.dict()))
    # Add to game history list
    redis_client.lpush("game_history", game_id)
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

def get_game_history(limit: int = 10) -> List[GameHistory]:
    try:
        game_ids = redis_client.lrange("game_history", 0, limit - 1)
        history = []
        for game_id in game_ids:
            try:
                game_id = game_id.decode('utf-8')
                game_data = redis_client.get(f"game:{game_id}")
                if game_data:
                    game_state = GameState(**json.loads(game_data))
                    history.append(GameHistory(
                        game_id=game_state.game_id,
                        created_at=game_state.created_at,
                        player_color=game_state.player_color,
                        winner=game_state.winner,
                        moves_count=game_state.moves_count
                    ))
            except Exception as e:
                print(f"Error processing game {game_id}: {str(e)}")
                continue
        return history
    except Exception as e:
        print(f"Error in get_game_history: {str(e)}")
        return []

@app.get("/test")
async def test_endpoint():
    return {"message": "API is working"}

@app.get("/games/history", response_model=List[GameHistory])
async def get_history(limit: int = 10):
    return get_game_history(limit)

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
    # If the game is over, we need to get the winning positions
    if game_state.game_over and game_state.winner:
        board = cboard(
            num_cols=configs['board']['num_cols'],
            num_rows=configs['board']['num_rows'],
            win_streak=configs['board']['win_streak']
        )
        board.current_board = np.array(game_state.board)
        # Set the player to the opposite of the winner to get correct winning positions
        board.player = 0 if game_state.winner == "white" else 1
        game_state.winning_positions = board.get_winning_positions()
    
    return GameResponse(
        game_id=game_state.game_id,
        board=game_state.board,
        current_player=game_state.current_player,
        game_over=game_state.game_over,
        winner=game_state.winner,
        message="Game state retrieved",
        player_color=game_state.player_color,
        winning_positions=game_state.winning_positions
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
            player_color=game_state.player_color,
            winning_positions=game_state.winning_positions
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
    winning_positions = None
    if board.check_winner():
        game_over = True
        # The winner is the previous player (the one who just made the move)
        winner = "black" if game_state.current_player == 0 else "white"
        winning_positions = board.get_winning_positions()
    elif not board.actions():
        game_over = True
    
    # Update game state
    game_state.board = board.current_board.tolist()
    game_state.current_player = 1 - game_state.current_player
    game_state.moves_count += 1
    game_state.game_over = game_over
    game_state.winner = winner
    game_state.winning_positions = winning_positions
    
    update_game_state(game_state)
    
    return GameResponse(
        game_id=game_state.game_id,
        board=game_state.board,
        current_player=game_state.current_player,
        game_over=game_state.game_over,
        winner=game_state.winner,
        message="Move successful",
        player_color=game_state.player_color,
        winning_positions=game_state.winning_positions
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
            player_color=game_state.player_color,
            winning_positions=game_state.winning_positions
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
    winning_positions = None
    if board.check_winner():
        game_over = True
        # The winner is the previous player (the one who just made the move)
        winner = "black" if game_state.current_player == 0 else "white"
        winning_positions = board.get_winning_positions()
    elif not board.actions():
        game_over = True
    
    # Update game state
    game_state.board = board.current_board.tolist()
    game_state.current_player = 1 - game_state.current_player
    game_state.moves_count += 1
    game_state.game_over = game_over
    game_state.winner = winner
    game_state.winning_positions = winning_positions
    
    update_game_state(game_state)
    
    return GameResponse(
        game_id=game_state.game_id,
        board=game_state.board,
        current_player=game_state.current_player,
        game_over=game_state.game_over,
        winner=game_state.winner,
        message="AI move successful",
        player_color=game_state.player_color,
        winning_positions=game_state.winning_positions
    )

@app.get("/config", response_model=GameConfig)
async def get_config():
    return GameConfig(
        num_cols=configs['board']['num_cols'],
        num_rows=configs['board']['num_rows'],
        win_streak=configs['board']['win_streak']
    )