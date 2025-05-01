'use client';

import { useEffect, useState } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { getGameState, makeMove, makeAIMove, createGame } from '@/app/actions';

interface GameConfig {
  num_cols: number;
  num_rows: number;
  win_streak: number;
}

interface GameState {
  game_id: string;
  board: string[][];
  current_player: number;
  game_over: boolean;
  winner: string | null;
  message: string;
  player_color: string;
}

export default function GamePage() {
  const router = useRouter();
  const params = useParams();
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [gameConfig, setGameConfig] = useState<GameConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [aiThinking, setAiThinking] = useState(false);
  const gameId = params.gameId as string;

  useEffect(() => {
    const loadGameState = async () => {
      try {
        const data = await getGameState(gameId);
        setGameState(data);
      } catch (err) {
        if (err instanceof Error && err.message === 'Game not found') {
          router.push('/');
          return;
        }
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    loadGameState();
  }, [gameId, router]);

  const handleMakeMove = async (column: number) => {
    try {
      // Make player's move
      const data = await makeMove(gameId, column);
      setGameState(data);

      // If it's still the AI's turn after player's move, wait and make AI move
      if (!data.game_over && data.current_player !== (data.player_color === 'black' ? 0 : 1)) {
        setAiThinking(true);
        // Wait for 1 second before making AI move
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        const aiData = await makeAIMove(gameId);
        setGameState(aiData);
        setAiThinking(false);
      }
    } catch (err) {
      if (err instanceof Error && err.message === "Please wait for your turn") {
        setError(err.message);
        // Refresh the page after 2 seconds
        setTimeout(() => {
          window.location.reload();
        }, 2000);
      } else {
        setError(err instanceof Error ? err.message : 'An error occurred');
        // Refresh the page after 3 seconds for other errors
        setTimeout(() => {
          window.location.reload();
        }, 3000);
      }
      setAiThinking(false);
    }
  };

  if (loading) {
    return <div className="flex items-center justify-center min-h-screen">Loading...</div>;
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen gap-4">
        <p className="text-red-500">{error}</p>
        <Button 
          onClick={() => router.push('/')}
          className="transition-all duration-300 hover:scale-105 hover:shadow-lg"
        >
          Return to Home
        </Button>
      </div>
    );
  }

  if (!gameState) {
    return null;
  }

  const isPlayerTurn = (
    (gameState.current_player === 0 && gameState.player_color === 'black') ||
    (gameState.current_player === 1 && gameState.player_color === 'white')
  );

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 gap-8">
      {gameState.game_over ? (
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-4">Game Over!</h2>
          {gameState.winner ? (
            <p className="text-xl mb-4">
              {gameState.winner === gameState.player_color 
                ? "Congratulations! You won!" 
                : "The AI won this game. Better luck next time!"}
            </p>
          ) : (
            <p className="text-xl mb-4">It's a draw!</p>
          )}
          <div className="grid gap-2 mb-8" style={{ gridTemplateColumns: `repeat(${gameConfig?.num_cols || 7}, minmax(0, 1fr))` }}>
            {gameState.board.map((row, rowIndex) =>
              row.map((cell, colIndex) => (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className="w-12 h-12 border border-gray-300 flex items-center justify-center"
                >
                  {cell}
                </div>
              ))
            )}
          </div>
          <div className="flex gap-4">
            <Button 
              onClick={async () => {
                try {
                  const data = await createGame(gameState.player_color as 'black' | 'white');
                  router.push(`/game/${data.game_id}`);
                } catch (error) {
                  console.error('Error creating game:', error);
                }
              }}
              variant="default"
              className="transition-all duration-300 hover:scale-105 hover:shadow-lg hover:bg-primary/90"
            >
              Start New Game
            </Button>
            <Button 
              onClick={() => router.push('/')} 
              variant="outline"
              className="transition-all duration-300 hover:scale-105 hover:shadow-lg hover:bg-accent"
            >
              Return to Home
            </Button>
          </div>
        </div>
      ) : (
        <>
          <div className="text-center">
            <h2 className="text-2xl font-bold">Game ID: {gameState.game_id}</h2>
            <p className="text-lg mt-2">
              You are playing as <span className="font-semibold">{gameState.player_color === 'black' ? 'O' : 'X'}</span>
            </p>
            <p className="text-lg">
              {isPlayerTurn ? 'Your turn' : aiThinking ? 'AI is thinking...' : 'AI is making a move...'}
            </p>
          </div>
          <div className="flex flex-col items-center gap-4">
            <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${gameConfig?.num_cols || 7}, minmax(0, 1fr))` }}>
              {gameState.board.map((row, rowIndex) =>
                row.map((cell, colIndex) => (
                  <div
                    key={`${rowIndex}-${colIndex}`}
                    className="w-12 h-12 border border-gray-300 flex items-center justify-center"
                  >
                    {cell}
                  </div>
                ))
              )}
            </div>
            <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${gameConfig?.num_cols || 7}, minmax(0, 1fr))` }}>
              {Array.from({ length: gameConfig?.num_cols || 7 }).map((_, index) => (
                <div key={index} className="w-12 flex justify-center">
                  <Button
                    onClick={() => handleMakeMove(index)}
                    disabled={gameState.game_over || !isPlayerTurn || aiThinking}
                    className="transition-all duration-300 hover:scale-105 hover:shadow-lg hover:bg-primary/90 disabled:hover:scale-100 disabled:hover:shadow-none disabled:hover:bg-primary cursor-pointer"
                  >
                    Drop
                  </Button>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
} 