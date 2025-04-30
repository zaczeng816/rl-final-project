'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { use } from 'react';

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
}

export default function GamePage({ params }: { params: Promise<{ gameId: string }> }) {
  const router = useRouter();
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [gameConfig, setGameConfig] = useState<GameConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { gameId } = use(params);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await fetch('http://localhost:8001/config');
        if (!response.ok) {
          throw new Error('Failed to fetch game config');
        }
        const config = await response.json();
        setGameConfig(config);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      }
    };

    fetchConfig();
  }, []);

  useEffect(() => {
    const fetchGameState = async () => {
      try {
        const response = await fetch(`http://localhost:8001/games/${gameId}`);
        if (!response.ok) {
          if (response.status === 404) {
            router.push('/');
            return;
          }
          throw new Error('Failed to fetch game state');
        }
        const data = await response.json();
        setGameState(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchGameState();
  }, [gameId, router]);

  const handleMakeMove = async (column: number) => {
    try {
      const response = await fetch(`http://localhost:8001/games/${gameId}/move`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ column }),
      });

      if (!response.ok) {
        throw new Error('Failed to make move');
      }

      const data = await response.json();
      setGameState(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    }
  };

  if (loading || !gameConfig) {
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

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 gap-8">
      {gameState.game_over ? (
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-4">Game Over!</h2>
          {gameState.winner ? (
            <p className="text-xl mb-4">Winner: {gameState.winner}</p>
          ) : (
            <p className="text-xl mb-4">It's a draw!</p>
          )}
          <div className="grid gap-2 mb-8" style={{ gridTemplateColumns: `repeat(${gameConfig.num_cols}, minmax(0, 1fr))` }}>
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
              onClick={() => router.push('/')} 
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
          <h2 className="text-2xl font-bold">Game ID: {gameState.game_id}</h2>
          <div className="flex flex-col items-center gap-4">
            <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${gameConfig.num_cols}, minmax(0, 1fr))` }}>
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
            <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${gameConfig.num_cols}, minmax(0, 1fr))` }}>
              {Array.from({ length: gameConfig.num_cols }).map((_, index) => (
                <div key={index} className="w-12 flex justify-center">
                  <Button
                    onClick={() => handleMakeMove(index)}
                    disabled={gameState.game_over}
                    className="transition-all duration-300 hover:scale-105 hover:shadow-lg hover:bg-primary/90 disabled:hover:scale-100 disabled:hover:shadow-none disabled:hover:bg-primary"
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