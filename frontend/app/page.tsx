'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';

export default function Home() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [showSelection, setShowSelection] = useState(false);

  const handleCreateGame = async (playerColor: 'black' | 'white') => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8001/games', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ player_color: playerColor }),
      });
      if (!response.ok) {
        throw new Error('Failed to create game');
      }
      const data = await response.json();
      router.push(`/game/${data.game_id}`);
    } catch (error) {
      console.error('Error creating game:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 gap-8">
      <h1 className="text-4xl font-bold">Connect 4</h1>
      
      {!showSelection ? (
        <Button 
          onClick={() => setShowSelection(true)}
          disabled={loading}
          className="text-lg px-8 py-6 transition-all duration-300 hover:scale-105 hover:shadow-lg hover:bg-primary/90"
        >
          Start New Game
        </Button>
      ) : (
        <div className="flex flex-col items-center gap-4">
          <h2 className="text-2xl font-semibold">Choose Your Piece</h2>
          <div className="flex gap-4">
            <Button 
              onClick={() => handleCreateGame('black')}
              disabled={loading}
              className="text-lg px-8 py-6 bg-black text-white hover:bg-gray-800 transition-all duration-300 hover:scale-105 hover:shadow-lg"
            >
              Play as O (First)
            </Button>
            <Button 
              onClick={() => handleCreateGame('white')}
              disabled={loading}
              className="text-lg px-8 py-6 bg-white text-black border-2 border-black hover:bg-gray-100 transition-all duration-300 hover:scale-105 hover:shadow-lg"
            >
              Play as X (Second)
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
