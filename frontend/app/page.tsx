'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';

export default function Home() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);

  const handleCreateGame = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8001/games', {
        method: 'POST',
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
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <h1 className="text-4xl font-bold mb-8">Connect 4</h1>
      <Button 
        onClick={handleCreateGame} 
        disabled={loading}
        className="text-lg px-8 py-6 transition-all duration-300 hover:scale-105 hover:shadow-lg hover:bg-primary/90"
      >
        {loading ? 'Creating Game...' : 'Start New Game'}
      </Button>
    </div>
  );
}
