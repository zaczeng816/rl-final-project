import { Suspense } from 'react';
import GameClient from './GameClient';

export default function GamePage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center min-h-screen">Loading...</div>}>
      <GameClient />
    </Suspense>
  );
} 