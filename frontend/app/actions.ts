'use server';

import { revalidatePath } from 'next/cache';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

export async function createGame(playerColor: 'black' | 'white') {
  try {
    const response = await fetch(`${API_URL}/games`, {
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
    return data;
  } catch (error) {
    console.error('Error creating game:', error);
    throw error;
  }
}

export async function getGameState(gameId: string) {
  try {
    const response = await fetch(`${API_URL}/games/${gameId}`);
    
    if (!response.ok) {
      if (response.status === 404) {
        throw new Error('Game not found');
      }
      throw new Error('Failed to fetch game state');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching game state:', error);
    throw error;
  }
}

export async function makeMove(gameId: string, column: number) {
  try {
    const response = await fetch(`${API_URL}/games/${gameId}/move`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ column }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      if (response.status === 400 && errorData.detail === "Not your turn") {
        throw new Error('Please wait for your turn');
      }
      throw new Error('Failed to make move');
    }

    const data = await response.json();
    revalidatePath(`/game/${gameId}`);
    return data;
  } catch (error) {
    console.error('Error making move:', error);
    throw error;
  }
}

export async function makeAIMove(gameId: string) {
  try {
    const response = await fetch(`${API_URL}/games/${gameId}/ai-move`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error('Failed to make AI move');
    }

    const data = await response.json();
    revalidatePath(`/game/${gameId}`);
    return data;
  } catch (error) {
    console.error('Error making AI move:', error);
    throw error;
  }
}

export async function getGameHistory() {
  try {
    const response = await fetch(`${API_URL}/games/history`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch game history');
    }

    const data = await response.json();
    return data || [];
  } catch (error) {
    console.error('Error fetching game history:', error);
    return [];
  }
} 