"use server";

import { revalidatePath } from "next/cache";
import { cookies } from "next/headers";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

// Generate a unique browser signature
async function generateBrowserSignature() {
  const cookieStore = await cookies();
  let signature = cookieStore.get("browser_signature")?.value;

  if (!signature) {
    // Generate a new signature if none exists
    signature =
      Math.random().toString(36).substring(2) + Date.now().toString(36);
    cookieStore.set("browser_signature", signature, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "strict",
      maxAge: 60 * 60 * 24 * 365, // 1 year
    });
  }

  return signature;
}

export async function createGame(playerColor: "black" | "white") {
  try {
    const signature = await generateBrowserSignature();
    const response = await fetch(`${API_URL}/games`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Browser-Signature": signature,
      },
      body: JSON.stringify({ player_color: playerColor }),
    });

    if (!response.ok) {
      throw new Error("Failed to create game");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error creating game:", error);
    throw error;
  }
}

export async function getGameState(gameId: string) {
  try {
    const signature = await generateBrowserSignature();
    const response = await fetch(`${API_URL}/games/${gameId}`, {
      headers: {
        "X-Browser-Signature": signature,
      },
    });

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error("Game not found");
      }
      throw new Error("Failed to fetch game state");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error fetching game state:", error);
    throw error;
  }
}

export async function makeMove(gameId: string, column: number) {
  try {
    const signature = await generateBrowserSignature();
    const response = await fetch(`${API_URL}/games/${gameId}/move`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Browser-Signature": signature,
      },
      body: JSON.stringify({ column }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      if (response.status === 400 && errorData.detail === "Not your turn") {
        throw new Error("Please wait for your turn");
      }
      throw new Error("Failed to make move");
    }

    const data = await response.json();
    revalidatePath(`/game/${gameId}`);
    return data;
  } catch (error) {
    console.error("Error making move:", error);
    throw error;
  }
}

export async function makeAIMove(gameId: string) {
  try {
    const signature = await generateBrowserSignature();
    const response = await fetch(`${API_URL}/games/${gameId}/ai-move`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Browser-Signature": signature,
      },
    });

    if (!response.ok) {
      throw new Error("Failed to make AI move");
    }

    const data = await response.json();
    revalidatePath(`/game/${gameId}`);
    return data;
  } catch (error) {
    console.error("Error making AI move:", error);
    throw error;
  }
}

export async function getGameHistory() {
  try {
    const signature = await generateBrowserSignature();
    const response = await fetch(`${API_URL}/games/history`, {
      headers: {
        "X-Browser-Signature": signature,
      },
    });

    if (!response.ok) {
      throw new Error("Failed to fetch game history");
    }

    const data = await response.json();
    return data || [];
  } catch (error) {
    console.error("Error fetching game history:", error);
    return [];
  }
}
