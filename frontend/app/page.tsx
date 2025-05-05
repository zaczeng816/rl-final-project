"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { format } from "date-fns";
import { createGame, getGameHistory } from "./actions";

interface GameHistory {
  game_id: string;
  created_at: number;
  player_color: string;
  winner: string | null;
  moves_count: number;
}

export default function Home() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("new-game");
  const [gameHistory, setGameHistory] = useState<GameHistory[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);

  const handleCreateGame = async (playerColor: "black" | "white") => {
    setLoading(true);
    try {
      const data = await createGame(playerColor);
      router.push(`/game/${data.game_id}`);
    } catch (error) {
      console.error("Error creating game:", error);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (timestamp: number) => {
    return format(new Date(timestamp), "MMM d, yyyy HH:mm:ss");
  };

  const loadHistory = async () => {
    setHistoryLoading(true);
    try {
      const data = await getGameHistory();
      setGameHistory(data);
    } catch (error) {
      console.error("Error loading game history:", error);
      setGameHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === "history") {
      loadHistory();
    }
  }, [activeTab]);

  const NewGame = () => {
    return (
      <TabsContent value="new-game" className="mt-6">
        <div className="flex flex-col items-center gap-4">
          <h2 className="text-2xl font-semibold">Choose Your Piece</h2>
          <div className="flex gap-4">
            <Button
              onClick={() => handleCreateGame("black")}
              disabled={loading}
              className="text-lg px-8 py-6 bg-black text-white hover:bg-gray-800 transition-all duration-300 hover:scale-105 hover:shadow-lg cursor-pointer"
            >
              Play as O (First)
            </Button>
            <Button
              onClick={() => handleCreateGame("white")}
              disabled={loading}
              className="text-lg px-8 py-6 bg-white text-black border-2 border-black hover:bg-gray-100 transition-all duration-300 hover:scale-105 hover:shadow-lg cursor-pointer"
            >
              Play as X (Second)
            </Button>
          </div>
        </div>
      </TabsContent>
    );
  };

  const History = () => {
    return (
      <TabsContent value="history" className="mt-6">
        {historyLoading ? (
          <div className="text-center">Loading game history...</div>
        ) : gameHistory.length === 0 ? (
          <div className="text-center">No games played yet</div>
        ) : (
          <div className="w-full space-y-4">
            {gameHistory.map((game) => (
              <div
                key={game.game_id}
                className="p-4 border rounded-lg hover:bg-gray-50 cursor-pointer"
                onClick={() => router.push(`/game/${game.game_id}`)}
              >
                <div className="flex justify-between items-center">
                  <div>
                    <div className="font-semibold">
                      Game {game.game_id.split("-")[0]}
                    </div>
                    <div className="text-sm text-gray-500">
                      {formatDate(game.created_at)}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-medium">
                      Played as {game.player_color === "black" ? "O" : "X"}
                    </div>
                    <div className="text-sm">
                      {game.winner
                        ? game.winner === game.player_color
                          ? "You won!"
                          : "AI won"
                        : game.moves_count > 0
                        ? "In progress"
                        : "Not started"}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </TabsContent>
    );
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 gap-8">
      <h1 className="text-4xl font-bold">Connect 4</h1>
      <code className="text-gray-500 text-center">
        made with ❤️ by
        <br className="my-1" />
        [zac, jasper, cindy, mish]
      </code>

      <Tabs
        value={activeTab}
        onValueChange={setActiveTab}
        className="w-full max-w-2xl"
      >
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger
            value="new-game"
            className="hover:scale-105 data-[state=active]:scale-105 transition-transform duration-200 cursor-pointer"
          >
            New Game
          </TabsTrigger>
          <TabsTrigger
            value="history"
            className="hover:scale-105 data-[state=active]:scale-105 transition-transform duration-200 cursor-pointer"
          >
            Game History
          </TabsTrigger>
        </TabsList>
        <NewGame />
        <History />
      </Tabs>
    </div>
  );
}
