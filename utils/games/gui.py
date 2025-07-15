import tkinter as tk
import numpy as np
import random
import time
from tkinter import messagebox

class GUI:
    def __init__(self, args, bot):
        self.root = tk.Tk()
        self.bot = bot # AI bot
        self.game = bot.game

        self.root.title(f"{args.game.capitalize()} (Player vs. AI)")
        self.root.configure(bg='#F0F0F0')

        # hard coded board UI configuration
        self.BOARD_SIZE = args.board_size
        self.CELL_SIZE = 40
        self.MARGIN = 30
        self.STONE_RADIUS = self.CELL_SIZE // 2 - 4

        self.board = self.game.get_empty_board()
        self.current_player = 1 
        self.previous_player = -1
        self.game_over = False
        self.is_bot_turn = False

        self.status_label = tk.Label(
            self.root, 
            text="Your Turn (Black)", 
            font=("Arial", 16, "bold"), 
            pady=10,
            bg='#F0F0F0'
        )
        self.status_label.pack()

        canvas_size = self.BOARD_SIZE * self.CELL_SIZE + 2 * self.MARGIN
        self.canvas = tk.Canvas(
            self.root, 
            width=canvas_size, 
            height=canvas_size, 
            bg='#D2B48C'
        )
        self.canvas.pack(padx=10, pady=5)
        self.canvas.bind("<Button-1>", self.handle_click)

        self.new_game_button = tk.Button(
            self.root, 
            text="New Game", 
            font=("Arial", 14), 
            command=self.start_new_game,
            pady=5
        )
        self.new_game_button.pack(pady=10)

        self.start_new_game()
        self.root.mainloop()

    def start_new_game(self):
        """Resets the game state and redraws the board for a new game."""
        self.board = self.game.get_empty_board()

        self.current_player = -self.previous_player
        self.previous_player = self.current_player

        self.game_over = False
        self.is_bot_turn = False if self.current_player == 1 else True

        if self.current_player == 1:
            self.status_label.config(text="Your Turn (Black)")
            self.canvas.delete("all")
            self.draw_board()
        else:
            self.status_label.config(text="AI's Turn (White)")
            self.canvas.delete("all")
            self.draw_board()
            self.bot_turn()

    def draw_board(self):
        board_end = (self.BOARD_SIZE - 1) * self.CELL_SIZE + self.MARGIN
        for i in range(self.BOARD_SIZE):
            pos = i * self.CELL_SIZE + self.MARGIN
            self.canvas.create_line(pos, self.MARGIN, pos, board_end, fill="black")
            self.canvas.create_line(self.MARGIN, pos, board_end, pos, fill="black")

    def handle_click(self, event):
        if self.game_over or self.is_bot_turn:
            return

        col = round((event.x - self.MARGIN) / self.CELL_SIZE)
        row = round((event.y - self.MARGIN) / self.CELL_SIZE)

        if 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE and self.board[row][col] == 0:
            self.place_stone(row, col)
            
            # If game is not over, trigger the bot's turn
            if not self.game_over:
                self.is_bot_turn = True
                self.status_label.config(text="AI's Turn (White)")
                # Use 'after' to give a small delay for a more natural feel
                self.root.after(100, self.bot_turn)

    def bot_turn(self):
        if self.game_over:
            return

        action_id = self.bot.move(self.board)
        row, col = action_id // self.BOARD_SIZE, action_id % self.BOARD_SIZE
        self.place_stone(row, col)
        self.is_bot_turn = False # Allow player to click again

    def place_stone(self, row, col):

        color = "black" if self.current_player == 1 else "white"
        x = col * self.CELL_SIZE + self.MARGIN
        y = row * self.CELL_SIZE + self.MARGIN

        self.canvas.create_oval(
            x - self.STONE_RADIUS, y - self.STONE_RADIUS,
            x + self.STONE_RADIUS, y + self.STONE_RADIUS,
            fill=color, outline=color
        )
        self.board[row, col] = self.current_player

        if self.game.wins_here(self.board, row * self.BOARD_SIZE + col):
            self.game_over = True
            winner = "You (Black)" if self.current_player == 1 else "Bot (White)"
            self.status_label.config(text=f"{winner} wins!")
            messagebox.showinfo("Game Over", f"{winner} wins!")

        elif np.sum(self.game.get_valid_moves(self.board)) == 0:
            self.game_over = True
            self.status_label.config(text="It's a draw!")
            messagebox.showinfo("Game Over", "It's a draw!")

        else:
            self.current_player = -self.current_player
            if not self.game_over:
                if self.current_player == 1:
                    self.status_label.config(text="Your Turn (Black)")
                else:
                    self.status_label.config(text="AI's Turn (White)")