import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGame, Direction, Point

MAX_MEM = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0  # Parameter to control Randomness
        self.gamma = 0  # Discount Rate
        self.memory = deque(maxlen=MAX_MEM)  # Queue to store memory
        # TODO:  Model , Trainer

    def get_state(self, game):
        pass

    def remember(self, state, action, rewards, next_state, done):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self, state, action, rewards, next_state, done):
        pass

    def get_action(self, state):
        pass


def train():
    plot_scores = []
    plot_mean_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        # Get old states of the game
        state_old = agent.get_state(game)

        # Get Move
        final_move = agent.get_action(state_old)

        # Perform Move and get new State
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train Short Memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train Long Memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                # agent.model.save()

            print(f"Game: {agent.n_games} | Score: {score} | Record: {record}")


if __name__ == "__main__":
    train()
