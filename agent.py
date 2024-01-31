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
        self.model = None
        self.trainer = None

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # Danger Right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            # Danger Left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food Location
            game.food.x < game.head.x,  # Food Left
            game.food.x > game.head.x,  # Food Right
            game.food.y < game.head.y,  # Food Up
            game.food.y > game.head.y,  # Food Down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, rewards, next_state, done):
        self.memory.append(
            (state, action, rewards, next_state, done)
        )  # pop left is max_memory reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory, BATCH_SIZE
            )  # List of tuples with BATCH_SIZE length
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random Move: Tradeoff between exploration vs exploitation
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
