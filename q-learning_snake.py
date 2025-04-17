import pygame
import random
import numpy as np
import time

# Pygame setup
pygame.init()

# Game parameters
GRID_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 20
SCREEN_WIDTH = GRID_SIZE * GRID_WIDTH
SCREEN_HEIGHT = GRID_SIZE * GRID_HEIGHT

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Game settings
FPS = 15

# Actions
STRAIGHT = 0
LEFT = 1
RIGHT = 2
ACTIONS = [STRAIGHT, LEFT, RIGHT]

# Directions (UP, RIGHT, DOWN, LEFT)
UP = (0, -1)
RIGHT_DIR = (1, 0)
DOWN = (0, 1)
LEFT_DIR = (-1, 0)
DIRECTIONS = [UP, RIGHT_DIR, DOWN, LEFT_DIR]


class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(10, 10)]
        self.direction_idx = 1
        self.snake_direction = DIRECTIONS[self.direction_idx]
        self.food = self.generate_food()
        self.score = 0
        self.done = False

    def generate_food(self):
        while True:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food not in self.snake:
                return food

    def step(self, action):
        if action == LEFT:
            self.direction_idx = (self.direction_idx - 1) % 4
        elif action == RIGHT:
            self.direction_idx = (self.direction_idx + 1) % 4
        self.snake_direction = DIRECTIONS[self.direction_idx]

        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.snake_direction
        new_head = (head_x + dir_x, head_y + dir_y)

        if new_head in self.snake or not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
            self.done = True
            return -100

        self.snake = [new_head] + self.snake
        if new_head == self.food:
            self.food = self.generate_food()
            self.score += 1
            return 10
        else:
            self.snake.pop()
            return -0.01

    def get_state(self):
        head = self.snake[0]
        food = self.food
        rel_food = (np.sign(food[0] - head[0]), np.sign(food[1] - head[1]))
        return (rel_food, self.direction_idx)

    def render(self, screen):
        screen.fill(BLACK)
        for segment in self.snake:
            pygame.draw.rect(screen, GREEN, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(screen, RED, (self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.display.update()


class QLearningAgent:
    def __init__(self, game):
        self.game = game
        self.q_table = {}
        self.alpha = 0.1  # learning rate
        self.gamma = 0.9  # discount factor
        self.epsilon = 0.1  # exploration rate

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        q_values = [self.get_q(state, a) for a in ACTIONS]
        max_q = max(q_values)
        best_actions = [a for a in ACTIONS if self.get_q(state, a) == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        max_next_q = max([self.get_q(next_state, a) for a in ACTIONS])
        new_q = (1 - self.alpha) * self.get_q(state, action) + self.alpha * (reward + self.gamma * max_next_q)
        self.q_table[(state, action)] = new_q

    def train(self, episodes):
        for episode in range(episodes):
            self.game.reset()
            state = self.game.get_state()
            while not self.game.done:
                action = self.choose_action(state)
                reward = self.game.step(action)
                next_state = self.game.get_state()
                self.update(state, action, reward, next_state)
                state = next_state


if __name__ == "__main__":
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Q-Learning Snake")
    clock = pygame.time.Clock()

    game = SnakeGame()
    agent = QLearningAgent(game)
    num = input("How many times do you wnat to train before it runs? ")
    print("Training...")
    agent.train(int(num))
    print("Training complete. Press ESC to quit.")

    running = True
    game.reset()
    state = game.get_state()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        if not game.done:
            action = agent.choose_action(state)
            reward = game.step(action)
            next_state = game.get_state()
            state = next_state
            game.render(screen)
            clock.tick(FPS)
        else:
            game.reset()
            state = game.get_state()

    pygame.quit()