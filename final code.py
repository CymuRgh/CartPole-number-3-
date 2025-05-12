import os
import gymnasium as gym
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import contextlib
import pickle
import csv

# Function to save the episode scores into a CSV file for later analysis
def save_scores_to_csv(scores, filename="scores.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Score"])
        for i, score in enumerate(scores, start=1):
            writer.writerow([i, score])

# Setting device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (Optional) Force CPU usage and suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Context manager to suppress unwanted print outputs during environment resets
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Defining the DQN neural network architecture
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # First hidden layer
        self.fc2 = nn.Linear(512, 256)  # Second hidden layer
        self.fc3 = nn.Linear(256, 64)  # Third hidden layer
        self.out = nn.Linear(64, output_dim)  # Output layer (action space)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.out(x)  # Raw Q-values output

# DQN Agent that interacts with the environment
class DQNAgent:
    def __init__(self, mode="collect"):
        self.mode = mode  # 'collect' (data collection) or 'train' (training mode)
        if self.mode == "train":
            self.env = gym.make("CartPole-v1")
        else:
            self.env = gym.make("CartPole-v1")
            # self.env = gym.make("CartPole-v1", render_mode = "human")

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 5000 if mode == "collect" else 1000  # Training vs. Collecting episodes

        # Replay memory for Experience Replay
        self.memory = deque(maxlen=10000)

        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.batch_size = 64  # Mini-batch size for training
        self.train_start = 1000  # Number of experiences to store before starting training

        # Initializing the Q-network model
        self.model = DQN(self.state_size, self.action_size).to(device)
        self.criterion = nn.MSELoss()  # Loss function
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)

    # Storing experience in replay buffer
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Selecting action based on epsilon-greedy strategy
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action (exploration)
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()  # Action with max Q-value (exploitation)

    # Training the DQN with experiences sampled from memory
    def train(self):
        if len(self.memory) < self.train_start:
            return  # Wait until enough memories are collected

        actual_batch_size = min(len(self.memory), self.batch_size)
        minibatch = random.sample(self.memory, actual_batch_size)

        # converting minibatch into a 2D Numpy array
        mb_array = np.array(minibatch, dtype=object)
        # print(mb_array.shape)
        states = np.vstack(mb_array[:, 0])
        actions = mb_array[:, 1].astype(np.int64)
        rewards = mb_array[:, 2].astype(np.float32)
        next_states = np.vstack(mb_array[:, 3])
        dones = mb_array[:, 4].astype(np.float32)

        # Converting to Numpy arrays/torch tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(
            device)  # Because 'actions is used as an index in .gather(), we should use torch.LongTensor instead of torch.FloatTensor
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Current Q values
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Max Q values for next states
        next_q_values = self.model(next_states).max(1)[0]
        # Target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.detach()

        # Compute loss and backpropagate
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Saving replay buffer to file
    def save_memory(self, filename="replay_buffer.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.memory, f)

    # Loading replay buffer from file
    def load_memory(self, filename="replay_buffer.pkl"):
        with open(filename, "rb") as f:
            self.memory = pickle.load(f)

    # Saving model weights
    def save(self, name):
        torch.save(self.model.state_dict(), name)

    # Loading model weights
    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    # Main loop for running episodes
    def run(self, log_file_path=None):
        scores = []

        if self.mode == "train":
            self.load_memory()

        for e in range(self.EPISODES):
            print(f"Episode: {e + 1}/{self.EPISODES}", end=" ")

            if self.mode == "train":
                self.train()
                continue

            # Resetting environment at the beginning of each episode
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            i = 0  # step counter
            dones = False  # Setting a new variable for while loop '2025.05.12'

            while not dones:
                # Selecting action
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                dones = terminated or truncated
                next_state = np.reshape(next_state, [1, self.state_size])

                reward = 1.0
                if terminated and i < self.env.spec.max_episode_steps - 1:
                    reward -= 100  # Failure
                elif truncated:
                    reward += 10  # Bonus for full survival

                # Saving experience to memory
                self.remember(state, action, reward, next_state, terminated)

                # Train if enough memories are collected
                if len(self.memory) > self.train_start and self.mode == "collect":
                    self.train()

                state = next_state
                i += 1

            print(i)
            scores.append(i)  # Saving episode score

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        # Saving collected experiences if in collection mode
        if self.mode == "collect":
            self.save_memory()

        # Optionally, log scores into CSV
        if log_file_path:
            import csv
            with open(log_file_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Score"])
                for idx, score in enumerate(scores):
                    writer.writerow([idx + 1, score])

        return scores

# Main script execution
if __name__ == "__main__":
    agent = DQNAgent(mode="collect")  # 'collect' to gather experiences
    print("STARTING...")
    agent_scores = agent.run()
    save_scores_to_csv(agent_scores, "training_scores.csv")
    print("DONE!")