#!/usr/bin/env python3
"""
lunar_lander_dqn.py

Deep Q-Network agent for the LunarLander-v3 environment with layman-friendly comments.
Tracks per-episode rewards and plots both raw and smoothed learning curves.
"""

import random
from collections import deque, namedtuple

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# If you have a CUDA-compatible GPU, use it to speed up training; otherwise use your CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """A simple “brain” that takes in the lander's state and spits out value estimates for each action."""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        # Two hidden layers with a non-linear activation in between
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # input → hidden
            nn.ReLU(),                         # activation
            nn.Linear(hidden_dim, hidden_dim), # hidden → hidden
            nn.ReLU(),                         # activation
            nn.Linear(hidden_dim, action_dim), # hidden → output
        )

    def forward(self, x):
        # Given a state (observation), return Q-values for each possible action
        return self.net(x)


class ReplayBuffer:
    """A fixed-size “notebook” where the agent scribbles down experiences (state→action→reward→next)."""
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple(
            "Experience", ["state", "action", "reward", "next_state", "done"]
        )

    def __len__(self):
        # Let us quickly check how many experiences we’ve stored
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        # Write one new experience into our notebook (and drop oldest if full)
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, batch_size):
        # Randomly pick a batch of experiences to learn from (like reviewing past notes)
        batch = random.sample(self.buffer, batch_size)

        # Stack states and next_states into big arrays, then convert to tensors in one go
        states_np      = np.vstack([e.state      for e in batch]).astype(np.float32)
        next_states_np = np.vstack([e.next_state for e in batch]).astype(np.float32)

        states      = torch.from_numpy(states_np     ).to(device)
        next_states = torch.from_numpy(next_states_np).to(device)
        actions     = torch.tensor([e.action     for e in batch],
                                   dtype=torch.int64, device=device).unsqueeze(-1)
        rewards     = torch.tensor([e.reward     for e in batch],
                                   dtype=torch.float32, device=device).unsqueeze(-1)
        dones       = torch.tensor([e.done       for e in batch],
                                   dtype=torch.float32, device=device).unsqueeze(-1)

        return states, actions, rewards, next_states, dones


class DQNAgent:
    """
    The agent that:
     1) Uses one network to pick actions
     2) Uses a second “target” network to stabilize learning
     3) Occasionally reminds itself of old experiences (from ReplayBuffer)
     4) Follows an ε-greedy policy: sometimes explores, sometimes exploits
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        lr=1e-4,
        gamma=0.99,
        tau=1e-3,
        buffer_size=100_000,
        batch_size=64,
        update_every=5,
    ):
        # 1) Create the “online” network and the “target” network
        self.q_eval   = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=lr)

        # 2) Experience replay notebook
        self.buffer     = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        # 3) Learning hyperparameters
        self.gamma        = gamma        # how much we value future reward
        self.tau          = tau          # how softly we update the target net
        self.update_every = update_every # how many steps between learning updates
        self.step_counter = 0
        self.loss_fn      = nn.MSELoss() # simple mean-squared error loss

    def act(self, state, epsilon=0.1):
        """Decide on an action using ε-greedy: a bit of randomness, a bit of best guess."""
        if random.random() < epsilon:
            # Explore: pick a totally random action
            return random.randrange(self.q_eval.net[-1].out_features)
        else:
            # Exploit: pick the best action according to our current Q-network
            state_t = torch.tensor(state, dtype=torch.float32,
                                   device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_eval(state_t)
            return int(torch.argmax(q_values, dim=-1).item())

    def step(self, state, action, reward, next_state, done):
        # Remember this single snapshot
        self.buffer.push(state, action, reward, next_state, done)

        # Every few steps, pick a batch and learn from it
        self.step_counter = (self.step_counter + 1) % self.update_every
        if self.step_counter == 0 and len(self.buffer) >= self.batch_size:
            experiences = self.buffer.sample(self.batch_size)
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # 1) Compute our target: reward + γ * max_a' Q_target(next_state, a') * (1-done)
        q_next   = self.q_target(next_states).detach().max(dim=1, keepdim=True)[0]
        q_target = rewards + (self.gamma * q_next * (1 - dones))

        # 2) Compute what our online network predicts for those same (state, action) pairs
        q_eval = self.q_eval(states).gather(1, actions)

        # 3) Compute loss and take an optimization step
        loss = self.loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4) Softly update target network toward online network
        for eval_p, targ_p in zip(self.q_eval.parameters(),
                                  self.q_target.parameters()):
            targ_p.data.copy_(self.tau * eval_p.data + (1 - self.tau) * targ_p.data)


def train(
    agent,
    n_episodes=2000,
    max_steps=1000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
):
    """
    Run through many landing attempts ("episodes").
    Track total reward each time and print average every 100 episodes.
    """
    scores  = []
    epsilon = eps_start
    # We turn off rendering here to speed up training
    env     = gym.make("LunarLander-v3", render_mode=None)

    for ep in range(1, n_episodes + 1):
        # Reset for a fresh landing attempt
        state, _    = env.reset(seed=ep)
        total_reward = 0.0

        for _ in range(max_steps):
            # Decide on an action, see what happens, store that experience
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)
        # Gradually shift from exploring to exploiting
        epsilon = max(eps_end, eps_decay * epsilon)

        # Every 100 episodes, show how we're doing on average
        if ep % 100 == 0:
            avg = np.mean(scores[-100:])
            print(f"Episode {ep}\tAverage (last 100): {avg:.2f}")

    env.close()
    return scores


if __name__ == "__main__":
    # 1) Figure out how many inputs/outputs we need
    env        = gym.make("LunarLander-v3")
    state_dim  = env.observation_space.shape[0]  # e.g. 8 values
    action_dim = env.action_space.n              # e.g. 4 possible thrusts
    env.close()

    # 2) Create agent and train
    agent  = DQNAgent(state_dim, action_dim)
    scores = train(agent,
                   n_episodes=2000,
                   max_steps=1000,
                   eps_start=1.0,
                   eps_end=0.01,
                   eps_decay=0.995)

    # 3) Plot what happened: reward per episode + moving average
    plt.figure(figsize=(10, 6))
    plt.plot(scores, label="Reward per episode")

    window = 100
    cumsum = np.cumsum([0] + scores)
    moving_avg = (cumsum[window:] - cumsum[:-window]) / window
    plt.plot(range(window, len(scores) + 1), moving_avg,
             label="100-episode moving avg")

    plt.axhline(200, color="gray", linestyle="--",
                label="Solved threshold (200)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("LunarLander DQN Learning Curve")
    plt.legend()
    plt.savefig("training_curve.png")  # save a copy
    plt.show()  # display it interactively