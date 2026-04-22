#!/usr/bin/env python3

import os
import sys
import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
    from torch_geometric.data import Batch
except ImportError:
    print("Error: torch_geometric not found. Please install torch-geometric.")
    sys.exit(1)

from vtr_env import VTREnv

# --------------------------------------------------------------------------
# Environment Loading (from run-vtr.py pattern)
# --------------------------------------------------------------------------

def load_env_file(env_file: Path) -> None:
    if not env_file.is_file():
        return # Skip if missing, use defaults or existing env

    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = os.path.expandvars(value.strip().strip("'\""))

# --------------------------------------------------------------------------
# Model Definitions
# --------------------------------------------------------------------------

class ActorCriticGNN(nn.Module):
    def __init__(self, node_features, action_dim, hidden_dim=64):
        super(ActorCriticGNN, self).__init__()
        
        # GNN Backbone
        self.conv1 = SAGEConv(node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        # Actor Head
        self.actor_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic Head
        self.critic_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GNN Layers
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        
        # Global Pooling
        x = global_mean_pool(x, batch)
        
        # Actor & Critic
        mu = self.actor_fc(x)
        value = self.critic_fc(x)
        
        return mu, value

# --------------------------------------------------------------------------
# PPO Agent
# --------------------------------------------------------------------------

class PPOAgent:
    def __init__(self, node_features, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.policy = ActorCriticGNN(node_features, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCriticGNN(node_features, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.mse_loss = nn.MSELoss()

    def select_action(self, state_data):
        with torch.no_grad():
            mu, _ = self.policy_old(state_data)
            std = torch.exp(self.policy.log_std)
            dist = Normal(mu, std)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), action_logprob.detach()

    def update(self, memory):
        # Convert memory to tensors
        # memory.states is a list of PyG Data objects
        # We need to batch them
        states = Batch.from_data_list(memory.states)
        actions = torch.stack(memory.actions)
        old_logprobs = torch.stack(memory.logprobs)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        for _ in range(self.k_epochs):
            mu, state_values = self.policy(states)
            std = torch.exp(self.policy.log_std)
            dist = Normal(mu, std)
            logprobs = dist.log_prob(actions).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            
            # PPO Ratio
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Advantages
            advantages = rewards - state_values.detach().squeeze()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values.squeeze(), rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# --------------------------------------------------------------------------
# Main Search Loop
# --------------------------------------------------------------------------

def main():
    # Load Env (run-vtr pattern)
    script_dir = Path(__file__).resolve().parent.parent
    load_env_file(script_dir / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=Path, default=os.environ.get("VTR_ARCH"))
    parser.add_argument("--circuits", type=Path, nargs="+", default=[Path(os.environ.get("VTR_CIRCUIT"))] if os.environ.get("VTR_CIRCUIT") else None)
    parser.add_argument("--runs-root", type=Path, default=Path(os.environ.get("VTR_RUNS_ROOT", "./runs/ppo_gnn_simple")))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if not args.arch or not args.circuits:
        print("Error: --arch and --circuits must be provided or set in .env")
        sys.exit(1)

    # Logging
    args.runs_root.mkdir(parents=True, exist_ok=True)
    log_file = args.runs_root / "training.log"
    logging.basicConfig(level=logging.INFO, filename=log_file, format='%(asctime)s %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    # Initialize Env and Agent
    env = VTREnv(args.arch, args.circuits, args.runs_root)
    # BLIF features are 6 dims (5 types + 1 num_inputs)
    agent = PPOAgent(node_features=6, action_dim=3)
    memory = Memory()
    
    def denormalize_action(raw_action):
        # raw_action[0] -> aspect_ratio: centered around 1.0, range [0.5, 1.5]
        # raw_action[1] -> dsp_x: range [2, 14]
        # raw_action[2] -> bram_x: range [2, 14]
        a = 1.0 + torch.tanh(raw_action[0]).item() * 0.5
        d = 8.0 + torch.tanh(raw_action[1]).item() * 6.0
        b = 8.0 + torch.tanh(raw_action[2]).item() * 6.0
        
        # Prevent overlapping columns
        if int(round(d)) == int(round(b)):
            b += 1.0
            
        return [max(0.1, a), max(1, d), max(1, b)]

    logging.info("Starting PPO-GNN Training...")
    
    best_reward = -float('inf')

    for epoch in range(args.epochs):
        for _ in range(args.batch_size):
            state_data = env.reset()
            # Wrap state in a single-item batch for the model
            batch_state = Batch.from_data_list([state_data])
            
            raw_action, logprob = agent.select_action(batch_state)
            action = denormalize_action(raw_action[0])
            
            _, reward, done, info = env.step(action)
            
            memory.states.append(state_data)
            memory.actions.append(raw_action[0])
            memory.logprobs.append(logprob[0])
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            logging.info(f"Epoch {epoch} | Circuit {env.circuit_list[env.current_circuit_idx].name} | Action {action} | Reward {reward:.4f} | Success {info['success']}")
            
            if reward > best_reward:
                best_reward = reward
                torch.save(agent.policy.state_dict(), args.runs_root / "best_policy.pth")
                logging.info(f"New best reward: {best_reward:.4f}")

        agent.update(memory)
        memory.clear()
        
        # Save checkpoint
        torch.save(agent.policy.state_dict(), args.runs_root / "latest_policy.pth")

if __name__ == "__main__":
    main()
