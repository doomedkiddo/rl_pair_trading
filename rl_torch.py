"""
Enhanced Reinforcement Learning Trading System using PyTorch
--------------------------------------------------------
Key improvements:
1. Modern PyTorch implementation with GPU support
2. Enhanced network architecture
3. Improved training efficiency
4. Better memory management
"""

import warnings
import os
import time
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression
from torch.cuda.amp import autocast, GradScaler
# Suppress warnings for cleaner output
warnings.simplefilter('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TradingNetwork(nn.Module):
    """Modern PyTorch implementation of the trading network"""
    def __init__(self, n_states: int, n_actions: int):
        super(TradingNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_states, 256),  # 128->256
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),  # 256->512
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(512, 256),  # 128->256
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=-1)) 
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer with GPU support"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        
        return (torch.tensor(state, device=device, dtype=torch.float32),
            torch.tensor(action, device=device, dtype=torch.long),
            torch.tensor(reward, device=device, dtype=torch.float32),
            torch.tensor(next_state, device=device, dtype=torch.float32)) 

    def __len__(self) -> int:
        return len(self.buffer)

class TradingAgent:
    """Modern implementation of the trading agent"""
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        buffer_size: int = 1000000,
        batch_size: int = 64
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Initialize network and move to GPU if available
        self.network = TradingNetwork(n_states, n_actions).to(device)
        self.target_network = TradingNetwork(n_states, n_actions).to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.losses = []
        self.rewards = []
        self.q_values = []
        
    def select_action(self, state: np.ndarray) -> int:
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.network(state_tensor)
                self.q_values.append(q_values.cpu().numpy())
                return q_values.argmax().item()
        return random.randrange(self.n_actions)
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 使用CUDA事件优化数据传输
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        states, actions, rewards, next_states = self.memory.sample(self.batch_size)
        
        # 批量处理数据传输
        with torch.cuda.stream(torch.cuda.Stream()):
            current_q_values = self.network(states).gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values)
            
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        end_event.record()
        torch.cuda.synchronize()
        
        self.losses.append(loss.item())
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())
        
    def save_model(self, path: str):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses,
            'rewards': self.rewards,
            'q_values': self.q_values
        }, path)
        
    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.losses = checkpoint['losses']
        self.rewards = checkpoint['rewards']
        self.q_values = checkpoint['q_values']

class CointegrationTrading:
    """Trading environment using cointegration strategy"""
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame, transaction_cost: float = 0.001):
        self.x = x
        self.y = y
        self.transaction_cost = transaction_cost
        self.reset()
        
    def reset(self):
        self.position = 0
        self.cash = 1000000  # Initial capital
        self.portfolio_value = self.cash
        self.trades = []
        return self._get_state()
        
    def _get_state(self) -> np.ndarray:
        # Create state representation
        return np.array([
            self.position,
            self.portfolio_value / self.cash - 1,  # Returns
            self.x['close'].iloc[-1] / self.x['close'].iloc[-2] - 1,  # X returns
            self.y['close'].iloc[-1] / self.y['close'].iloc[-2] - 1,  # Y returns
        ])
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        # Execute trading action
        old_portfolio_value = self.portfolio_value
        
        if action == 0:  # Buy
            if self.position <= 0:
                self._execute_trade(1)
        elif action == 1:  # Sell
            if self.position >= 0:
                self._execute_trade(-1)
        # action == 2 is hold
        
        # Calculate reward
        new_portfolio_value = self._calculate_portfolio_value()
        reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        
        # Update state
        self.portfolio_value = new_portfolio_value
        done = len(self.x) <= 1  # Check if we've run out of data
        
        return self._get_state(), reward, done
        
    def _execute_trade(self, direction: int):
        price = self.y['close'].iloc[-1]
        cost = abs(direction) * price * self.transaction_cost
        
        self.cash -= direction * price + cost
        self.position += direction
        
        self.trades.append({
            'timestamp': self.y.index[-1],
            'price': price,
            'direction': direction,
            'cost': cost
        })
        
    def _calculate_portfolio_value(self) -> float:
        position_value = self.position * self.y['close'].iloc[-1]
        return self.cash + position_value

def train_agent(
    agent: TradingAgent,
    env: CointegrationTrading,
    n_episodes: int,
    target_update_freq: int = 10,
    save_path: Optional[str] = None
):
    """Train the agent with enhanced progress tracking"""
    print("\nStarting training process...")
    print(f"Total episodes to run: {n_episodes}")
    print(f"Using device: {device}")
    print("-" * 50)
    
    episode_rewards = []
    best_reward = float('-inf')
    start_time = time.time()
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state)
            state = next_state
            agent.train()
            total_reward += reward
            steps += 1
            
            # Log training metrics every 100 steps
            if steps % 100 == 0:
                avg_loss = np.mean(agent.losses[-100:]) if agent.losses else 0
                print(f"Episode {episode}/{n_episodes} - Step {steps}")
                print(f"Current epsilon: {agent.epsilon:.4f}")
                print(f"Average loss (last 100): {avg_loss:.6f}")
                print(f"Current reward: {total_reward:.4f}\n")
        
        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()
            print(f"Target network updated at episode {episode}")
        
        # Update exploration rate
        agent.update_epsilon()
        
        # Record metrics
        agent.rewards.append(total_reward)
        episode_rewards.append(total_reward)
        
        # Calculate and log progress metrics
        elapsed_time = time.time() - start_time
        avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
        
        if total_reward > best_reward:
            best_reward = total_reward
            if save_path:
                agent.save_model(f"{save_path}_best.pt")
                print(f"New best model saved with reward: {best_reward:.4f}")
        
        # Print detailed progress every 10 episodes
        if episode % 10 == 0:
            print(f"\nEpisode {episode}/{n_episodes} completed")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
            print(f"Average reward (last 100): {avg_reward:.4f}")
            print(f"Best reward so far: {best_reward:.4f}")
            print(f"Current epsilon: {agent.epsilon:.4f}")
            print(f"Memory buffer size: {len(agent.memory)}")
            print("-" * 50)
        
        # Save model periodically
        if save_path and episode % 100 == 0:
            agent.save_model(f"{save_path}_episode_{episode}.pt")
            print(f"\nCheckpoint saved at episode {episode}")
    
    # Final statistics
    print("\nTraining completed!")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Final average reward: {np.mean(episode_rewards[-100:]):.4f}")
    print(f"Best reward achieved: {best_reward:.4f}")
    
    # Save final model
    if save_path:
        agent.save_model(f"{save_path}_final.pt")
        print("Final model saved")
        
def plot_training_results(agent: TradingAgent):
    """Plot training metrics"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot rewards
    ax1.plot(agent.rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    # Plot losses
    ax2.plot(agent.losses)
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    
    # Plot Q-values
    q_values = np.array(agent.q_values)
    ax3.plot(q_values.mean(axis=1))
    ax3.set_title('Average Q-Values')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Q-Value')
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        # Load and prepare data
        print("Loading data...")
        x_data = pd.read_csv('DATA/P.csv')
        y_data = pd.read_csv('DATA/Y.csv')
        print("Data loaded successfully")
        print(f"X data shape: {x_data.shape}")
        print(f"Y data shape: {y_data.shape}")
        
        print("\nInitializing environment and agent...")
        env = CointegrationTrading(x_data, y_data)
        agent = TradingAgent(
            n_states=4,
            n_actions=3,
            learning_rate=0.01,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995
        )
        print("Environment and agent initialized")
        
        print("\nStarting training...")
        train_agent(
            agent,
            env,
            n_episodes=1000,
            save_path="models/trading_agent"
        )
        
        print("\nPlotting results...")
        plot_training_results(agent)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data files - {e}")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
