"""
Depth-based Reinforcement Learning Trading System
-----------------------------------------------
This system uses 20 levels of order book data for BTC and SOL to train an RL agent for pair trading.
Key features:
1. Uses full order book data (20 levels)
2. Implements pair trading strategy between BTC and SOL
3. PyTorch-based implementation with GPU support
4. Enhanced state representation using order book features
"""

import os
import time
import random
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import pyarrow.feather as feather

# Suppress warnings for cleaner output
warnings.simplefilter('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class OrderBookFeatureExtractor:
    """Extract features from order book data"""
    
    @staticmethod
    def calculate_spread(data: pd.DataFrame) -> pd.Series:
        """Calculate bid-ask spread"""
        return data['ask_0_price'] - data['bid_0_price']
    
    @staticmethod
    def calculate_mid_price(data: pd.DataFrame) -> pd.Series:
        """Calculate mid price"""
        return (data['ask_0_price'] + data['bid_0_price']) / 2
    
    @staticmethod
    def calculate_imbalance(data: pd.DataFrame, levels: int = 5) -> pd.Series:
        """Calculate order book imbalance for given levels"""
        bid_volume = sum(data[f'bid_{i}_size'] for i in range(levels))
        ask_volume = sum(data[f'ask_{i}_size'] for i in range(levels))
        return (bid_volume - ask_volume) / (bid_volume + ask_volume)
    
    @staticmethod
    def calculate_weighted_mid_price(data: pd.DataFrame, levels: int = 5) -> pd.Series:
        """Calculate volume-weighted mid price"""
        bid_prices = np.array([data[f'bid_{i}_price'] for i in range(levels)])
        bid_sizes = np.array([data[f'bid_{i}_size'] for i in range(levels)])
        ask_prices = np.array([data[f'ask_{i}_price'] for i in range(levels)])
        ask_sizes = np.array([data[f'ask_{i}_size'] for i in range(levels)])
        
        weighted_bid = np.sum(bid_prices * bid_sizes, axis=0) / np.sum(bid_sizes, axis=0)
        weighted_ask = np.sum(ask_prices * ask_sizes, axis=0) / np.sum(ask_sizes, axis=0)
        
        return (weighted_bid + weighted_ask) / 2
    
    @staticmethod
    def calculate_price_impact(data: pd.DataFrame, volume: float = 1.0, side: str = 'buy') -> pd.Series:
        """Calculate price impact for a given volume"""
        result = []
        for idx in data.index:
            row = data.loc[idx]
            remaining_volume = volume
            executed_value = 0
            
            if side == 'buy':
                for level in range(20):  # Using all 20 levels
                    level_price = row[f'ask_{level}_price']
                    level_size = row[f'ask_{level}_size']
                    
                    if remaining_volume <= level_size:
                        executed_value += level_price * remaining_volume
                        remaining_volume = 0
                        break
                    else:
                        executed_value += level_price * level_size
                        remaining_volume -= level_size
                
                # If not all volume can be executed, use the last available price
                if remaining_volume > 0:
                    executed_value += row['ask_19_price'] * remaining_volume
                
                avg_price = executed_value / volume
                result.append(avg_price / row['ask_0_price'] - 1)
                
            else:  # sell
                for level in range(20):
                    level_price = row[f'bid_{level}_price']
                    level_size = row[f'bid_{level}_size']
                    
                    if remaining_volume <= level_size:
                        executed_value += level_price * remaining_volume
                        remaining_volume = 0
                        break
                    else:
                        executed_value += level_price * level_size
                        remaining_volume -= level_size
                
                if remaining_volume > 0:
                    executed_value += row['bid_19_price'] * remaining_volume
                
                avg_price = executed_value / volume
                result.append(1 - avg_price / row['bid_0_price'])
                
        return pd.Series(result, index=data.index)
    
    @staticmethod
    def extract_features(data: pd.DataFrame) -> pd.DataFrame:
        """Extract all order book features"""
        features = pd.DataFrame(index=data.index)
        
        # Basic features
        features['spread'] = OrderBookFeatureExtractor.calculate_spread(data)
        features['mid_price'] = OrderBookFeatureExtractor.calculate_mid_price(data)
        
        # Imbalance features at different levels
        features['imbalance_5'] = OrderBookFeatureExtractor.calculate_imbalance(data, 5)
        features['imbalance_10'] = OrderBookFeatureExtractor.calculate_imbalance(data, 10)
        features['imbalance_20'] = OrderBookFeatureExtractor.calculate_imbalance(data, 20)
        
        # Weighted prices
        features['weighted_mid_5'] = OrderBookFeatureExtractor.calculate_weighted_mid_price(data, 5)
        features['weighted_mid_10'] = OrderBookFeatureExtractor.calculate_weighted_mid_price(data, 10)
        
        # Price impact
        features['buy_impact'] = OrderBookFeatureExtractor.calculate_price_impact(data, volume=10.0, side='buy')
        features['sell_impact'] = OrderBookFeatureExtractor.calculate_price_impact(data, volume=10.0, side='sell')
        
        return features


class TradingNetwork(nn.Module):
    """Deep neural network for order book trading with transformer architecture"""
    def __init__(self, n_features=9, time_steps=90, n_assets=1, hidden_dim=64, output_dim=1):
        super(TradingNetwork, self).__init__()
        
        self.n_features = n_features
        self.time_steps = time_steps
        self.n_assets = n_assets
        self.hidden_dim = hidden_dim
        
        # Initial MLP to process raw features
        self.input_mlp = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 2D Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, time_steps, n_assets, hidden_dim))
        
        # Axial Transformer Encoder (for processing both time and asset dimensions)
        self.axial_transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.axial_transformer_encoder = nn.TransformerEncoder(
            self.axial_transformer_encoder_layer,
            num_layers=2
        )
        
        # Pooling for time series axis
        self.time_pool = nn.AdaptiveAvgPool2d((1, n_assets))
        
        # Second Transformer Encoder (after pooling)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=2
        )
        
        # Final MLP for output
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
            
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x):
        # x shape should be: [batch_size, time_steps, assets=1, features]
        batch_size = x.size(0)
        
        # Process each time step through initial MLP
        # Reshape to [batch*time_steps*assets, features]
        x_reshaped = x.reshape(-1, self.n_features)
        x_processed = self.input_mlp(x_reshaped)
        
        # Reshape back to [batch, time_steps, assets, hidden]
        x_processed = x_processed.reshape(batch_size, self.time_steps, self.n_assets, self.hidden_dim)
        
        # Add positional embeddings
        x_with_pos = x_processed + self.pos_embedding
        
        # Reshape for axial transformer [batch, sequence_length=time_steps*assets, hidden]
        x_for_transformer = x_with_pos.reshape(batch_size, self.time_steps * self.n_assets, self.hidden_dim)
        
        # Process through axial transformer encoder
        x_encoded = self.axial_transformer_encoder(x_for_transformer)
        
        # Reshape back to [batch, time_steps, assets, hidden]
        x_encoded = x_encoded.reshape(batch_size, self.time_steps, self.n_assets, self.hidden_dim)
        
        # Pool across time dimension
        # First reshape to match AdaptiveAvgPool2d input: [batch, channels=hidden_dim, height=time_steps, width=assets]
        x_for_pool = x_encoded.permute(0, 3, 1, 2)
        x_pooled = self.time_pool(x_for_pool)
        
        # Reshape from [batch, hidden_dim, 1, assets] to [batch, assets, hidden_dim]
        x_pooled = x_pooled.squeeze(2).permute(0, 2, 1)
        
        # For single asset, we can simplify
        if self.n_assets == 1:
            # For a single asset, shape is [batch, 1, hidden_dim]
            # Squeeze to [batch, hidden_dim]
            x_for_output = x_pooled.squeeze(1)
        else:
            # For multiple assets, use the second transformer
            x_final_encoded = self.transformer_encoder(x_pooled)
            # Reshape to [batch*assets, hidden_dim]
            x_for_output = x_final_encoded.reshape(-1, self.hidden_dim)
        
        # Process through final MLP
        output = self.output_mlp(x_for_output)
        
        # Reshape output based on number of assets
        if self.n_assets == 1:
            # For single asset, reshape to [batch, 1, output_dim]
            output = output.unsqueeze(1)
        else:
            # For multiple assets: [batch, assets, output_dim]
            output = output.reshape(batch_size, self.n_assets, -1)
        
        return output


class ReplayBuffer:
    """Experience replay buffer with GPU support for time series data"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        # Convert time series states to proper tensor format
        # Each state is already a numpy array with shape [time_steps, features]
        return (torch.tensor(np.array(state), device=device, dtype=torch.float32),
                torch.tensor(action, device=device, dtype=torch.long).unsqueeze(1),
                torch.tensor(reward, device=device, dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(next_state), device=device, dtype=torch.float32),
                torch.tensor(done, device=device, dtype=torch.float32).unsqueeze(1))

    def __len__(self) -> int:
        return len(self.buffer)


class TradingAgent:
    """RL agent for trading with order book data"""
    def __init__(
        self,
        n_features=9,
        time_steps=90,
        n_assets=1,
        n_actions=3,  # [hold, buy, sell]
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=10
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_count = 0
        
        # Initialize networks and move to GPU if available
        self.policy_network = TradingNetwork(
            n_features=n_features,
            time_steps=time_steps,
            n_assets=n_assets,
            hidden_dim=64,
            output_dim=n_actions
        ).to(device)
        
        self.target_network = TradingNetwork(
            n_features=n_features,
            time_steps=time_steps,
            n_assets=n_assets,
            hidden_dim=64,
            output_dim=n_actions
        ).to(device)
        
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()  # Target network is only used for inference
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        self.loss_fn = nn.MSELoss()
        
        # Training metrics
        self.losses = []
        self.rewards = []
        self.q_values = []
        
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select action using epsilon-greedy policy"""
        if eval_mode or random.random() > self.epsilon:
            with torch.no_grad():
                # Add batch and asset dimensions: [time_steps, features] -> [1, time_steps, 1, features]
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(2).to(device)
                # Get action values: output shape is [1, 1, n_actions]
                q_values = self.policy_network(state_tensor)
                # Remove batch and asset dimensions to get [n_actions]
                q_values = q_values.squeeze(0).squeeze(0)
                self.q_values.append(q_values.cpu().numpy())
                return q_values.argmax().item()
        return random.randrange(self.n_actions)
        
    def update_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train(self):
        """Train the policy network using batch from replay buffer"""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample batch from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        
        # Reshape state tensors to match model input expectations
        # From [batch, time_steps, features] to [batch, time_steps, 1, features]
        state_batch = state_batch.unsqueeze(2)
        next_state_batch = next_state_batch.unsqueeze(2)
        
        # Forward pass through policy network
        q_values = self.policy_network(state_batch)  # Shape: [batch, 1, n_actions]
        q_values = q_values.squeeze(1)  # Remove asset dimension, now [batch, n_actions]
        q_values = q_values.gather(1, action_batch)  # Get action values
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch)  # Shape: [batch, 1, n_actions]
            next_q_values = next_q_values.squeeze(1)  # Remove asset dimension
            next_q_values = next_q_values.max(1, keepdim=True)[0]  # Get max values
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = self.loss_fn(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_network()
            
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
    def save_model(self, path: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses,
            'rewards': self.rewards,
            'q_values': self.q_values
        }, path)
        print(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=device)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.losses = checkpoint['losses']
        self.rewards = checkpoint['rewards']
        self.q_values = checkpoint['q_values']
        print(f"Model loaded from {path}")


class SingleAssetTradingEnvironment:
    """Environment for single asset trading with order book data"""
    
    def __init__(self, sol_data: pd.DataFrame, window_size: int = 90, 
                 initial_balance: float = 1_000_000, transaction_cost: float = 0.001):
        self.sol_data = sol_data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Feature extractor
        self.sol_features = OrderBookFeatureExtractor.extract_features(sol_data)
        
        # Calculate basic time series
        self.sol_mid_price = self.sol_features['mid_price']
        
        # Get timestamps
        self.sol_timestamps = pd.to_datetime(self.sol_data['origin_time'])
        
        # Process data for time series analysis
        self._process_data()
        
        # Reset environment
        self.reset()
        
    def _process_data(self):
        """Process data to create time windows for the model"""
        # Resample to regular intervals
        self.sol_resampled = self.sol_features.set_index(self.sol_timestamps).resample('1min').last().dropna()
        
        # Create feature windows for the transformer architecture
        self.feature_windows = []
        self.price_changes = []
        
        # Select features to use
        feature_columns = [
            'mid_price', 'spread', 'imbalance_5', 'imbalance_10', 'imbalance_20',
            'weighted_mid_5', 'weighted_mid_10', 'buy_impact', 'sell_impact'
        ]
        
        features_df = self.sol_resampled[feature_columns]
        
        # Normalize features
        self.feature_means = features_df.mean()
        self.feature_stds = features_df.std()
        normalized_features = (features_df - self.feature_means) / self.feature_stds
        
        # Create rolling windows
        for i in range(len(normalized_features) - self.window_size):
            # Extract window
            window = normalized_features.iloc[i:i+self.window_size].values
            self.feature_windows.append(window)
            
            # Calculate price change (return) for the next step
            current_price = self.sol_resampled['mid_price'].iloc[i+self.window_size-1]
            next_price = self.sol_resampled['mid_price'].iloc[i+self.window_size]
            price_change = (next_price / current_price) - 1
            self.price_changes.append(price_change)
            
        # Convert to numpy arrays for faster processing
        self.feature_windows = np.array(self.feature_windows)
        self.price_changes = np.array(self.price_changes)
        
        print(f"Created {len(self.feature_windows)} time windows with shape: {self.feature_windows.shape}")
        
    def reset(self):
        """Reset the environment"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.sol_position = 0
        self.trade_history = []
        self.portfolio_values = [self.initial_balance]
        
        return self._get_state()
        
    def _get_state(self):
        """Get the current state representation"""
        if self.current_step >= len(self.feature_windows):
            return None
        
        # Get current feature window - this is already shaped as [time_steps, features]
        # where time_steps is typically 90 and features is the number of extracted features
        current_window = self.feature_windows[self.current_step].copy()
        
        # The model expects state to be [time_steps, features]
        # No need to reshape as it's already in the correct format
        return current_window
        
    def step(self, action):
        """Take a step in the environment
        Actions:
        0: Hold (no action)
        1: Buy
        2: Sell
        """
        if self.current_step >= len(self.feature_windows) - 1:
            return None, 0, True
        
        # Get current price
        current_index = self.current_step + self.window_size - 1
        sol_price = self.sol_resampled['mid_price'].iloc[current_index]
        
        # Calculate old portfolio value
        old_portfolio_value = self.balance + self.sol_position * sol_price
        
        # Trade amount - 10% of current balance
        trade_amount = self.balance * 0.1
        
        if action == 1:  # Buy
            sol_qty = trade_amount / sol_price
            cost = sol_qty * sol_price * (1 + self.transaction_cost)
            
            self.balance -= cost
            self.sol_position += sol_qty
            
            self.trade_history.append({
                'step': self.current_step,
                'action': 'BUY',
                'price': sol_price,
                'quantity': sol_qty,
                'cost': cost
            })
                
        elif action == 2:  # Sell
            if self.sol_position > 0:
                # Sell all
                proceeds = self.sol_position * sol_price * (1 - self.transaction_cost)
                
                self.balance += proceeds
                
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'price': sol_price,
                    'quantity': -self.sol_position,
                    'proceeds': proceeds
                })
                
                self.sol_position = 0
        
        # Advance to next step
        self.current_step += 1
        
        # Calculate new portfolio value
        new_index = self.current_step + self.window_size - 1
        new_sol_price = self.sol_resampled['mid_price'].iloc[new_index]
        new_portfolio_value = self.balance + self.sol_position * new_sol_price
        
        self.portfolio_values.append(new_portfolio_value)
        
        # Calculate reward (percentage change in portfolio value)
        reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        
        # Penalize excessive trading
        if action > 0:
            reward -= 0.0001  # Small penalty for trading to encourage efficient trades
        
        # Check if episode is done
        done = self.current_step >= len(self.feature_windows) - 1
        
        # Get next state
        next_state = self._get_state()
        
        return next_state, reward, done
        
    def render(self):
        """Render the environment"""
        portfolio_values = np.array(self.portfolio_values)
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(portfolio_values)
        plt.title('Portfolio Value')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        
        # Plot SOL price
        start_idx = self.window_size - 1
        end_idx = start_idx + len(self.portfolio_values)
        price_series = self.sol_resampled['mid_price'].iloc[start_idx:end_idx]
        
        plt.plot(price_series.values)
        plt.title('SOL Price')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


def load_order_book_data(sol_files, max_samples=100000):
    """Load order book data from feather files"""
    print("\nLoading SOL data...")
    sol_data = []
    for file in sol_files:
        print(f"Reading SOL file: {file}...")
        df = feather.read_feather(file)
        sol_data.append(df)
    sol_data = pd.concat(sol_data).reset_index(drop=True)
    
    print(f"Loaded SOL data with shape: {sol_data.shape}")
    print(f"SOL data columns: {sol_data.columns.tolist()}")
    
    # Sample data to reduce memory usage if needed
    if max_samples and len(sol_data) > max_samples:
        sol_data = sol_data.sample(max_samples, random_state=42).reset_index(drop=True)
    
    print("Preprocessing data...")
    print(f"SOL bid price cols: {[col for col in sol_data.columns if 'bid_' in col and 'price' in col]}")
    print(f"SOL ask price cols: {[col for col in sol_data.columns if 'ask_' in col and 'price' in col]}")
    
    return sol_data


def train_agent(agent, env, n_episodes, save_path=None, eval_freq=10):
    """Train the agent using the environment"""
    print("\nStarting training process...")
    print(f"Total episodes to run: {n_episodes}")
    print(f"Using device: {device}")
    print("-" * 50)
    
    all_rewards = []
    best_reward = float('-inf')
    start_time = time.time()
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_rewards = []
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            if next_state is not None:
                agent.memory.push(state, action, reward, next_state, done)
                episode_rewards.append(reward)
                total_reward += reward
                state = next_state
                agent.train()
            else:
                done = True
        
        # Update exploration rate
        agent.update_epsilon()
        
        # Record metrics
        agent.rewards.append(total_reward)
        all_rewards.append(total_reward)
        
        # Print progress
        elapsed_time = time.time() - start_time
        print(f"Episode {episode+1}/{n_episodes} - Reward: {total_reward:.4f}, Epsilon: {agent.epsilon:.4f}")
        
        # Evaluate agent performance
        if (episode + 1) % eval_freq == 0:
            avg_reward = np.mean(all_rewards[-eval_freq:])
            print(f"Average reward over last {eval_freq} episodes: {avg_reward:.4f}")
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                if save_path:
                    agent.save_model(f"{save_path}_best.pt")
                    print(f"New best model saved with avg reward: {best_reward:.4f}")
        
        # Save checkpoint
        if save_path and (episode + 1) % 50 == 0:
            agent.save_model(f"{save_path}_episode_{episode+1}.pt")
            print(f"Checkpoint saved at episode {episode+1}")
    
    # Final save
    if save_path:
        agent.save_model(f"{save_path}_final.pt")
    
    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds!")
    print(f"Best average reward: {best_reward:.4f}")
    
    return all_rewards


def evaluate_agent(agent, env, n_episodes=10):
    """Evaluate the trained agent"""
    print("\nEvaluating agent performance...")
    
    eval_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, done = env.step(action)
            
            if next_state is not None:
                total_reward += reward
                state = next_state
            else:
                done = True
        
        eval_rewards.append(total_reward)
        print(f"Evaluation episode {episode+1}/{n_episodes} - Reward: {total_reward:.4f}")
    
    avg_reward = np.mean(eval_rewards)
    print(f"Average evaluation reward: {avg_reward:.4f}")
    
    # Render final state
    env.render()
    
    return eval_rewards


def main():
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load order book data - only SOL files now
    sol_files = [
        'data/crypto-lake/book/BINANCE_FUTURES/SOL-USDT-PERP/SOL-USDT-PERP_20250207.feather',
        'data/crypto-lake/book/BINANCE_FUTURES/SOL-USDT-PERP/SOL-USDT-PERP_20250208.feather',
        'data/crypto-lake/book/BINANCE_FUTURES/SOL-USDT-PERP/SOL-USDT-PERP_20250209.feather'
    ]
    
    # Load data
    sol_data = load_order_book_data(sol_files)
    
    # Create environment
    env = SingleAssetTradingEnvironment(sol_data, window_size=90, transaction_cost=0.0005)
    
    # Get state dimension from environment
    state = env.reset()
    n_features = state.shape[1]  # Features dimension
    time_steps = state.shape[0]  # Time window length
    n_actions = 3  # [hold, buy, sell]
    
    print(f"State shape: {state.shape}")
    print(f"Feature dimension: {n_features}")
    print(f"Time steps: {time_steps}")
    print(f"Action dimension: {n_actions}")
    
    # Create agent
    agent = TradingAgent(
        n_features=n_features,
        time_steps=time_steps,
        n_assets=1,
        n_actions=n_actions,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=10
    )
    
    # Train agent
    train_rewards = train_agent(
        agent=agent,
        env=env,
        n_episodes=500,
        save_path='models/sol_trading_agent',
        eval_freq=10
    )
    
    # Plot training rewards
    plt.figure(figsize=(12, 6))
    plt.plot(train_rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('training_rewards.png')
    plt.show()
    
    # Evaluate agent
    eval_rewards = evaluate_agent(agent, env, n_episodes=5)
    
    print("SOL trading system training and evaluation complete!")


if __name__ == "__main__":
    main() 