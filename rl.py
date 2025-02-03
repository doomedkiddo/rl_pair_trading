"""
Enhanced Reinforcement Learning Trading System
-------------------------------------------
Added features:
1. Model saving and loading
2. Enhanced backtesting with capital management
3. Detailed performance metrics
"""

import warnings
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import Basics as basics
import Reinforcement as RL
from Cointegration import EGCointegration

# Suppress warnings for cleaner output
warnings.simplefilter('ignore')

def load_and_prepare_data(x_path, y_path, train_split=0.7):
    """Load and prepare data for training and testing"""
    x = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    
    x, y = EGCointegration.clean_data(x, y, 'date', 'close')
    
    train_len = round(len(x) * train_split)
    idx_train = list(range(0, train_len))
    idx_test = list(range(train_len, len(x)))
    
    eg_train = EGCointegration(x.iloc[idx_train, :], y.iloc[idx_train, :], 'date', 'close')
    eg_test = EGCointegration(x.iloc[idx_test, :], y.iloc[idx_test, :], 'date', 'close')
    
    return eg_train, eg_test

def create_action_space():
    """Define the action space parameters"""
    return {
        'n_hist': list(np.arange(60, 601, 60)),      # Historical window size
        'n_forward': list(np.arange(120, 1201, 120)), # Trading window size
        'trade_th': list(np.arange(1, 5.1, 1)),      # Trading threshold
        'stop_loss': list(np.arange(1, 2.1, 0.5)),   # Stop loss threshold
        'cl': list(np.arange(0.05, 0.11, 0.05))      # Confidence level
    }

def create_network(n_state, n_action):
    """Create the neural network architecture"""
    one_hot = {
        'one_hot': {
            'func_name': 'one_hot',
            'input_arg': 'indices',
            'layer_para': {
                'indices': None,
                'depth': n_state
            }
        }
    }
    
    output_layer = {
        'final': {
            'func_name': 'fully_connected',
            'input_arg': 'inputs',
            'layer_para': {
                'inputs': None,
                'num_outputs': n_action,
                'biases_initializer': None,
                'activation_fn': tf.nn.relu,
                'weights_initializer': tf.ones_initializer()
            }
        }
    }
    
    state_in = tf.placeholder(shape=[1], dtype=tf.int32)
    network = basics.Network(state_in)
    network.build_layers(one_hot)
    network.add_layer_duplicates(output_layer, 1)
    
    return network, state_in

def get_training_config():
    """Define training configuration"""
    return {
        'StateSpaceState': {'transaction_cost': [0.001]},
        'ActionSpaceAction': create_action_space(),
        'StateSpaceNetworkSampleType': 'index',
        'StateSpaceEngineSampleConversion': 'index_to_dict',
        'ActionSpaceNetworkSampleType': 'exploration',
        'ActionSpaceEngineSampleConversion': 'index_to_dict',
        'AgentLearningRate': 0.001,
        'AgentEpochCounter': 'Counter_1',
        'AgentIterationCounter': 'Counter_2',
        'ExplorationCounter': 'Counter_3',
        'AgentIsUpdateNetwork': True,
        'ExperienceReplay': False,
        'ExperienceBufferBufferSize': 10000,
        'ExperienceBufferSamplingSize': 1,
        'ExperienceReplayFreq': 5,
        'RecorderDataField': ['NETWORK_ACTION', 'ENGINE_REWARD', 'ENGINE_RECORD'],
        'RecorderRecordFreq': 1,
        'Counter': {
            'Counter_1': {
                'name': 'Epoch',
                'start_num': 0,
                'end_num': 10,
                'step_size': 1,
                'n_buffer': 0,
                'is_descend': False,
                'print_freq': 1
            },
            'Counter_2': {
                'name': 'Iteration',
                'start_num': 0,
                'end_num': 10000,
                'step_size': 1,
                'n_buffer': 10000,
                'is_descend': False,
                'print_freq': 10000
            },
            'Counter_3': {
                'name': 'Exploration',
                'start_num': 1,
                'end_num': 0.1,
                'step_size': 0.0001,
                'n_buffer': 10000,
                'is_descend': True,
                'print_freq': None
            }
        }
    }

def plot_action_values(qvalues):
    """Plot the evolution of action values during training"""
    steps, action_value = qvalues.shape
    time_steps, dimensions = np.meshgrid(np.arange(steps), np.arange(action_value))
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(time_steps, dimensions, qvalues.T, cmap='jet')
    ax.set_title('Action Value Over Time')
    ax.set_xlabel('Training Step (x100)')
    ax.set_ylabel('Action Space')
    ax.set_zlabel('Action Value')
    ax.view_init(elev=80, azim=270)
    plt.show()

def calculate_performance_metrics(pnl):
    """Calculate comprehensive performance metrics"""
    metrics = {}
    
    # Basic return metrics
    metrics['Total Return'] = (pnl['Capital'].iloc[-1] - pnl['Capital'].iloc[0]) / pnl['Capital'].iloc[0]
    
    # Trading metrics
    metrics['Number of Trades'] = (pnl['N_Trade'] != 0).sum()
    metrics['Win Rate'] = (pnl['Trade_Profit'] > 0).sum() / metrics['Number of Trades']
    
    # Risk metrics
    returns = pnl['Capital'].pct_change().dropna()
    metrics['Volatility'] = returns.std() * np.sqrt(252)
    metrics['Sharpe Ratio'] = returns.mean() / returns.std() * np.sqrt(252)
    
    # Drawdown analysis
    cummax = pnl['Capital'].cummax()
    drawdown = (pnl['Capital'] - cummax) / cummax
    metrics['Max Drawdown'] = drawdown.min()
    
    # Position analysis
    metrics['Avg Position Size'] = pnl['Position_Value'].mean()
    metrics['Max Position Size'] = pnl['Position_Value'].max()
    
    return pd.Series(metrics)

def enhanced_backtest_model(session, agent, test_engine, opt_action, initial_capital=1000000):
    """Enhanced backtesting with capital management"""
    action_dict = agent.action_space.convert(opt_action, 'index_to_dict')
    indices = range(action_dict['n_hist'] + 1, len(test_engine.x) - action_dict['n_forward'], 10)
    
    # Initialize tracking dataframe
    pnl = pd.DataFrame()
    pnl['Time'] = test_engine.timestamp
    pnl['Capital'] = initial_capital
    pnl['Trade_Profit'] = 0
    pnl['Cost'] = 0
    pnl['N_Trade'] = 0
    pnl['Position_Value'] = 0
    pnl['Risk_Level'] = 0
    
    current_capital = initial_capital
    max_position_size = initial_capital * 0.1  # Max 10% per position
    
    for i in indices:
        test_engine.process(index=i, transaction_cost=0.001, **action_dict)
        trade_record = test_engine.record
        
        if trade_record is not None and len(trade_record) > 0:
            trade_record = pd.DataFrame(trade_record)
            
            # Position sizing based on current capital
            position_size = min(max_position_size, current_capital * 0.1)
            
            # Calculate trade metrics with position sizing
            trade_cost = trade_record.groupby('trade_time')['trade_cost'].sum() * position_size
            close_cost = trade_record.groupby('close_time')['close_cost'].sum() * position_size
            profit = trade_record.groupby('close_time')['profit'].sum() * position_size
            open_pos = trade_record.groupby('trade_time')['long_short'].sum()
            close_pos = trade_record.groupby('close_time')['long_short'].sum() * -1
            
            # Update PnL dataframe
            for time in trade_cost.index:
                idx = pnl[pnl['Time'] == time].index[0]
                pnl.loc[idx, 'Cost'] += trade_cost[time]
                pnl.loc[idx, 'N_Trade'] += open_pos[time]
                pnl.loc[idx, 'Position_Value'] += position_size
                current_capital -= trade_cost[time]
                pnl.loc[idx:, 'Capital'] = current_capital
            
            for time in close_cost.index:
                idx = pnl[pnl['Time'] == time].index[0]
                pnl.loc[idx, 'Cost'] += close_cost[time]
                pnl.loc[idx, 'Trade_Profit'] += profit[time]
                pnl.loc[idx, 'N_Trade'] += close_pos[time]
                pnl.loc[idx, 'Position_Value'] -= position_size
                current_capital += profit[time] - close_cost[time]
                pnl.loc[idx:, 'Capital'] = current_capital
    
    pnl['Risk_Level'] = pnl['Position_Value'] / pnl['Capital']
    return pnl

def save_model(session, rl_agent, model_path):
    """Save the trained model"""
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    rl_agent.saver.save(session, model_path)
    print(f"Model saved to {model_path}")

def load_model(session, rl_agent, model_path):
    """Load a trained model"""
    rl_agent.saver.restore(session, model_path)
    print(f"Model loaded from {model_path}")

def main():
    """Main execution function"""
    # Model paths
    MODEL_DIR = "models"
    MODEL_PATH = os.path.join(MODEL_DIR, "rl_trader_model")
    
    # Load and prepare data
    eg_train, eg_test = load_and_prepare_data('DATA/P.csv', 'DATA/Y.csv')
    
    # Create action space and get dimensions
    actions = create_action_space()
    n_action = int(np.product([len(actions[key]) for key in actions.keys()]))
    n_state = len({'transaction_cost': [0.001]})
    
    # Create network and training configuration
    network, _ = create_network(n_state, n_action)
    config = get_training_config()
    
    # Initialize RL agent
    rl_train = RL.Trader(network, config, eg_train)
    
    with tf.Session() as sess:
        # Check if model exists
        if os.path.exists(MODEL_PATH + ".index"):
            print("Loading existing model...")
            load_model(sess, rl_train, MODEL_PATH)
        else:
            print("Training new model...")
            rl_train.process(sess)
            save_model(sess, rl_train, MODEL_PATH)
        
        # Plot training results
        qvalue_ = np.array(rl_train.exploration.qvalue_)
        plot_action_values(qvalue_)
        
        # Analyze training results
        action = rl_train.recorder.record['NETWORK_ACTION']
        reward = rl_train.recorder.record['ENGINE_REWARD']
        print(f"Mean reward: {np.mean(reward)}")
        
        df1 = pd.DataFrame({'action': action, 'reward': reward})
        mean_reward = df1.groupby('action').mean()
        sns.distplot(mean_reward)
        plt.show()
        
        # Get optimal action and perform backtesting
        [opt_action] = sess.run([rl_train.output], feed_dict=rl_train.feed_dict)
        opt_action = np.argmax(opt_action)
        
        # Run enhanced backtesting
        pnl = enhanced_backtest_model(sess, rl_train, eg_test, opt_action)
        
        # Calculate and display performance metrics
        metrics = calculate_performance_metrics(pnl)
        print("\nBacktesting Results:")
        print("===================")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Plot backtesting results
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot capital curve
        ax1.plot(pnl['Time'], pnl['Capital'])
        ax1.set_title('Capital Curve')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Capital')
        
        # Plot position value
        ax2.plot(pnl['Time'], pnl['Position_Value'])
        ax2.set_title('Position Value Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Position Value')
        
        # Plot risk level
        ax3.plot(pnl['Time'], pnl['Risk_Level'])
        ax3.set_title('Risk Level Over Time')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Risk Level')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
