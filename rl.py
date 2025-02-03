"""
Reinforcement Learning Trading System
-----------------------------------
This script implements a reinforcement learning approach for pairs trading.
It uses the Engle-Granger cointegration method to identify trading opportunities
and trains an RL agent to optimize trading parameters.
"""

import warnings
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
    """
    Load price data and split into training and testing sets.
    
    Args:
        x_path (str): Path to first asset price data
        y_path (str): Path to second asset price data
        train_split (float): Fraction of data to use for training
        
    Returns:
        tuple: Training and testing cointegration objects
    """
    # Load price data
    x = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    
    # Clean and prepare data
    x, y = EGCointegration.clean_data(x, y, 'date', 'close')
    
    # Split into training and testing sets
    train_len = round(len(x) * train_split)
    idx_train = list(range(0, train_len))
    idx_test = list(range(train_len, len(x)))
    
    # Create cointegration objects
    eg_train = EGCointegration(x.iloc[idx_train, :], y.iloc[idx_train, :], 'date', 'close')
    eg_test = EGCointegration(x.iloc[idx_test, :], y.iloc[idx_test, :], 'date', 'close')
    
    return eg_train, eg_test

def create_action_space():
    """
    Define the action space for the reinforcement learning agent.
    
    Returns:
        dict: Action space parameters
    """
    return {
        'n_hist': list(np.arange(60, 601, 60)),      # Historical window size
        'n_forward': list(np.arange(120, 1201, 120)), # Trading window size
        'trade_th': list(np.arange(1, 5.1, 1)),      # Trading threshold
        'stop_loss': list(np.arange(1, 2.1, 0.5)),   # Stop loss threshold
        'cl': list(np.arange(0.05, 0.11, 0.05))      # Confidence level
    }

def create_network(n_state, n_action):
    """
    Create the neural network architecture for the RL agent.
    
    Args:
        n_state (int): Number of states
        n_action (int): Number of possible actions
        
    Returns:
        tuple: Network object and input placeholder
    """
    # Define network layers
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
    
    # Create network
    state_in = tf.placeholder(shape=[1], dtype=tf.int32)
    network = basics.Network(state_in)
    network.build_layers(one_hot)
    network.add_layer_duplicates(output_layer, 1)
    
    return network, state_in

def get_training_config():
    """
    Define the configuration for training the RL agent.
    
    Returns:
        dict: Training configuration parameters
    """
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
    """
    Plot the evolution of action values during training.
    
    Args:
        qvalues (numpy.array): Array of action values over time
    """
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

def backtest_model(session, agent, test_engine, opt_action):
    """
    Perform backtesting of the trained model.
    
    Args:
        session: TensorFlow session
        agent: Trained RL agent
        test_engine: Test dataset
        opt_action: Optimal action from training
        
    Returns:
        pandas.DataFrame: Backtesting results
    """
    action_dict = agent.action_space.convert(opt_action, 'index_to_dict')
    indices = range(action_dict['n_hist'] + 1, len(test_engine.x) - action_dict['n_forward'], 10)
    
    pnl = pd.DataFrame()
    pnl['Time'] = test_engine.timestamp
    pnl['Trade_Profit'] = 0
    pnl['Cost'] = 0
    pnl['N_Trade'] = 0
    
    for i in indices:
        test_engine.process(index=i, transaction_cost=0.001, **action_dict)
        trade_record = test_engine.record
        
        if trade_record is not None and len(trade_record) > 0:
            trade_record = pd.DataFrame(trade_record)
            
            # Calculate trade metrics
            trade_cost = trade_record.groupby('trade_time')['trade_cost'].sum()
            close_cost = trade_record.groupby('close_time')['close_cost'].sum()
            profit = trade_record.groupby('close_time')['profit'].sum()
            open_pos = trade_record.groupby('trade_time')['long_short'].sum()
            close_pos = trade_record.groupby('close_time')['long_short'].sum() * -1
            
            # Update PnL dataframe
            pnl.loc[pnl['Time'].isin(trade_cost.index), 'Cost'] += trade_cost.values
            pnl.loc[pnl['Time'].isin(close_cost.index), 'Cost'] += close_cost.values
            pnl.loc[pnl['Time'].isin(close_cost.index), 'Trade_Profit'] += profit.values
            pnl.loc[pnl['Time'].isin(trade_cost.index), 'N_Trade'] += open_pos.values
            pnl.loc[pnl['Time'].isin(close_cost.index), 'N_Trade'] += close_pos.values
    
    pnl['PnL'] = (pnl['Trade_Profit'] - pnl['Cost']).cumsum()
    return pnl

def main():
    """Main execution function"""
    
    # Load and prepare data
    eg_train, eg_test = load_and_prepare_data('DATA/P.csv', 'DATA/Y.csv')
    
    # Create action space and get dimensions
    actions = create_action_space()
    n_action = int(np.product([len(actions[key]) for key in actions.keys()]))
    n_state = len({'transaction_cost': [0.001]})
    
    # Create network and training configuration
    network, _ = create_network(n_state, n_action)
    config = get_training_config()
    
    # Initialize and train the RL agent
    rl_train = RL.Trader(network, config, eg_train)
    with tf.Session() as sess:
        rl_train.process(sess)
        
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
        
        pnl = backtest_model(sess, rl_train, eg_test, opt_action)
        
        # Plot backtesting results
        plt.figure(figsize=(12, 6))
        plt.plot(pnl['PnL'])
        plt.title('Cumulative Profit and Loss')
        plt.xlabel('Time')
        plt.ylabel('PnL')
        plt.legend(['Profit'])
        plt.show()

if __name__ == "__main__":
    main()