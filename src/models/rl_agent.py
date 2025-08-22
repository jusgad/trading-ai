import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from typing import Tuple, List, Optional
from loguru import logger
from config.config import config

class TradingEnvironment:
    """Trading environment for reinforcement learning"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000.0):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.position_value = 0
        self.max_steps = len(data) - 1
        
        # State features (normalized technical indicators)
        self.state_columns = [
            'close', 'volume', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 
            'stoch_k', 'stoch_d', 'atr', 'williams_r', 'cci', 'volatility'
        ]
        
        # Normalize data
        self.normalized_data = self._normalize_data()
        
    def _normalize_data(self) -> pd.DataFrame:
        """Normalize the data for better training"""
        normalized = self.data.copy()
        for col in self.state_columns:
            if col in normalized.columns:
                normalized[col] = (normalized[col] - normalized[col].mean()) / normalized[col].std()
        return normalized
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.current_step = config.LOOKBACK_PERIOD
        self.balance = self.initial_balance
        self.position = 0
        self.position_value = 0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        if self.current_step < config.LOOKBACK_PERIOD:
            # Pad with zeros if not enough historical data
            state = np.zeros((config.LOOKBACK_PERIOD, len(self.state_columns)))
            available_steps = self.current_step + 1
            start_idx = config.LOOKBACK_PERIOD - available_steps
            
            for i, col in enumerate(self.state_columns):
                if col in self.normalized_data.columns:
                    state[start_idx:, i] = self.normalized_data[col][:available_steps].values
        else:
            state = np.zeros((config.LOOKBACK_PERIOD, len(self.state_columns)))
            start_idx = self.current_step - config.LOOKBACK_PERIOD + 1
            
            for i, col in enumerate(self.state_columns):
                if col in self.normalized_data.columns:
                    state[:, i] = self.normalized_data[col][start_idx:self.current_step + 1].values
        
        # Add portfolio information
        portfolio_info = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position,  # Current position
            self.position_value / self.initial_balance if self.position_value else 0  # Normalized position value
        ])
        
        # Flatten state and concatenate with portfolio info
        flattened_state = state.flatten()
        return np.concatenate([flattened_state, portfolio_info])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute action and return new state, reward, done"""
        current_price = self.data.iloc[self.current_step]['close']
        previous_price = self.data.iloc[self.current_step - 1]['close'] if self.current_step > 0 else current_price
        
        # Calculate reward based on action and price movement
        reward = 0
        
        # Execute action: 0=hold, 1=buy, 2=sell
        if action == 1 and self.position <= 0:  # Buy signal
            if self.balance > current_price:
                shares_to_buy = self.balance // current_price
                self.position_value = shares_to_buy * current_price
                self.balance -= self.position_value
                self.position = 1
                
        elif action == 2 and self.position >= 0:  # Sell signal
            if self.position_value > 0:
                self.balance += self.position_value
                self.position_value = 0
                self.position = 0
        
        # Calculate reward based on portfolio performance
        if self.position == 1:  # Long position
            price_change = (current_price - previous_price) / previous_price
            reward = price_change * 100  # Scale reward
        elif self.position == 0:  # No position
            reward = 0
        
        # Penalty for excessive trading
        if action != 0:
            reward -= 0.1  # Small penalty for transaction costs
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_state(), reward, done
    
    def get_portfolio_value(self) -> float:
        """Get current total portfolio value"""
        current_price = self.data.iloc[self.current_step]['close']
        if self.position == 1:
            current_position_value = self.position_value * (current_price / self.data.iloc[self.current_step - 1]['close'])
            return self.balance + current_position_value
        return self.balance


class DQNAgent:
    """Deep Q-Network agent for trading decisions"""
    
    def __init__(self, state_size: int, action_size: int = 3):
        self.state_size = state_size
        self.action_size = action_size  # 0: hold, 1: buy, 2: sell
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = config.LEARNING_RATE
        
        # Build neural network
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        
    def _build_model(self) -> Model:
        """Build the neural network model"""
        model = Sequential([
            Dense(128, input_dim=self.state_size, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])
        
        # Current Q values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Next Q values from target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Update Q values
        for i in range(batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + 0.99 * np.max(next_q_values[i])
        
        # Train the model
        self.q_network.fit(states, current_q_values, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        self.q_network.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.q_network = tf.keras.models.load_model(filepath)
        self.target_network = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


class RLTrainer:
    """Trainer for the reinforcement learning agent"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.env = TradingEnvironment(data)
        
        # Calculate state size
        sample_state = self.env.reset()
        state_size = len(sample_state)
        
        self.agent = DQNAgent(state_size)
        
    def train(self, episodes: int = 1000) -> dict:
        """Train the agent"""
        scores = []
        portfolio_values = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            
            while True:
                action = self.agent.act(state)
                next_state, reward, done = self.env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Train the agent
            self.agent.replay()
            
            # Update target network periodically
            if episode % 10 == 0:
                self.agent.update_target_network()
            
            scores.append(total_reward)
            portfolio_values.append(self.env.get_portfolio_value())
            
            if episode % 100 == 0:
                avg_score = np.mean(scores[-100:])
                avg_portfolio = np.mean(portfolio_values[-100:])
                logger.info(f"Episode {episode}, Avg Score: {avg_score:.2f}, "
                          f"Avg Portfolio: ${avg_portfolio:.2f}, Epsilon: {self.agent.epsilon:.3f}")
        
        return {
            'scores': scores,
            'portfolio_values': portfolio_values,
            'final_epsilon': self.agent.epsilon
        }
    
    def get_signal(self, current_data: pd.DataFrame) -> dict:
        """Generate trading signal for current market data"""
        # Prepare environment with current data
        temp_env = TradingEnvironment(current_data)
        state = temp_env._get_state()
        
        # Get action from trained agent
        action = self.agent.act(state, training=False)
        
        # Get Q-values for confidence
        q_values = self.agent.q_network.predict(state.reshape(1, -1), verbose=0)[0]
        confidence = np.max(q_values) - np.min(q_values)
        
        # Map action to signal
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        
        return {
            'signal': action_map[action],
            'confidence': float(confidence),
            'q_values': {
                'hold': float(q_values[0]),
                'buy': float(q_values[1]),
                'sell': float(q_values[2])
            }
        }