"""
Deep Reinforcement Learning Assignment 2
HalfCheetah-v5 Implementation with Q-learning, DQN, and DDQN

This implementation covers:
- Q-learning with action space discretization
- Deep Q-Network (DQN)
- Double Deep Q-Network (DDQN)

Authors: [Your Team Name]
Date: January 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better compatibility
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from typing import List, Tuple, Dict
import pickle
import warnings
warnings.filterwarnings('ignore')

# Optional: seaborn for better plot styling
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    print("Seaborn not installed. Plots will use default matplotlib styling.")
    pass

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for the experiments"""
    
    # Environment settings
    ENV_NAME = "HalfCheetah-v5"
    MAX_EPISODES = 10000  # Start with 100 for testing (change to 10000 for final)
    MAX_STEPS_PER_EPISODE = 1000
    
    # Q-learning specific
    QLEARNING_BINS_PER_DIM = 5  # Number of discrete bins per action dimension
    QLEARNING_STATE_BINS = 10   # Number of bins for state discretization
    QLEARNING_LEARNING_RATE = 0.1
    QLEARNING_DISCOUNT_FACTOR = 0.99
    QLEARNING_EPSILON_START = 1.0
    QLEARNING_EPSILON_END = 0.01
    QLEARNING_EPSILON_DECAY = 0.995
    
    # DQN/DDQN specific
    DQN_LEARNING_RATE = 0.001
    DQN_DISCOUNT_FACTOR = 0.99
    DQN_EPSILON_START = 1.0
    DQN_EPSILON_END = 0.01
    DQN_EPSILON_DECAY = 0.995
    DQN_BATCH_SIZE = 64
    DQN_REPLAY_BUFFER_SIZE = 10000
    DQN_TARGET_UPDATE_FREQ = 10  # Update target network every N episodes
    DQN_HIDDEN_SIZE = 256
    
    # Evaluation
    EVAL_FREQ = 10  # Evaluate every N episodes
    SAVE_FREQ = 50  # Save checkpoints every N episodes

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def discretize_action_space(action_dim: int, bins_per_dim: int) -> np.ndarray:
    """
    Discretize continuous action space into discrete bins.
    
    Args:
        action_dim: Dimension of action space
        bins_per_dim: Number of bins per dimension
    
    Returns:
        Array of discrete action vectors
    """
    # Create bins for each dimension from -1 to 1
    bins = np.linspace(-1, 1, bins_per_dim)
    
    # Generate all combinations of discretized actions
    from itertools import product
    discrete_actions = list(product(bins, repeat=action_dim))
    
    return np.array(discrete_actions)


def discretize_state(state: np.ndarray, bins: int, bounds: Tuple) -> Tuple:
    """
    Discretize continuous state into bins for Q-table indexing.
    
    Args:
        state: Continuous state vector
        bins: Number of bins per dimension
        bounds: Tuple of (min_bound, max_bound) for state values
    
    Returns:
        Tuple of discretized state indices
    """
    min_bound, max_bound = bounds
    # Clip state to bounds
    state_clipped = np.clip(state, min_bound, max_bound)
    # Normalize to [0, 1]
    state_normalized = (state_clipped - min_bound) / (max_bound - min_bound)
    # Convert to bin indices
    state_indices = (state_normalized * (bins - 1)).astype(int)
    
    return tuple(state_indices)


# ============================================================================
# Q-LEARNING IMPLEMENTATION
# ============================================================================

class QLearningAgent:
    """
    Q-learning agent with discretized action and state spaces.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Config):
        """
        Initialize Q-learning agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration object
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Discretize action space
        self.discrete_actions = discretize_action_space(
            action_dim, config.QLEARNING_BINS_PER_DIM
        )
        self.n_actions = len(self.discrete_actions)
        
        print(f"Q-learning: Discretized {action_dim}D continuous actions into {self.n_actions} discrete actions")
        print(f"Action bins per dimension: {config.QLEARNING_BINS_PER_DIM}")
        
        # Initialize Q-table as dictionary (sparse representation)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        
        # State bounds for discretization (estimated for HalfCheetah)
        self.state_bounds = (-10, 10)
        
        # Exploration parameters
        self.epsilon = config.QLEARNING_EPSILON_START
        self.epsilon_end = config.QLEARNING_EPSILON_END
        self.epsilon_decay = config.QLEARNING_EPSILON_DECAY
        
        # Learning parameters
        self.alpha = config.QLEARNING_LEARNING_RATE
        self.gamma = config.QLEARNING_DISCOUNT_FACTOR
        
        # Statistics
        self.action_usage = defaultdict(int)
        self.action_rewards = defaultdict(list)
        
    def get_discrete_state(self, state: np.ndarray) -> Tuple:
        """Convert continuous state to discrete tuple for Q-table indexing."""
        return discretize_state(state, self.config.QLEARNING_STATE_BINS, self.state_bounds)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, np.ndarray]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
        
        Returns:
            Tuple of (action_index, continuous_action)
        """
        discrete_state = self.get_discrete_state(state)
        
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action_idx = np.random.randint(0, self.n_actions)
        else:
            # Exploit: best action from Q-table
            q_values = self.q_table[discrete_state]
            action_idx = np.argmax(q_values)
        
        # Record action usage
        self.action_usage[action_idx] += 1
        
        return action_idx, self.discrete_actions[action_idx]
    
    def update(self, state: np.ndarray, action_idx: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """
        Update Q-table using Q-learning update rule.
        
        Args:
            state: Current state
            action_idx: Action index taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        discrete_state = self.get_discrete_state(state)
        discrete_next_state = self.get_discrete_state(next_state)
        
        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        current_q = self.q_table[discrete_state][action_idx]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[discrete_next_state])
            target_q = reward + self.gamma * max_next_q
        
        # Update Q-value
        self.q_table[discrete_state][action_idx] += self.alpha * (target_q - current_q)
        
        # Record reward for this action
        self.action_rewards[action_idx].append(reward)
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ============================================================================
# DEEP Q-NETWORK (DQN) IMPLEMENTATION
# ============================================================================

class QNetwork(nn.Module):
    """
    Neural network for approximating Q-values.
    Outputs Q-values for discretized actions.
    """
    
    def __init__(self, state_dim: int, n_actions: int, hidden_size: int = 256):
        """
        Initialize Q-Network.
        
        Args:
            state_dim: Dimension of state space
            n_actions: Number of discrete actions
            hidden_size: Size of hidden layers
        """
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, n_actions)
        
    def forward(self, x):
        """Forward pass through network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action_idx, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action_idx, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        
        states, action_indices, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(action_indices),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Config, device='cpu'):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration object
            device: Device to run on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        
        # Discretize action space
        self.discrete_actions = discretize_action_space(
            action_dim, config.QLEARNING_BINS_PER_DIM
        )
        self.n_actions = len(self.discrete_actions)
        
        print(f"DQN: Discretized {action_dim}D continuous actions into {self.n_actions} discrete actions")
        
        # Q-networks
        self.policy_net = QNetwork(state_dim, self.n_actions, config.DQN_HIDDEN_SIZE).to(device)
        self.target_net = QNetwork(state_dim, self.n_actions, config.DQN_HIDDEN_SIZE).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.DQN_LEARNING_RATE)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.DQN_REPLAY_BUFFER_SIZE)
        
        # Exploration parameters
        self.epsilon = config.DQN_EPSILON_START
        self.epsilon_end = config.DQN_EPSILON_END
        self.epsilon_decay = config.DQN_EPSILON_DECAY
        
        # Statistics
        self.action_usage = defaultdict(int)
        self.training_losses = []
        self.q_values_history = []
        self.action_selection_history = []
        
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, np.ndarray]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
        
        Returns:
            Tuple of (action_index, continuous_action)
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action_idx = np.random.randint(0, self.n_actions)
        else:
            # Exploit: best action from policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.argmax().item()
                
                # Record Q-values for analysis
                if training:
                    self.q_values_history.append(q_values.max().item())
        
        # Record action usage
        self.action_usage[action_idx] += 1
        self.action_selection_history.append(action_idx)
        
        return action_idx, self.discrete_actions[action_idx]
    
    def update(self, batch_size: int):
        """
        Update policy network using experience replay.
        
        Args:
            batch_size: Size of batch to sample
        """
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch from replay buffer
        states, action_indices, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        action_indices = torch.LongTensor(action_indices).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.policy_net(states).gather(1, action_indices.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.config.DQN_DISCOUNT_FACTOR * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Record loss
        self.training_losses.append(loss.item())
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ============================================================================
# DOUBLE DQN (DDQN) IMPLEMENTATION
# ============================================================================

class DDQNAgent(DQNAgent):
    """
    Double Deep Q-Network agent.
    Uses policy network for action selection and target network for evaluation.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Config, device='cpu'):
        """Initialize DDQN agent (inherits from DQN)."""
        super().__init__(state_dim, action_dim, config, device)
        print(f"DDQN: Using Double DQN update rule")
    
    def update(self, batch_size: int):
        """
        Update policy network using Double DQN update rule.
        
        Args:
            batch_size: Size of batch to sample
        """
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch from replay buffer
        states, action_indices, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        action_indices = torch.LongTensor(action_indices).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.policy_net(states).gather(1, action_indices.unsqueeze(1))
        
        # Double DQN: Use policy network to select action, target network to evaluate
        with torch.no_grad():
            # Select best actions using policy network
            next_actions = self.policy_net(next_states).argmax(1)
            # Evaluate using target network
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.config.DQN_DISCOUNT_FACTOR * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Record loss
        self.training_losses.append(loss.item())


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_qlearning(env, agent: QLearningAgent, config: Config) -> Dict:
    """
    Train Q-learning agent.
    
    Args:
        env: Gymnasium environment
        agent: Q-learning agent
        config: Configuration object
    
    Returns:
        Dictionary containing training statistics
    """
    print("\n" + "="*80)
    print("TRAINING Q-LEARNING AGENT")
    print("="*80)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(config.MAX_EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0
        episode_length = 0
        
        for step in range(config.MAX_STEPS_PER_EPISODE):
            # Select action
            action_idx, action = agent.select_action(state, training=True)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update Q-table
            agent.update(state, action_idx, reward, next_state, done)
            
            # Update statistics
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
            if done:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{config.MAX_EPISODES} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'action_usage': dict(agent.action_usage),
        'action_rewards': {k: v for k, v in agent.action_rewards.items()},
        'q_table_size': len(agent.q_table)
    }


def train_dqn(env, agent: DQNAgent, config: Config, agent_type: str = "DQN") -> Dict:
    """
    Train DQN or DDQN agent.
    
    Args:
        env: Gymnasium environment
        agent: DQN/DDQN agent
        config: Configuration object
        agent_type: Type of agent ("DQN" or "DDQN")
    
    Returns:
        Dictionary containing training statistics
    """
    print("\n" + "="*80)
    print(f"TRAINING {agent_type} AGENT")
    print("="*80)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(config.MAX_EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0
        episode_length = 0
        
        for step in range(config.MAX_STEPS_PER_EPISODE):
            # Select action
            action_idx, action = agent.select_action(state, training=True)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience in replay buffer
            agent.replay_buffer.push(state, action_idx, reward, next_state, done)
            
            # Update network
            agent.update(config.DQN_BATCH_SIZE)
            
            # Update statistics
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
            if done:
                break
        
        # Update target network periodically
        if (episode + 1) % config.DQN_TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_loss = np.mean(agent.training_losses[-100:]) if agent.training_losses else 0
            print(f"Episode {episode+1}/{config.MAX_EPISODES} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Loss: {avg_loss:.6f} | "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'action_usage': dict(agent.action_usage),
        'training_losses': agent.training_losses,
        'q_values_history': agent.q_values_history,
        'action_selection_history': agent.action_selection_history
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_qlearning_analysis(results: Dict, agent: QLearningAgent, save_prefix: str = "qlearning"):
    """
    Create analysis plots for Q-learning.
    
    Args:
        results: Training results dictionary
        agent: Trained Q-learning agent
        save_prefix: Prefix for saved plot files
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Q-Learning Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Rewards
    ax = axes[0, 0]
    ax.plot(results['episode_rewards'], alpha=0.6, label='Episode Reward')
    window = 10
    if len(results['episode_rewards']) >= window:
        rolling_avg = np.convolve(results['episode_rewards'], 
                                  np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(results['episode_rewards'])), 
               rolling_avg, 'r-', linewidth=2, label=f'{window}-Episode Average')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Action Usage Statistics
    ax = axes[0, 1]
    action_counts = list(results['action_usage'].values())
    action_indices = list(results['action_usage'].keys())
    ax.bar(range(len(action_counts)), action_counts)
    ax.set_xlabel('Action Index')
    ax.set_ylabel('Usage Count')
    ax.set_title('Action Usage Statistics')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Reward Distribution per Discrete Action (Top 20 most used actions)
    ax = axes[1, 0]
    
    # Only plot top 20 most-used actions to avoid rendering issues
    top_actions = sorted(results['action_usage'].items(), key=lambda x: x[1], reverse=True)[:20]
    action_rewards_data = []
    action_labels = []
    
    for action_idx, _ in top_actions:
        if action_idx in results['action_rewards'] and len(results['action_rewards'][action_idx]) > 0:
            action_rewards_data.append(results['action_rewards'][action_idx])
            action_labels.append(f"{action_idx}")
    
    if action_rewards_data and len(action_rewards_data) > 0:
        try:
            bp = ax.boxplot(action_rewards_data, labels=action_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax.set_xlabel('Action Index (Top 20 Most Used)')
            ax.set_ylabel('Reward')
            ax.set_title('Reward Distribution per Discrete Action')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
        except Exception as e:
            print(f"Warning: Could not create boxplot: {e}")
            ax.text(0.5, 0.5, 'Boxplot rendering failed\nSee console for details', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel('Action Index')
            ax.set_ylabel('Reward')
            ax.set_title('Reward Distribution per Discrete Action')
    else:
        ax.text(0.5, 0.5, 'No reward data available', 
               ha='center', va='center', transform=ax.transAxes)
    
    # Plot 4: Episode Lengths
    ax = axes[1, 1]
    ax.plot(results['episode_lengths'], alpha=0.6)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length (steps)')
    ax.set_title('Episode Lengths Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    try:
        plt.savefig(f'{save_prefix}_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved Q-learning analysis plot to {save_prefix}_analysis.png")
    except Exception as e:
        print(f"Warning: Could not save plot to PNG file: {e}")
        print("Attempting to save with lower DPI...")
        try:
            plt.savefig(f'{save_prefix}_analysis.png', dpi=150, bbox_inches='tight')
            print(f"Saved Q-learning analysis plot to {save_prefix}_analysis.png (lower DPI)")
        except Exception as e2:
            print(f"Error: Could not save plot: {e2}")
            print("Continuing without saving plot...")
    
    plt.close()
    
    # Additional statistics
    print(f"\nQ-Learning Statistics:")
    print(f"  Total unique states visited: {results['q_table_size']}")
    print(f"  Total discrete actions: {agent.n_actions}")
    print(f"  Most used action: {max(results['action_usage'], key=results['action_usage'].get)}")
    print(f"  Average episode reward: {np.mean(results['episode_rewards']):.2f}")


def plot_dqn_analysis(results: Dict, agent: DQNAgent, save_prefix: str = "dqn"):
    """
    Create analysis plots for DQN/DDQN.
    
    Args:
        results: Training results dictionary
        agent: Trained DQN/DDQN agent
        save_prefix: Prefix for saved plot files
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{save_prefix.upper()} Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Rewards
    ax = axes[0, 0]
    ax.plot(results['episode_rewards'], alpha=0.6, label='Episode Reward')
    window = 10
    if len(results['episode_rewards']) >= window:
        rolling_avg = np.convolve(results['episode_rewards'], 
                                  np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(results['episode_rewards'])), 
               rolling_avg, 'r-', linewidth=2, label=f'{window}-Episode Average')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training Loss
    ax = axes[0, 1]
    if results['training_losses']:
        ax.plot(results['training_losses'], alpha=0.6)
        window = 100
        if len(results['training_losses']) >= window:
            rolling_avg = np.convolve(results['training_losses'], 
                                      np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(results['training_losses'])), 
                   rolling_avg, 'r-', linewidth=2)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Q-Values History
    ax = axes[0, 2]
    if results['q_values_history']:
        ax.plot(results['q_values_history'], alpha=0.6)
        window = 50
        if len(results['q_values_history']) >= window:
            rolling_avg = np.convolve(results['q_values_history'], 
                                      np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(results['q_values_history'])), 
                   rolling_avg, 'r-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Max Q-Value')
    ax.set_title('Predicted Q-Values')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Action Selection Frequency
    ax = axes[1, 0]
    action_counts = list(results['action_usage'].values())
    action_indices = list(results['action_usage'].keys())
    ax.bar(range(len(action_counts)), action_counts)
    ax.set_xlabel('Action Index')
    ax.set_ylabel('Selection Count')
    ax.set_title('Action Selection Frequency')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Action Selection Over Time (heatmap)
    ax = axes[1, 1]
    if results['action_selection_history']:
        # Create bins for episodes
        n_bins = min(50, len(results['action_selection_history']) // 10)
        actions_per_bin = len(results['action_selection_history']) // n_bins
        action_matrix = []
        
        for i in range(n_bins):
            start_idx = i * actions_per_bin
            end_idx = (i + 1) * actions_per_bin
            bin_actions = results['action_selection_history'][start_idx:end_idx]
            action_counts = [bin_actions.count(a) for a in range(agent.n_actions)]
            action_matrix.append(action_counts)
        
        action_matrix = np.array(action_matrix).T
        im = ax.imshow(action_matrix, aspect='auto', cmap='viridis')
        ax.set_xlabel('Time Bin')
        ax.set_ylabel('Action Index')
        ax.set_title('Action Selection Over Time')
        plt.colorbar(im, ax=ax, label='Count')
    
    # Plot 6: Episode Lengths
    ax = axes[1, 2]
    ax.plot(results['episode_lengths'], alpha=0.6)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length (steps)')
    ax.set_title('Episode Lengths')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    try:
        plt.savefig(f'{save_prefix}_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved {save_prefix.upper()} analysis plot to {save_prefix}_analysis.png")
    except Exception as e:
        print(f"Warning: Could not save plot to PNG file: {e}")
        print("Attempting to save with lower DPI...")
        try:
            plt.savefig(f'{save_prefix}_analysis.png', dpi=150, bbox_inches='tight')
            print(f"Saved {save_prefix.upper()} analysis plot to {save_prefix}_analysis.png (lower DPI)")
        except Exception as e2:
            print(f"Error: Could not save plot: {e2}")
            print("Continuing without saving plot...")
    
    plt.close()


def plot_comparison(qlearning_results: Dict, dqn_results: Dict, ddqn_results: Dict):
    """
    Create comparison plots for all three methods.
    
    Args:
        qlearning_results: Q-learning results
        dqn_results: DQN results
        ddqn_results: DDQN results
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Comparison: Q-Learning vs DQN vs DDQN', fontsize=16, fontweight='bold')
    
    # Plot 1: Cumulative Episode Returns
    ax = axes[0]
    window = 10
    
    # Q-learning
    ql_rewards = qlearning_results['episode_rewards']
    if len(ql_rewards) >= window:
        ql_avg = np.convolve(ql_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(ql_rewards)), ql_avg, 
               label='Q-Learning', linewidth=2)
    
    # DQN
    dqn_rewards = dqn_results['episode_rewards']
    if len(dqn_rewards) >= window:
        dqn_avg = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(dqn_rewards)), dqn_avg, 
               label='DQN', linewidth=2)
    
    # DDQN
    ddqn_rewards = ddqn_results['episode_rewards']
    if len(ddqn_rewards) >= window:
        ddqn_avg = np.convolve(ddqn_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(ddqn_rewards)), ddqn_avg, 
               label='DDQN', linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title(f'{window}-Episode Moving Average Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Final Performance Comparison
    ax = axes[1]
    methods = ['Q-Learning', 'DQN', 'DDQN']
    final_rewards = [
        np.mean(ql_rewards[-20:]),
        np.mean(dqn_rewards[-20:]),
        np.mean(ddqn_rewards[-20:])
    ]
    
    bars = ax.bar(methods, final_rewards, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Average Reward (Last 20 Episodes)')
    ax.set_title('Final Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom')
    
    plt.tight_layout()
    
    try:
        plt.savefig('comparison_all_methods.png', dpi=300, bbox_inches='tight')
        print("Saved comparison plot to comparison_all_methods.png")
    except Exception as e:
        print(f"Warning: Could not save plot to PNG file: {e}")
        print("Attempting to save with lower DPI...")
        try:
            plt.savefig('comparison_all_methods.png', dpi=150, bbox_inches='tight')
            print("Saved comparison plot to comparison_all_methods.png (lower DPI)")
        except Exception as e2:
            print(f"Error: Could not save plot: {e2}")
            print("Continuing without saving plot...")
    
    plt.close()


def create_final_summary_visualization(qlearning_results: Dict, dqn_results: Dict, 
                                       ddqn_results: Dict, qlearning_agent: QLearningAgent,
                                       dqn_agent: DQNAgent, ddqn_agent: DDQNAgent):
    """
    Create a comprehensive final summary visualization with all key metrics.
    
    Args:
        qlearning_results: Q-learning training results
        dqn_results: DQN training results
        ddqn_results: DDQN training results
        qlearning_agent: Trained Q-learning agent
        dqn_agent: Trained DQN agent
        ddqn_agent: Trained DDQN agent
    """
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('Deep Reinforcement Learning Assignment 2 - Final Summary', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # ========================================================================
    # Row 1: Learning Curves Comparison
    # ========================================================================
    
    # Plot 1: Q-Learning Learning Curve
    ax1 = fig.add_subplot(gs[0, 0])
    ql_rewards = qlearning_results['episode_rewards']
    ax1.plot(ql_rewards, alpha=0.4, color='#1f77b4')
    window = 10
    if len(ql_rewards) >= window:
        ql_smooth = np.convolve(ql_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(ql_rewards)), ql_smooth, 
                color='#1f77b4', linewidth=2, label=f'{window}-ep avg')
    ax1.set_title('Q-Learning Performance', fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: DQN Learning Curve
    ax2 = fig.add_subplot(gs[0, 1])
    dqn_rewards = dqn_results['episode_rewards']
    ax2.plot(dqn_rewards, alpha=0.4, color='#ff7f0e')
    if len(dqn_rewards) >= window:
        dqn_smooth = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(dqn_rewards)), dqn_smooth, 
                color='#ff7f0e', linewidth=2, label=f'{window}-ep avg')
    ax2.set_title('DQN Performance', fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: DDQN Learning Curve
    ax3 = fig.add_subplot(gs[0, 2])
    ddqn_rewards = ddqn_results['episode_rewards']
    ax3.plot(ddqn_rewards, alpha=0.4, color='#2ca02c')
    if len(ddqn_rewards) >= window:
        ddqn_smooth = np.convolve(ddqn_rewards, np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(ddqn_rewards)), ddqn_smooth, 
                color='#2ca02c', linewidth=2, label=f'{window}-ep avg')
    ax3.set_title('DDQN Performance', fontweight='bold')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reward')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: All Methods Comparison
    ax4 = fig.add_subplot(gs[0, 3])
    if len(ql_rewards) >= window:
        ax4.plot(range(window-1, len(ql_rewards)), ql_smooth, 
                label='Q-Learning', linewidth=2, color='#1f77b4')
    if len(dqn_rewards) >= window:
        ax4.plot(range(window-1, len(dqn_rewards)), dqn_smooth, 
                label='DQN', linewidth=2, color='#ff7f0e')
    if len(ddqn_rewards) >= window:
        ax4.plot(range(window-1, len(ddqn_rewards)), ddqn_smooth, 
                label='DDQN', linewidth=2, color='#2ca02c')
    ax4.set_title('Methods Comparison', fontweight='bold')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Reward (10-ep avg)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ========================================================================
    # Row 2: Performance Metrics
    # ========================================================================
    
    # Plot 5: Final Performance Bar Chart
    ax5 = fig.add_subplot(gs[1, 0])
    methods = ['Q-Learning', 'DQN', 'DDQN']
    final_rewards = [
        np.mean(ql_rewards[-20:]),
        np.mean(dqn_rewards[-20:]),
        np.mean(ddqn_rewards[-20:])
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax5.bar(methods, final_rewards, color=colors)
    ax5.set_ylabel('Average Reward')
    ax5.set_title('Final Performance\n(Last 20 Episodes)', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Best Performance Achieved
    ax6 = fig.add_subplot(gs[1, 1])
    best_rewards = [
        np.max(ql_rewards),
        np.max(dqn_rewards),
        np.max(ddqn_rewards)
    ]
    bars = ax6.bar(methods, best_rewards, color=colors)
    ax6.set_ylabel('Best Reward')
    ax6.set_title('Peak Performance\n(Single Episode)', fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 7: Training Stability (Standard Deviation)
    ax7 = fig.add_subplot(gs[1, 2])
    stability = [
        np.std(ql_rewards[-50:]) if len(ql_rewards) >= 50 else np.std(ql_rewards),
        np.std(dqn_rewards[-50:]) if len(dqn_rewards) >= 50 else np.std(dqn_rewards),
        np.std(ddqn_rewards[-50:]) if len(ddqn_rewards) >= 50 else np.std(ddqn_rewards)
    ]
    bars = ax7.bar(methods, stability, color=colors)
    ax7.set_ylabel('Std Dev of Reward')
    ax7.set_title('Training Stability\n(Lower is Better)', fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 8: Performance Statistics Table
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    
    stats_data = [
        ['Metric', 'Q-Learning', 'DQN', 'DDQN'],
        ['Final Avg', f'{final_rewards[0]:.1f}', f'{final_rewards[1]:.1f}', f'{final_rewards[2]:.1f}'],
        ['Best', f'{best_rewards[0]:.1f}', f'{best_rewards[1]:.1f}', f'{best_rewards[2]:.1f}'],
        ['Std Dev', f'{stability[0]:.1f}', f'{stability[1]:.1f}', f'{stability[2]:.1f}'],
        ['Actions Used', f'{len(qlearning_results["action_usage"])}', 
         f'{len(dqn_results["action_usage"])}', f'{len(ddqn_results["action_usage"])}']
    ]
    
    table = ax8.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#cccccc')
        table[(0, i)].set_text_props(weight='bold')
    
    ax8.set_title('Performance Summary Table', fontweight='bold', pad=20)
    
    # ========================================================================
    # Row 3: Action Space Analysis and Training Loss
    # ========================================================================
    
    # Plot 9: Action Usage Distribution (Q-Learning)
    ax9 = fig.add_subplot(gs[2, 0])
    ql_action_counts = list(qlearning_results['action_usage'].values())
    ax9.hist(ql_action_counts, bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax9.set_xlabel('Times Action Used')
    ax9.set_ylabel('Number of Actions')
    ax9.set_title('Q-Learning Action Usage\nDistribution', fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='y')
    ax9.set_yscale('log')
    
    # Plot 10: Top 10 Most Used Actions Comparison
    ax10 = fig.add_subplot(gs[2, 1])
    
    # Get top 10 actions for each method
    ql_top = sorted(qlearning_results['action_usage'].items(), key=lambda x: x[1], reverse=True)[:10]
    dqn_top = sorted(dqn_results['action_usage'].items(), key=lambda x: x[1], reverse=True)[:10]
    ddqn_top = sorted(ddqn_results['action_usage'].items(), key=lambda x: x[1], reverse=True)[:10]
    
    x = np.arange(10)
    width = 0.25
    
    ax10.bar(x - width, [count for _, count in ql_top], width, label='Q-Learning', color='#1f77b4')
    ax10.bar(x, [count for _, count in dqn_top], width, label='DQN', color='#ff7f0e')
    ax10.bar(x + width, [count for _, count in ddqn_top], width, label='DDQN', color='#2ca02c')
    
    ax10.set_xlabel('Action Rank')
    ax10.set_ylabel('Usage Count')
    ax10.set_title('Top 10 Actions Usage\nComparison', fontweight='bold')
    ax10.set_xticks(x)
    ax10.set_xticklabels([f'{i+1}' for i in range(10)])
    ax10.legend()
    ax10.grid(True, alpha=0.3, axis='y')
    
    # Plot 11: DQN Training Loss
    ax11 = fig.add_subplot(gs[2, 2])
    if dqn_results['training_losses']:
        ax11.plot(dqn_results['training_losses'], alpha=0.4, color='#ff7f0e')
        loss_window = 100
        if len(dqn_results['training_losses']) >= loss_window:
            loss_smooth = np.convolve(dqn_results['training_losses'], 
                                     np.ones(loss_window)/loss_window, mode='valid')
            ax11.plot(range(loss_window-1, len(dqn_results['training_losses'])), 
                     loss_smooth, color='#ff7f0e', linewidth=2, label=f'{loss_window}-step avg')
    ax11.set_xlabel('Update Step')
    ax11.set_ylabel('Loss')
    ax11.set_title('DQN Training Loss', fontweight='bold')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # Plot 12: DDQN Training Loss
    ax12 = fig.add_subplot(gs[2, 3])
    if ddqn_results['training_losses']:
        ax12.plot(ddqn_results['training_losses'], alpha=0.4, color='#2ca02c')
        if len(ddqn_results['training_losses']) >= loss_window:
            loss_smooth = np.convolve(ddqn_results['training_losses'], 
                                     np.ones(loss_window)/loss_window, mode='valid')
            ax12.plot(range(loss_window-1, len(ddqn_results['training_losses'])), 
                     loss_smooth, color='#2ca02c', linewidth=2, label=f'{loss_window}-step avg')
    ax12.set_xlabel('Update Step')
    ax12.set_ylabel('Loss')
    ax12.set_title('DDQN Training Loss', fontweight='bold')
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    
    # Save the comprehensive summary
    try:
        plt.savefig('final_summary_comprehensive.png', dpi=300, bbox_inches='tight')
        print("\n" + "="*80)
        print("✓ SAVED: Comprehensive final summary to 'final_summary_comprehensive.png'")
        print("="*80)
    except Exception as e:
        print(f"Warning: Could not save comprehensive summary at 300 DPI: {e}")
        try:
            plt.savefig('final_summary_comprehensive.png', dpi=150, bbox_inches='tight')
            print("\n" + "="*80)
            print("✓ SAVED: Comprehensive final summary to 'final_summary_comprehensive.png' (lower DPI)")
            print("="*80)
        except Exception as e2:
            print(f"Error: Could not save comprehensive summary: {e2}")
    
    plt.close()
    
    # Create a simpler summary for quick reference
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Quick Summary - Key Results', fontsize=16, fontweight='bold')
    
    # Subplot 1: Learning curves comparison
    ax = axes[0, 0]
    if len(ql_rewards) >= window:
        ax.plot(range(window-1, len(ql_rewards)), ql_smooth, 
               label='Q-Learning', linewidth=2.5, color='#1f77b4')
    if len(dqn_rewards) >= window:
        ax.plot(range(window-1, len(dqn_rewards)), dqn_smooth, 
               label='DQN', linewidth=2.5, color='#ff7f0e')
    if len(ddqn_rewards) >= window:
        ax.plot(range(window-1, len(ddqn_rewards)), ddqn_smooth, 
               label='DDQN', linewidth=2.5, color='#2ca02c')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Learning Curves Comparison', fontweight='bold', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Subplot 2: Final performance comparison
    ax = axes[0, 1]
    bars = ax.bar(methods, final_rewards, color=colors, width=0.6)
    ax.set_ylabel('Average Reward (Last 20 Ep)', fontsize=12)
    ax.set_title('Final Performance', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        # Highlight the winner
        if height == max(final_rewards):
            bar.set_edgecolor('gold')
            bar.set_linewidth(3)
    
    # Subplot 3: Improvement over episodes
    ax = axes[1, 0]
    segments = 10  # Divide training into segments
    segment_size = len(ql_rewards) // segments
    
    ql_segments = [np.mean(ql_rewards[i*segment_size:(i+1)*segment_size]) for i in range(segments)]
    dqn_segments = [np.mean(dqn_rewards[i*segment_size:(i+1)*segment_size]) for i in range(segments)]
    ddqn_segments = [np.mean(ddqn_rewards[i*segment_size:(i+1)*segment_size]) for i in range(segments)]
    
    x_segments = range(1, segments + 1)
    ax.plot(x_segments, ql_segments, 'o-', label='Q-Learning', linewidth=2, markersize=8, color='#1f77b4')
    ax.plot(x_segments, dqn_segments, 's-', label='DQN', linewidth=2, markersize=8, color='#ff7f0e')
    ax.plot(x_segments, ddqn_segments, '^-', label='DDQN', linewidth=2, markersize=8, color='#2ca02c')
    ax.set_xlabel('Training Phase', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Performance Progression', fontweight='bold', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Subplot 4: Key statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    TRAINING SUMMARY
    {'='*50}
    
    Q-LEARNING:
    • Final Avg Reward: {final_rewards[0]:.2f}
    • Best Episode: {np.max(ql_rewards):.2f}
    • Unique Actions: {len(qlearning_results['action_usage'])}
    • States Visited: {qlearning_results['q_table_size']}
    
    DQN:
    • Final Avg Reward: {final_rewards[1]:.2f}
    • Best Episode: {np.max(dqn_rewards):.2f}
    • Unique Actions: {len(dqn_results['action_usage'])}
    • Final Loss: {np.mean(dqn_results['training_losses'][-100:]):.6f}
    
    DDQN:
    • Final Avg Reward: {final_rewards[2]:.2f}
    • Best Episode: {np.max(ddqn_rewards):.2f}
    • Unique Actions: {len(ddqn_results['action_usage'])}
    • Final Loss: {np.mean(ddqn_results['training_losses'][-100:]):.6f}
    
    WINNER: {methods[np.argmax(final_rewards)]} 🏆
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
           family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    try:
        plt.savefig('final_summary_quick.png', dpi=300, bbox_inches='tight')
        print("✓ SAVED: Quick summary to 'final_summary_quick.png'")
    except Exception as e:
        print(f"Warning: Could not save quick summary at 300 DPI: {e}")
        try:
            plt.savefig('final_summary_quick.png', dpi=150, bbox_inches='tight')
            print("✓ SAVED: Quick summary to 'final_summary_quick.png' (lower DPI)")
        except Exception as e2:
            print(f"Error: Could not save quick summary: {e2}")
    
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("DEEP REINFORCEMENT LEARNING ASSIGNMENT 2")
    print("HalfCheetah-v5 with Q-Learning, DQN, and DDQN")
    print("="*80)
    
    config = Config()
    
    # Create environment
    print("\nCreating HalfCheetah-v5 environment...")
    env = gym.make('HalfCheetah-v5')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action space bounds: [{env.action_space.low[0]:.2f}, {env.action_space.high[0]:.2f}]")
    
    # ========================================================================
    # Q-LEARNING
    # ========================================================================
    
    print("\n" + "="*80)
    print("PART 1: Q-LEARNING")
    print("="*80)
    
    qlearning_agent = QLearningAgent(state_dim, action_dim, config)
    qlearning_results = train_qlearning(env, qlearning_agent, config)
    plot_qlearning_analysis(qlearning_results, qlearning_agent, "qlearning")
    
    # ========================================================================
    # DQN
    # ========================================================================
    
    print("\n" + "="*80)
    print("PART 2: DQN")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    dqn_agent = DQNAgent(state_dim, action_dim, config, device)
    dqn_results = train_dqn(env, dqn_agent, config, "DQN")
    plot_dqn_analysis(dqn_results, dqn_agent, "dqn")
    
    # ========================================================================
    # DDQN
    # ========================================================================
    
    print("\n" + "="*80)
    print("PART 3: DDQN")
    print("="*80)
    
    ddqn_agent = DDQNAgent(state_dim, action_dim, config, device)
    ddqn_results = train_dqn(env, ddqn_agent, config, "DDQN")
    plot_dqn_analysis(ddqn_results, ddqn_agent, "ddqn")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    
    print("\n" + "="*80)
    print("COMPARISON OF ALL METHODS")
    print("="*80)
    
    plot_comparison(qlearning_results, dqn_results, ddqn_results)
    
    # ========================================================================
    # FINAL COMPREHENSIVE SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE FINAL SUMMARY VISUALIZATIONS")
    print("="*80)
    
    create_final_summary_visualization(qlearning_results, dqn_results, ddqn_results,
                                      qlearning_agent, dqn_agent, ddqn_agent)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print("\nQ-Learning:")
    print(f"  Average reward (last 20 episodes): {np.mean(qlearning_results['episode_rewards'][-20:]):.2f}")
    print(f"  Best episode reward: {np.max(qlearning_results['episode_rewards']):.2f}")
    print(f"  Total unique states visited: {qlearning_results['q_table_size']}")
    print(f"  Total discrete actions: {qlearning_agent.n_actions}")
    print(f"  Actions actually used: {len(qlearning_results['action_usage'])}")
    print(f"  Exploration coverage: {len(qlearning_results['action_usage']) / qlearning_agent.n_actions * 100:.2f}%")
    
    print("\nDQN:")
    print(f"  Average reward (last 20 episodes): {np.mean(dqn_results['episode_rewards'][-20:]):.2f}")
    print(f"  Best episode reward: {np.max(dqn_results['episode_rewards']):.2f}")
    print(f"  Final training loss: {np.mean(dqn_results['training_losses'][-100:]):.6f}")
    print(f"  Actions actually used: {len(dqn_results['action_usage'])}")
    print(f"  Exploration coverage: {len(dqn_results['action_usage']) / dqn_agent.n_actions * 100:.2f}%")
    
    print("\nDDQN:")
    print(f"  Average reward (last 20 episodes): {np.mean(ddqn_results['episode_rewards'][-20:]):.2f}")
    print(f"  Best episode reward: {np.max(ddqn_results['episode_rewards']):.2f}")
    print(f"  Final training loss: {np.mean(ddqn_results['training_losses'][-100:]):.6f}")
    print(f"  Actions actually used: {len(ddqn_results['action_usage'])}")
    print(f"  Exploration coverage: {len(ddqn_results['action_usage']) / ddqn_agent.n_actions * 100:.2f}%")
    
    # Determine winner
    final_rewards = [
        np.mean(qlearning_results['episode_rewards'][-20:]),
        np.mean(dqn_results['episode_rewards'][-20:]),
        np.mean(ddqn_results['episode_rewards'][-20:])
    ]
    methods = ['Q-Learning', 'DQN', 'DDQN']
    winner_idx = np.argmax(final_rewards)
    
    print("\n" + "="*80)
    print(f"🏆 WINNER: {methods[winner_idx]} with average reward of {final_rewards[winner_idx]:.2f}")
    print("="*80)
    
    print("\n" + "="*80)
    print("OUTPUT FILES GENERATED:")
    print("="*80)
    print("  1. qlearning_analysis.png - Q-Learning detailed analysis")
    print("  2. dqn_analysis.png - DQN detailed analysis")
    print("  3. ddqn_analysis.png - DDQN detailed analysis")
    print("  4. comparison_all_methods.png - Basic comparison")
    print("  5. final_summary_comprehensive.png - Comprehensive summary (12 plots)")
    print("  6. final_summary_quick.png - Quick reference summary (4 plots)")
    print("="*80)
    
    env.close()
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
