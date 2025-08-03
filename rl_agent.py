import numpy as np
import random
import pickle
import os

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.3, discount_factor=0.9, epsilon=0.3):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table with zeros
        self.q_table = {}
        
        # Training statistics
        self.episode_scores = []
        self.episode_lengths = []
        self.epsilon_history = []
        
    def get_state_key(self, state):
        """Convert state tuple to string key for dictionary"""
        return str(state)
    
    def get_action(self, state):
        """Choose action using epsilon-greedy policy"""
        state_key = self.get_state_key(state)
        
        # Initialize Q-values for this state if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state_key])
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-values using Q-learning algorithm"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize Q-values if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # Q-learning update rule
        current_q = self.q_table[state_key][action]
        next_max_q = np.max(self.q_table[next_state_key])
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * next_max_q
        
        self.q_table[state_key][action] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self, decay_rate=0.9995, min_epsilon=0.01):
        """Decay epsilon for exploration vs exploitation balance"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
    
    def save_model(self, filename='frog_ai_model.pkl'):
        """Save the trained model"""
        model_data = {
            'q_table': self.q_table,
            'episode_scores': self.episode_scores,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='frog_ai_model.pkl'):
        """Load a trained model"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            self.q_table = model_data['q_table']
            self.episode_scores = model_data.get('episode_scores', [])
            self.episode_lengths = model_data.get('episode_lengths', [])
            self.epsilon_history = model_data.get('epsilon_history', [])
            print(f"Model loaded from {filename}")
            return True
        return False

class FrogGameEnvironment:
    def __init__(self, width=800, height=700):
        self.width = width
        self.height = height
        self.frog_width = 200
        self.frog_height = 200
        self.insect_width = 100
        self.insect_height = 100
        
        # Game state
        self.frog_x = 300
        self.insect_x = random.randint(-30, 630)
        self.insect_y = -60
        self.score = 0
        self.frame_count = 0
        self.game_over = False
        
        # Insect consistency
        self.current_insect_type = random.randint(1, 9)
        
        # Eating animation state
        self.eating = False
        self.eating_timer = 0
        self.eating_duration = 3  # frames to show eating animation
        
        # State discretization parameters - Much simpler and more effective
        self.position_bins = 10  # Reduced from 20 to 10 for better generalization
        
    def get_state(self):
        """Get current game state as a discrete tuple - Simplified and more effective"""
        # Discretize frog position (10 bins instead of 20)
        frog_bin = int(self.frog_x / (self.width / self.position_bins))
        frog_bin = max(0, min(self.position_bins - 1, frog_bin))
        
        # Discretize insect position (10 bins)
        insect_x_bin = int(self.insect_x / (self.width / self.position_bins))
        insect_x_bin = max(0, min(self.position_bins - 1, insect_x_bin))
        
        # Simple distance categories (much simpler)
        insect_y_distance = self.insect_y - 470
        if insect_y_distance < 0:
            distance_category = 0  # Insect above screen
        elif insect_y_distance < 100:
            distance_category = 1  # Insect far away
        elif insect_y_distance < 200:
            distance_category = 2  # Insect medium distance
        else:
            distance_category = 3  # Insect close
        
        # Simple alignment state
        alignment_error = abs(self.frog_x - self.insect_x)
        if alignment_error < 50:
            alignment_state = 0  # Well aligned
        elif self.frog_x < self.insect_x:
            alignment_state = 1  # Frog left of insect
        else:
            alignment_state = 2  # Frog right of insect
        
        return (frog_bin, insect_x_bin, distance_category, alignment_state)
    
    def step(self, action):
        """Execute one game step based on action"""
        # Actions: 0 = left, 1 = right, 2 = stay
        if action == 0:  # Left
            self.frog_x -= 30
            if self.frog_x < -170:
                self.frog_x = 750
        elif action == 1:  # Right
            self.frog_x += 30
            if self.frog_x > 770:
                self.frog_x = -150
        
        # Move insect down
        self.insect_y += 1
        
        # Check if insect goes off screen
        if self.insect_y > 727.5:
            self.insect_x = random.randint(-30, 630)
            self.insect_y = -60
            self.current_insect_type = random.randint(1, 9)
        
        # Handle eating animation
        if self.eating:
            self.eating_timer += 1
            if self.eating_timer >= self.eating_duration:
                self.eating = False
                self.eating_timer = 0
        
        # Much simpler and more effective reward system
        reward = 0
        
        # Check collision
        if (self.insect_x + 80 > self.frog_x and 
            self.insect_x < self.frog_x + 100 and 
            self.insect_y + 10 > 470 and 
            not self.eating):
            reward = 50  # High reward for catching
            self.score += 1
            self.insect_x = random.randint(-30, 630)
            self.insect_y = -60
            self.current_insect_type = random.randint(1, 9)
            self.eating = True
            self.eating_timer = 0
        
        # Simple alignment reward when insect is close
        insect_y_distance = self.insect_y - 470
        if 100 < insect_y_distance < 300:  # Insect is approaching
            alignment_error = abs(self.frog_x - self.insect_x)
            if alignment_error < 50:  # Well aligned
                reward += 1  # Small positive reward for good positioning
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.1
        
        # Get new state
        new_state = self.get_state()
        
        return new_state, reward, self.game_over
    
    def reset(self):
        """Reset the game environment"""
        self.frog_x = 300
        self.insect_x = random.randint(-30, 630)
        self.insect_y = -60
        self.score = 0
        self.frame_count = 0
        self.game_over = False
        self.current_insect_type = random.randint(1, 9)
        self.eating = False
        self.eating_timer = 0
        return self.get_state()
    
    def get_game_state(self):
        """Get current game state for rendering"""
        return {
            'frog_x': self.frog_x,
            'insect_x': self.insect_x,
            'insect_y': self.insect_y,
            'score': self.score,
            'frame_count': self.frame_count,
            'current_insect_type': self.current_insect_type,
            'eating': self.eating
        } 