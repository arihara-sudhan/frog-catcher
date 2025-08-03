import pygame
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from rl_agent import QLearningAgent, FrogGameEnvironment

pygame.init()

WIDTH = 800
HEIGHT = 700

# Game setup
pygame.display.set_caption('Frog Catcher - AI Learning Mode')
screen = pygame.display.set_mode((WIDTH, HEIGHT))
frog = pygame.image.load("assets/frog.png")
frog = pygame.transform.scale(frog, (200, 200))
eating_frog = pygame.image.load("assets/eating-frog.png")
eating_frog = pygame.transform.scale(eating_frog, (200, 200))

# Load animated frog frames for menu background (if available)
menu_frog_frames = []
try:
    import os
    frog_frames_dir = "assets/frog_frames"
    if os.path.exists(frog_frames_dir):
        frame_files = sorted([f for f in os.listdir(frog_frames_dir) if f.endswith('.png')])
        for frame_file in frame_files:
            frame_path = os.path.join(frog_frames_dir, frame_file)
            frame = pygame.image.load(frame_path)
            frame = pygame.transform.scale(frame, (WIDTH, HEIGHT))  # Medium size for better UI fit
            menu_frog_frames.append(frame)
        print(f"Loaded {len(menu_frog_frames)} animated frog frames for menu background")
    else:
        # Fallback to static frog
        menu_frog_frames = [pygame.transform.scale(frog, (300, 300))]
        print("Using static frog for menu background (run extract_frog_frames.py to extract animated frames)")
except Exception as e:
    print(f"Error loading frog frames: {e}")
    menu_frog_frames = [pygame.transform.scale(frog, (300, 300))]

insects = [
    pygame.transform.rotate(
        pygame.transform.scale(
            pygame.image.load(f"assets/poochi ({i}).png"), (100,100)), 180)
    for i in range(1, 10)
]

bg_frames = [
    pygame.transform.scale(
        pygame.image.load(f"assets/bg-frames/bg ({i}).jpg"),
        (WIDTH, HEIGHT))
    for i in range(1, 51)
]

# Font setup
font = pygame.font.Font('freesansbold.ttf', 32)
small_font = pygame.font.Font('freesansbold.ttf', 16)

# RL Agent setup
env = FrogGameEnvironment(WIDTH, HEIGHT)
agent = QLearningAgent(state_size=10*10*4*3, action_size=3)  # 3 actions: left, right, stay (simplified state space)

# Training parameters
TRAINING_EPISODES = 500  # Reduced for faster training
EPISODE_LENGTH = 500     # Shorter episodes for faster learning
RENDER_TRAINING = True
SAVE_INTERVAL = 50       # Save more frequently

# Game state
GAME_MODE = "training"  # "training", "human", "ai_play"
episode = 0
total_score = 0
best_score = 0

def render_game(game_state, episode_info=""):
    """Render the game with current state"""
    screen.fill((255, 255, 255))
    
    # Background
    frame_i = game_state['frame_count'] % len(bg_frames)
    screen.blit(bg_frames[frame_i], (0, 0))
    
    # Insect - use the specific insect type from environment
    insect_type = game_state.get('current_insect_type', 1)
    insect = insects[insect_type - 1]  # Convert 1-9 to 0-8 index
    screen.blit(insect, (game_state['insect_x'], game_state['insect_y']))
    
    # Frog - show eating animation if eating
    if game_state.get('eating', False):
        screen.blit(eating_frog, (game_state['frog_x'], 470))
    else:
        screen.blit(frog, (game_state['frog_x'], 470))
    
    # Score
    score_text = font.render(f'SCORE: {game_state["score"]}', True, (255, 255, 255))
    score_rect = score_text.get_rect()
    score_rect.center = (90, 20)
    screen.blit(score_text, score_rect)
    
    # Episode info
    if episode_info:
        episode_text = small_font.render(episode_info, True, (255, 255, 255))
        episode_rect = episode_text.get_rect()
        episode_rect.center = (400, 20)
        screen.blit(episode_text, episode_rect)
    
    # Mode indicator
    mode_text = small_font.render(f'Mode: {GAME_MODE.upper()}', True, (255, 255, 255))
    mode_rect = mode_text.get_rect()
    mode_rect.center = (700, 20)
    screen.blit(mode_text, mode_rect)
    
    pygame.display.flip()

def train_agent():
    """Train the RL agent"""
    global episode, total_score, best_score
    
    print("Starting AI training...")
    print("Press ESC to stop training and switch to human mode")
    
    for ep in range(TRAINING_EPISODES):
        state = env.reset()
        episode_reward = 0
        episode_score = 0
        
        for step in range(EPISODE_LENGTH):
            # Get action from agent
            action = agent.get_action(state)
            
            # Execute action
            next_state, reward, done = env.step(action)
            
            # Learn from experience
            agent.learn(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_score = env.score
            state = next_state
            
            # Render if enabled
            if RENDER_TRAINING and ep % 10 == 0:  # Render every 10th episode
                game_state = env.get_game_state()
                episode_info = f"Episode: {ep+1}/{TRAINING_EPISODES} | Epsilon: {agent.epsilon:.3f}"
                render_game(game_state, episode_info)
                
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return True
            
            if done:
                break
        
        # Update statistics
        agent.episode_scores.append(episode_score)
        agent.episode_lengths.append(step + 1)
        agent.epsilon_history.append(agent.epsilon)
        
        total_score += episode_score
        if episode_score > best_score:
            best_score = episode_score
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Print progress
        if (ep + 1) % 50 == 0:
            avg_score = total_score / (ep + 1)
            print(f"Episode {ep+1}/{TRAINING_EPISODES} - Score: {episode_score}, Avg: {avg_score:.1f}, Best: {best_score}, Epsilon: {agent.epsilon:.3f}")
        
        # Save model periodically
        if (ep + 1) % SAVE_INTERVAL == 0:
            agent.save_model()
    
    return True

def human_play():
    """Human player mode"""
    global GAME_MODE
    
    print("Human player mode - Use LEFT/RIGHT arrow keys to move")
    
    # Reset game state
    frog_x = 300
    insect_x = random.randint(-30, 630)
    insect_y = -60
    score = 0
    frame_i = 0
    current_insect_type = random.randint(1, 9)
    insect = insects[current_insect_type - 1]
    eaten = False
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True
                if event.key == pygame.K_RIGHT:
                    frog_x = frog_x + 30
                    if frog_x > 770:
                        frog_x = -150
                if event.key == pygame.K_LEFT:
                    frog_x = frog_x - 30
                    if frog_x < -170:
                        frog_x = 750
        
        # Game logic
        screen.fill((255, 255, 255))
        screen.blit(bg_frames[frame_i], (0, 0))
        frame_i += 1
        if frame_i >= len(bg_frames):
            frame_i = 0
        
        screen.blit(insect, (insect_x, insect_y))
        insect_y += 1
        
        # Score display
        text = font.render(f'SCORE: {score}', True, (255, 255, 255))
        textRect = text.get_rect()
        textRect.center = (90, 20)
        screen.blit(text, textRect)
        
        # Mode indicator
        mode_text = small_font.render('Mode: HUMAN', True, (255, 255, 255))
        mode_rect = mode_text.get_rect()
        mode_rect.center = (700, 20)
        screen.blit(mode_text, mode_rect)
        
        if insect_y > 727.5:
            insect_x = random.randint(-30, 630)
            insect_y = -60
            current_insect_type = random.randint(1, 9)
            insect = insects[current_insect_type - 1]
        
        if (insect_x + 80 > frog_x and insect_x < frog_x + 100 and insect_y + 10 > 470):
            score += 1
            insect_x = random.randint(-30, 630)
            insect_y = -60
            current_insect_type = random.randint(1, 9)
            insect = insects[current_insect_type - 1]
            eaten = True
        
        if eaten:
            screen.blit(eating_frog, (frog_x, 470))
            eaten = False
            pygame.display.flip()
            time.sleep(0.3)
        
        screen.blit(frog, (frog_x, 470))
        pygame.display.flip()
    
    return True

def ai_play():
    """AI play mode - watch trained agent"""
    global GAME_MODE
    
    print("AI play mode - Watch the trained agent play")
    print("Controls:")
    print("  ESC - Return to menu")
    print("  1-5 - Set speed (1=slowest, 5=fastest)")
    print("  SPACE - Pause/Resume")
    
    # Speed settings
    speeds = {
        pygame.K_1: 0.1,   # Very slow
        pygame.K_2: 0.05,  # Slow
        pygame.K_3: 0.02,  # Normal
        pygame.K_4: 0.01,  # Fast
        pygame.K_5: 0.005  # Very fast
    }
    current_speed = 0.02  # Default to normal speed
    paused = False
    
    # Load trained model if available
    if not agent.load_model():
        show_message("No trained model found.\nPlease train the AI first!", 3)
        return True
    
    # Set epsilon to 0 for pure exploitation
    agent.epsilon = 0
    
    state = env.reset()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True
                elif event.key in speeds:
                    current_speed = speeds[event.key]
                    print(f"Speed set to: {current_speed:.3f}s delay")
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
        
        if not paused:
            # Get AI action
            action = agent.get_action(state)
            
            # Execute action
            next_state, reward, done = env.step(action)
            state = next_state
            
            # Render
            game_state = env.get_game_state()
            episode_info = f"AI Score: {game_state['score']} | Speed: {current_speed:.3f}s"
            render_game(game_state, episode_info)
            
            
            if done:
                state = env.reset()
    
    return True

def show_menu():
    """Show main menu UI"""
    global GAME_MODE
    
    # Menu options
    menu_options = [
        "Train AI Agent",
        "Human Player Mode", 
        "Watch AI Play",
        "Show Training Progress",
        "Exit"
    ]
    
    selected_option = 0
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_option = (selected_option - 1) % len(menu_options)
                elif event.key == pygame.K_DOWN:
                    selected_option = (selected_option + 1) % len(menu_options)
                elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    # Execute selected option
                    if selected_option == 0:
                        GAME_MODE = "training"
                        return train_agent()
                    elif selected_option == 1:
                        GAME_MODE = "human"
                        return human_play()
                    elif selected_option == 2:
                        GAME_MODE = "ai_play"
                        return ai_play()
                    elif selected_option == 3:
                        show_training_progress()
                        return True
                    elif selected_option == 4:
                        return False
        
        # Render menu
        screen.fill((255, 255, 255))
        
        # Animated frog background
        if menu_frog_frames:
            frog_frame_index = (pygame.time.get_ticks() // 150) % len(menu_frog_frames)
            frog_frame = menu_frog_frames[frog_frame_index]
            frog_rect = frog_frame.get_rect()
            frog_rect.center = (WIDTH // 2, HEIGHT // 2 )
            screen.blit(frog_frame, frog_rect)
        
        # Semi-transparent overlay for better text readability
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(100)  # 40% transparency for better frog visibility
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        
        # Title
        title_font = pygame.font.Font('freesansbold.ttf', 48)
        title_text = title_font.render('Frog Catcher AI', True, (255, 255, 255))
        title_rect = title_text.get_rect()
        title_rect.center = (WIDTH // 2, 100)
        screen.blit(title_text, title_rect)
        
        # Menu options
        menu_font = pygame.font.Font('freesansbold.ttf', 20)
        menu_y_start = 150
        
        for i, option in enumerate(menu_options):
            color = (255, 255, 0) if i == selected_option else (255, 255, 255)
            text = menu_font.render(f"{i+1}. {option}", True, color)
            rect = text.get_rect()
            rect.center = (WIDTH // 2, menu_y_start + i * 30)
            screen.blit(text, rect)
            
            # Draw selection indicator
            if i == selected_option:
                pygame.draw.rect(screen, (255, 255, 0), rect.inflate(20, 10), 3)
                
        pygame.display.flip()
    
    return True

def show_message(message, duration=2):
    """Show a message on screen for specified duration"""
    start_time = pygame.time.get_ticks()
    
    while pygame.time.get_ticks() - start_time < duration * 1000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                return True
        
        # Render message
        screen.fill((255, 255, 255))
        
        # Background
        frame_i = pygame.time.get_ticks() // 100 % len(bg_frames)
        screen.blit(bg_frames[frame_i], (0, 0))
        
        # Message
        message_font = pygame.font.Font('freesansbold.ttf', 24)
        lines = message.split('\n')
        
        for i, line in enumerate(lines):
            text = message_font.render(line, True, (255, 255, 255))
            rect = text.get_rect()
            rect.center = (WIDTH // 2, HEIGHT // 2 - 50 + i * 40)
            screen.blit(text, rect)
        
        pygame.display.flip()
    
    return True

def show_training_progress():
    """Show training progress plots"""
    if not agent.episode_scores:
        # Show message in UI instead of console
        show_message("No training data available.\nTrain the AI first!", 3)
        return
    
    # Show plots
    plt.figure(figsize=(15, 5))
    
    # Score progression
    plt.subplot(1, 3, 1)
    plt.plot(agent.episode_scores)
    plt.title('Training Progress - Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    
    # Average score (rolling window)
    plt.subplot(1, 3, 2)
    window_size = 50
    if len(agent.episode_scores) >= window_size:
        rolling_avg = np.convolve(agent.episode_scores, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(agent.episode_scores)), rolling_avg)
        plt.title(f'Average Score (Window: {window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
        plt.grid(True)
    
    # Epsilon decay
    plt.subplot(1, 3, 3)
    plt.plot(agent.epsilon_history)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Show return message
    show_message("Press any key to return to menu", 2)

def main():
    """Main game loop"""
    # Try to load existing model
    agent.load_model()
    
    running = True
    while running:
        running = show_menu()
    
    # Save final model
    agent.save_model()
    pygame.quit()

if __name__ == "__main__":
    main() 