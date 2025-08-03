# 🐸 Frog Catcher AI
<img src="./frog.gif" alt=""/>

A fun game where an AI learns to catch falling insects using reinforcement learning!

## What is this?
This is a simple game where a frog tries to catch insects falling from the sky. But here's the cool part - the frog is controlled by an AI that learns how to play better over time!

## How it works
1. **The AI starts knowing nothing** - it moves randomly at first
2. **It learns by playing** - every time it catches an insect, it gets a reward
3. **It gets smarter** - after playing many times, it learns the best strategies
4. **It becomes a pro** - eventually it can catch insects really well!

## What you need
- Python 3.7 or higher
- These Python packages:
  - pygame (for the game graphics)
  - numpy (for math)
  - matplotlib (for showing learning progress)

## How to install
1. **Download the files** to your computer
2. **Open a terminal/command prompt** in the folder
3. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

## How to play
1. **Run the game:**
   ```
   python main_rl.py
   ```

2. **Choose what you want to do:**
   - **Train AI Agent** - Watch the AI learn to play
   - **Human Player Mode** - Play the game yourself with arrow keys
   - **Watch AI Play** - See the trained AI play the game
   - **Show Training Progress** - See graphs of how the AI improved


## How the AI learns
The AI uses something called "Q-Learning" which works like this:
1. **State** - The AI looks at where the frog and insect are
2. **Action** - It decides to move left, right, or stay still
3. **Reward** - It gets points for catching insects
4. **Learning** - It remembers what worked and does it again
The more it plays, the better it gets at predicting where insects will fall!

## Have fun!
This is a great way to see AI learning in action. Watch how the frog goes from random movements to skilled insect catching! 🎯

---
© 2025 • [arihara-sudhan.github.io](https://arihara-sudhan.github.io)
