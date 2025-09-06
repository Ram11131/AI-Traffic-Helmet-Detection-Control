import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Environment and State Definition (Same as before) ---
class TrafficEnv:
    def __init__(self):
        self.num_lanes = 4
        self.max_cars = 10
        self.state = np.random.randint(0, self.max_cars, self.num_lanes)
        self.num_actions = 4

    def reset(self):
        self.state = np.random.randint(0, self.max_cars, self.num_lanes)
        return tuple(self.state)

    def step(self, action):
        reward = -np.sum(self.state)
        
        if action == 0:
            cars_passed = min(self.state[0], 5)
            self.state[0] = max(0, self.state[0] - cars_passed)
            cars_passed = min(self.state[1], 5)
            self.state[1] = max(0, self.state[1] - cars_passed)
        elif action == 1:
            cars_passed = min(self.state[0], 2)
            self.state[0] = max(0, self.state[0] - cars_passed)
            cars_passed = min(self.state[1], 2)
            self.state[1] = max(0, self.state[1] - cars_passed)
        elif action == 2:
            cars_passed = min(self.state[2], 5)
            self.state[2] = max(0, self.state[2] - cars_passed)
            cars_passed = min(self.state[3], 5)
            self.state[3] = max(0, self.state[3] - cars_passed)
        elif action == 3:
            cars_passed = min(self.state[2], 2)
            self.state[2] = max(0, self.state[2] - cars_passed)
            self.state[3] = max(0, self.state[3] - cars_passed)

        for i in range(self.num_lanes):
            self.state[i] = min(self.max_cars, self.state[i] + np.random.randint(0, 3))

        done = False
        return tuple(self.state), reward, done

# --- 2. Q-Learning Algorithm with Plotting Data Collection ---
def q_learning_with_plots():
    env = TrafficEnv()
    q_table = {}

    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    epochs = 2000

    # Lists to store data for plotting
    reward_history = []
    q_value_history = []

    for episode in range(epochs):
        state = env.reset()
        
        # We'll run for a fixed number of steps per episode to simulate time.
        total_reward = 0
        for _ in range(50):
            if state not in q_table:
                q_table[state] = np.zeros(env.num_actions)

            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(env.num_actions)
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done = env.step(action)
            total_reward += reward

            if next_state not in q_table:
                q_table[next_state] = np.zeros(env.num_actions)

            old_value = q_table[state][action]
            next_max = np.max(q_table[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state][action] = new_value

            # Record Q-value for a specific state-action pair for plotting
            # Let's track the Q-value for state (5, 5, 2, 2) and action 0
            if state == (5, 5, 2, 2) and action == 0:
                q_value_history.append(q_table[state][action])
            
            state = next_state
        
        reward_history.append(total_reward)

    return q_table, reward_history, q_value_history

# --- 3. Main execution and Plot Generation ---
if __name__ == "__main__":
    print("Training the Q-learning agent...")
    q_table_trained, rewards, q_values = q_learning_with_plots()
    print("Training complete.")

    # A. Plotting the Cumulative Reward over episodes
    plt.figure(figsize=(12, 6))
    plt.plot(pd.Series(rewards).rolling(window=100).mean())
    plt.title("Cumulative Reward Over Episodes (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Average Cumulative Reward")
    plt.grid(True)
    plt.show()

    # B. Plotting the Q-Value of a specific state-action pair
    plt.figure(figsize=(12, 6))
    plt.plot(q_values)
    plt.title("Q-Value for State=(5,5,2,2) and Action=0")
    plt.xlabel("Update Step")
    plt.ylabel("Q-Value")
    plt.grid(True)
    plt.show()
