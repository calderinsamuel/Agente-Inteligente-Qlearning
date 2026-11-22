"""Algoritmo tabular Q-learning y evaluación reproducible."""
from dataclasses import dataclass
import numpy as np
from .gridworld import GridWorld


@dataclass
class Training:
    q_table: np.ndarray
    returns: list[float]
    lengths: list[int]
    success: list[int]
    epsilons: list[float]


def train(episodes=4000, seed=42, alpha=0.18, gamma=0.97, epsilon=1.0, decay=0.996, min_epsilon=0.04) -> Training:
    if episodes < 1 or not 0 <= gamma <= 1 or not 0 < alpha <= 1:
        raise ValueError('Revisa episodios, alpha y gamma.')
    rng = np.random.default_rng(seed)
    env = GridWorld()
    q = np.zeros((env.size * env.size, 4), dtype=np.float64)
    returns, lengths, success, epsilons = [], [], [], []
    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        current_eps = epsilon
        for step in range(env.max_steps):
            if rng.random() < current_eps:
                action = int(rng.integers(4))
            else:
                actions = np.flatnonzero(q[state] == q[state].max())
                action = int(rng.choice(actions))
            next_state, reward, done, info = env.step(action)
            td_target = reward + (0 if done else gamma * np.max(q[next_state]))
            q[state, action] += alpha * (td_target - q[state, action])
            total += reward
            state = next_state
            if done:
                break
        returns.append(total)
        lengths.append(step + 1)
        success.append(int(info['success']))
        epsilons.append(current_eps)
        epsilon = max(min_epsilon, epsilon * decay)
    return Training(q, returns, lengths, success, epsilons)


def evaluate(q_table: np.ndarray, episodes=50) -> dict:
    """Evaluación greedy; el entorno es determinista, todos los episodios parten de S."""
    env = GridWorld()
    successes, rewards, steps, paths = 0, [], [], []
    for _ in range(episodes):
        state = env.reset()
        path = [env.position]
        cumulative = 0.0
        for i in range(env.max_steps):
            action = int(np.argmax(q_table[state]))
            state, reward, done, info = env.step(action)
            cumulative += reward
            path.append(env.position)
            if done:
                break
        successes += int(info['success'])
        rewards.append(cumulative)
        steps.append(i + 1)
        paths.append(path)
    return {'success_rate': successes / episodes, 'avg_reward': float(np.mean(rewards)),
            'avg_steps': float(np.mean(steps)), 'example_path': paths[0]}


def policy_text(q_table: np.ndarray) -> str:
    env = GridWorld()
    arrows = '↑→↓←'
    lines = []
    for r in range(env.size):
        symbols = []
        for c in range(env.size):
            pos = (r, c)
            symbols.append('S' if pos == env.start else 'G' if pos == env.goal else
                           '■' if pos in env.walls else 'X' if pos in env.traps else
                           arrows[int(np.argmax(q_table[env.state_index(pos)]))])
        lines.append(' '.join(symbols))
    return '\n'.join(lines)
