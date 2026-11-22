"""Entorno discreto 8x8 para un agente Q-learning, sin dependencias de Gym."""
from dataclasses import dataclass

ACTION_NAMES = ('↑', '→', '↓', '←')
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass
class GridWorld:
    size: int = 8
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (7, 7)
    walls: frozenset = frozenset({(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (4, 1), (4, 2), (5, 5), (6, 5)})
    traps: frozenset = frozenset({(2, 5), (3, 6), (6, 3)})
    max_steps: int = 100

    def __post_init__(self):
        self.position = self.start
        self.steps = 0

    def state_index(self, position: tuple[int, int]) -> int:
        return position[0] * self.size + position[1]

    def reset(self) -> int:
        self.position = self.start
        self.steps = 0
        return self.state_index(self.position)

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        if action not in range(4):
            raise ValueError('La acción debe estar entre 0 y 3.')
        dr, dc = DIRECTIONS[action]
        r, c = self.position
        target = (r + dr, c + dc)
        hit_wall = (not 0 <= target[0] < self.size or not 0 <= target[1] < self.size or target in self.walls)
        if not hit_wall:
            self.position = target
        self.steps += 1
        goal = self.position == self.goal
        trap = self.position in self.traps
        truncated = self.steps >= self.max_steps
        reward = 10.0 if goal else -5.0 if trap else -0.25 if hit_wall else -0.04
        return self.state_index(self.position), reward, (goal or trap or truncated), {
            'success': goal, 'trap': trap, 'truncated': truncated, 'wall': hit_wall
        }

    def render_ascii(self, path=None) -> str:
        path = set(path or [])
        chars = []
        for r in range(self.size):
            line = []
            for c in range(self.size):
                p = (r, c)
                line.append('S' if p == self.start else 'G' if p == self.goal else
                            '█' if p in self.walls else 'X' if p in self.traps else
                            '·' if p in path else ' ')
            chars.append(' '.join(line))
        return '\n'.join(chars)
