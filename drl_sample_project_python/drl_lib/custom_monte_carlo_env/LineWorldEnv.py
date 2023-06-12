import numpy as np

from ..do_not_touch.contracts import SingleAgentEnv


class LineWorldEnv(SingleAgentEnv):
    def __init__(self, cells_count: int):
        self.cells_count = cells_count #state_space
        self.agent_pos = -1
        self.action_space = 2
        self.reset()

    def state_id(self) -> int:
        return self.agent_pos

    def is_game_over(self) -> bool:
        return self.agent_pos == 0 or self.agent_pos == self.cells_count - 1

    def act_with_action_id(self, action_id: int):
        assert(not self.is_game_over())
        assert(action_id in self.available_actions_ids())
        self.agent_pos += -1 if action_id == 0 else 1

    def score(self) -> float:
        if self.agent_pos == 0:
          return -1.0
        if self.agent_pos == self.cells_count - 1:
          return 1.0
        return 0.0

    def available_actions_ids(self) -> np.ndarray:
        return np.array([0, 1]) if not self.is_game_over() else np.array([])  # Left, Right

    def reset(self):
        self.agent_pos = self.cells_count // 2

    def view(self):
        for cell in range(self.cells_count):
            print('X' if cell == self.agent_pos else '_', end='')
        print()

    def reset_random(self):
        self.agent_pos = np.random.randint(1, self.cells_count)


