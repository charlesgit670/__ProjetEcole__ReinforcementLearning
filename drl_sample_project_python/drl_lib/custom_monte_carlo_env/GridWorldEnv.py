import numpy as np

from ..do_not_touch.contracts import SingleAgentEnv


class GridWorldEnv(SingleAgentEnv):
    def __init__(self):
        self.agent_pos = -1
        self.reset()

    def state_id(self) -> int:
        return str(self.agent_pos)

    def is_game_over(self) -> bool:
        return (self.agent_pos[0] == 0 and self.agent_pos[1] == 4) or (self.agent_pos[0] == 4 and self.agent_pos[1] == 4)

    def act_with_action_id(self, action_id: int):
        assert(not self.is_game_over())
        assert(action_id in self.available_actions_ids())
        if action_id == 0:
                self.agent_pos[1] -= 1
        elif action_id == 1:
                self.agent_pos[1] += 1
        elif action_id == 2:
                self.agent_pos[0] += 1
        elif action_id == 3:
                self.agent_pos[0] -= 1

    def score(self) -> float:
        if self.agent_pos[0] == 0 and self.agent_pos[1] == 4:
            return -1.0
        if self.agent_pos[0] == 4 and self.agent_pos[1] == 4:
            return 1.0
        return 0.0

    def available_actions_ids(self) -> list:
        if self.is_game_over():
            return []
        all_actions = [0, 1, 2, 3] # Gauche, Droite, Bas, Haut
        row, col = self.agent_pos[0], self.agent_pos[1]
        if row == 0:
            all_actions.remove(3)
        if row == 4:
            all_actions.remove(2)
        if col == 0:
            all_actions.remove(0)
        if col == 4:
            all_actions.remove(1)
        return all_actions

    def reset(self):
        self.agent_pos = [5 // 2, 5 // 2]

    def view(self):
        pass

    def reset_random(self):
        row_position = np.random.randint(0, 4)
        if row_position == 0 or row_position == 4:
            col_position = np.random.randint(0, 3)
        else:
            col_position = np.random.randint(0, 4)
        self.agent_pos = [row_position, col_position]


