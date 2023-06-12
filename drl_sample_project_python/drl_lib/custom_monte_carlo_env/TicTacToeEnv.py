import numpy as np

from ..do_not_touch.contracts import SingleAgentEnv


class TicTacToeEnv(SingleAgentEnv):
    def __init__(self, play_first=True):
        self.states = [0 for _ in range(9)]
        self.play_first = play_first
        self.reset()

    def state_id(self) -> str:
        return str(self.states)

    def is_game_over(self) -> bool:
        for i in range(0, 9, 3):
            if self.states[i] == self.states[i+1] == self.states[i+2] != 0:
                return True
        for i in range(3):
            if self.states[i] == self.states[i+3] == self.states[i+6] != 0:
                return True
        if self.states[0] == self.states[4] == self.states[8] != 0:
            return True
        if self.states[2] == self.states[4] == self.states[6] != 0:
            return True
        if not any(x == 0 for x in self.states):
            return True
        return False

    def act_with_action_id(self, action_id: int):
        assert(not self.is_game_over())
        assert(action_id in self.available_actions_ids())
        self.states[action_id] = 1 if self.play_first else 2
        # random policy for opponent
        if not self.is_game_over():
            self.states[np.random.choice(self.available_actions_ids())] = 2 if self.play_first else 1


    def score(self) -> float:
        for i in range(0, 9, 3):
            if self.states[i] == self.states[i + 1] == self.states[i + 2] != 0:
                return 1 if ((self.states[i] == 1 and self.play_first) or \
                            (self.states[i] == 2 and not self.play_first)) else -1
        for i in range(3):
            if self.states[i] == self.states[i + 3] == self.states[i + 6] != 0:
                return 1 if ((self.states[i] == 1 and self.play_first) or \
                            (self.states[i] == 2 and not self.play_first)) else -1
        if self.states[0] == self.states[4] == self.states[8] != 0:
            return 1 if ((self.states[i] == 1 and self.play_first) or \
                        (self.states[i] == 2 and not self.play_first)) else -1
        if self.states[2] == self.states[4] == self.states[6] != 0:
            return 1 if ((self.states[i] == 1 and self.play_first) or \
                        (self.states[i] == 2 and not self.play_first)) else -1
        return 0

    def available_actions_ids(self) -> list:
        aa = []
        if not self.is_game_over():
            for i, value in enumerate(self.states):
                if value == 0:
                    aa.append(i)
        return aa

    def reset(self):
        self.states = [0 for _ in range(9)]
        if not self.play_first:
            self.states[np.random.choice(self.available_actions_ids())] = 1

    def view(self):
        for cell in range(self.cells_count):
            print('X' if cell == self.agent_pos else '_', end='')
        print()

    # same than reset
    def reset_random(self):
        self.states = [0 for _ in range(9)]
        if not self.play_first:
            self.states[np.random.choice(self.available_actions_ids())] = 1


