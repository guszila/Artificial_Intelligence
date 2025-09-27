# Complete this class for all parts of the project
import random
from pacman_module.game import Agent
from pacman_module.pacman import Directions

class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args

    def get_action(self, state, belief_state):
        """
        Given a pacman game state and a belief state,
                returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.
        - `belief_state`: a list of probability matrices.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """

        # XXX: Your code here to obtain bonus

        legal = state.getLegalActions()
        if not legal:
            return Directions.STOP
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # รวม belief ของผีทุกตัว
        combined = belief_state[0].copy()
        for b in belief_state[1:]:
            combined += b
        if combined.sum() == 0:
            return random.choice(legal)

        pac_pos = state.getPacmanPosition()
        width, height = state.getWalls().width, state.getWalls().height

        # คำนวณ Manhattan distance เอง
        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # แปลงทิศเป็นเวกเตอร์เอง
        dir_vec = {
            Directions.NORTH: (0, 1),
            Directions.SOUTH: (0, -1),
            Directions.EAST:  (1, 0),
            Directions.WEST:  (-1, 0)
        }

        # คำนวณ expected distance
        def expected_distance(pos):
            ex = 0.0
            for x in range(width):
                for y in range(height):
                    p = combined[x, y]
                    if p > 0.0:
                        ex += p * manhattan(pos, (x, y))
            return ex

        best_moves, best_val = [], float("inf")
        for a in legal:
            dx, dy = dir_vec[a]
            nxt = (pac_pos[0] + dx, pac_pos[1] + dy)
            val = expected_distance(nxt)
            if val < best_val:
                best_val, best_moves = val, [a]
            elif val == best_val:
                best_moves.append(a)

        return random.choice(best_moves) if best_moves else Directions.STOP
    
        # XXX: End of your code here to obtain bonus
