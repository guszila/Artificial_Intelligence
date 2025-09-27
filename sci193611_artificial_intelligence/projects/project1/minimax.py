# B6644673 ภาณุเดช ศรีวุฒิทรัพย์  
# B6638375 กิตติธัช จังพนาสิน


from pacman_module.game import Agent, Directions
from pacman_module.pacman import GameState

INF = float("inf")


class PacmanAgent(Agent):
    """
    Pacman Agent ที่ใช้ Minimax (depth-limited + alpha-beta pruning)
    """

    def __init__(self, depth: str = "2"):
        # ความลึกสูงสุดของการค้นหา (default = 2)
        self.depth_limit = int(depth)
        self.order = [Directions.NORTH, Directions.EAST,
                      Directions.SOUTH, Directions.WEST, Directions.STOP]

    def get_action(self, state: GameState):
        """
        เลือก action ที่ดีที่สุดสำหรับ Pacman โดยใช้ minimax
        """
        best_score, best_action = -INF, None

        # ลองทุก action ที่ Pacman ทำได้
        for act in sorted(state.getLegalActions(0), key=self.order.index):
            succ = state.generateSuccessor(0, act)
            score = self._min_layer(succ, 0, 1, -INF, INF)
            if score > best_score:
                best_score, best_action = score, act

        return best_action or Directions.STOP

    # ---------- layer ของ Pacman (MAX) ----------
    def _max_layer(self, state: GameState, depth: int, alpha: float, beta: float):
        if self._cutoff(state, depth):
            return self._evaluate(state)

        value = -INF
        for act in sorted(state.getLegalActions(0), key=self.order.index):
            succ = state.generateSuccessor(0, act)
            value = max(value, self._min_layer(succ, depth, 1, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    # ---------- layer ของ Ghost (MIN) ----------
    def _min_layer(self, state: GameState, depth: int, agent_index: int, alpha: float, beta: float):
        if self._cutoff(state, depth):
            return self._evaluate(state)

        value = INF
        num_agents = state.getNumAgents()

        for act in sorted(state.getLegalActions(agent_index), key=self.order.index):
            succ = state.generateSuccessor(agent_index, act)

            # agent ถัดไป (Pacman = 0)
            next_agent = (agent_index + 1) % num_agents
            next_depth = depth + 1 if next_agent == 0 else depth

            if next_agent == 0:
                value = min(value, self._max_layer(succ, next_depth, alpha, beta))
            else:
                value = min(value, self._min_layer(succ, next_depth, next_agent, alpha, beta))

            if value <= alpha:
                return value
            beta = min(beta, value)

        return value

    # ---------- cutoff ----------
    def _cutoff(self, state: GameState, depth: int) -> bool:
        return depth >= self.depth_limit or state.isWin() or state.isLose()

    # ---------- evaluation function ----------
    def _evaluate(self, state: GameState) -> float:
        if state.isWin():
            return 1e4
        if state.isLose():
            return -1e4

        pacman = state.getPacmanPosition()
        ghosts = state.getGhostPositions()
        foods = state.getFood().asList()

        # heuristic: หนีผี + ใกล้อาหาร
        ghost_pen = -1e4 if any(pacman == g for g in ghosts) else 0
        food_score = -len(foods) * 10
        nearest_food = -min(self._manhattan(pacman, f) for f in foods) if foods else 0

        return state.getScore() + ghost_pen + food_score + nearest_food

    @staticmethod
    def _manhattan(a, b): 
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
