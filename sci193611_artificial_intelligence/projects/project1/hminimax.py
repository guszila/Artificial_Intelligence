# B6644673 ภาณุเดช ศรีวุฒิทรัพย์ 
# B6638375 กิตติธัช จังพนาสิน


from pacman_module.game import Agent, Directions

INF = float("inf")

class PacmanAgent(Agent):
    """
    H-Minimax Agent (ใช้ heuristic + depth limit + alpha-beta pruning)
    """

    def __init__(self, args=None):
        # ความลึกสูงสุดของการค้นหา (default = 4)
        self.depth_limit = getattr(args, "depth", 4) if args else 4
        self.nodes_expanded = 0

        # ลำดับการเลือกทิศ (เอาไว้ใช้ตอน tie-break)
        self.order = [
            Directions.NORTH, Directions.EAST,
            Directions.SOUTH, Directions.WEST,
            Directions.STOP
        ]

    def get_action(self, state):
        """
        ฟังก์ชันหลักของ agent เริ่มจากตาของ Pacman
        """
        _, action = self._max(state, 0, -INF, INF)
        return action or Directions.STOP

    # ---------- ตาของ Pacman (Maximizer) ----------
    def _max(self, state, depth, alpha, beta):
        # ถ้าถึงจุด cutoff (ลึกเกินหรือจบเกม) → ประเมินค่า
        if self._cutoff(state, depth):
            return self._eval(state), None

        best_value, best_action = -INF, None
        actions = sorted(state.getLegalPacmanActions(), key=self.order.index)

        for act in actions:
            succ = state.generateSuccessor(0, act)
            self.nodes_expanded += 1

            value, _ = self._min(succ, depth + 1, alpha, beta)

            # เก็บค่าที่ดีที่สุด
            if value > best_value or (value == best_value and self._prefer(act, best_action)):
                best_value, best_action = value, act

            # pruning
            if best_value >= beta:
                break
            alpha = max(alpha, best_value)

        return best_value, best_action

    # ---------- ตาของ Ghost (Minimizer) ----------
    def _min(self, state, depth, alpha, beta):
        if self._cutoff(state, depth):
            return self._eval(state), None

        best_value, best_action = INF, None
        actions = sorted(state.getLegalActions(1), key=self.order.index)

        for act in actions:
            succ = state.generateSuccessor(1, act)
            self.nodes_expanded += 1

            value, _ = self._max(succ, depth + 1, alpha, beta)

            if value < best_value or (value == best_value and self._prefer(act, best_action)):
                best_value, best_action = value, act

            if best_value <= alpha:
                break
            beta = min(beta, best_value)

        return best_value, best_action

    # ---------- cutoff ----------
    def _cutoff(self, state, depth):
        return depth >= self.depth_limit or state.isWin() or state.isLose()

    # ---------- heuristic eval ----------
    def _eval(self, state):
        if state.isWin():
            return 1e4
        if state.isLose():
            return -1e4

        # ตำแหน่ง
        pacman = state.getPacmanPosition()
        ghosts = state.getGhostPositions()
        foods = state.getFood().asList()

        # bonus หนีผี
        ghost_bonus = 0
        if ghosts:
            if any(pacman == g for g in ghosts):
                return -1e4
            dist_ghost = min(self._manhattan(pacman, g) for g in ghosts)
            ghost_bonus = 3.0 * (dist_ghost ** 0.5)

        # penalty อาหาร
        food_pen = -8 * len(foods)
        near_food = -min(self._manhattan(pacman, f) for f in foods) if foods else 0

        return state.getScore() + ghost_bonus + food_pen + near_food

    # ---------- utilities ----------
    def _prefer(self, a, b):
        """ tie-break: ใช้ลำดับทิศ """
        return b is None or (a is not None and self.order.index(a) < self.order.index(b))

    @staticmethod
    def _manhattan(a, b):
        """ Manhattan distance """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
