# B6644673 ภาณุเดช ศรีวุฒิทรัพย์ 
# B6638375 กิตติธัช จังพนาสิน

from collections import deque
from pacman_module.game import Agent, Directions
from pacman_module.pacman import GameState

INF = float("inf")  # ค่ามากที่สุด ใช้แทน +∞

class PacmanAgent(Agent):
    """
    H-Minimax (Minimax + Heuristic + Depth Limit + Alpha-Beta)
    - รองรับหลายผี
    - กันเดินวนด้วยประวัติตำแหน่ง
    - เลี่ยง STOP/ย้อนศร และทำ move ordering เพื่อ pruning ดีขึ้น
    """

    def __init__(self, args=None):
        self.depth_limit = getattr(args, "depth", 2) if args else 2
        self.order = [Directions.NORTH, Directions.EAST,
                      Directions.SOUTH, Directions.WEST, Directions.STOP]
        self.pos_hist = deque(maxlen=8)   # กันเดินวน: จำ 8 ก้าวล่าสุด
        self.last_action = None           # ใช้เลี่ยงย้อนศร

    def get_action(self, state):
        GameState.resetNodeExpansionCounter()
        self.pos_hist.append(state.getPacmanPosition())  # เก็บตำแหน่งปัจจุบัน
        _, action = self._max(state, 0, -INF, INF)       # เริ่มที่ Pacman
        self.last_action = action
        return action or Directions.STOP

    # ---------- MAX (Pacman) ----------
    def _max(self, state, depth, alpha, beta):
        if self._cutoff(state, depth):
            return self._eval(state), None

        # ตัด STOP และดันย้อนศรไปท้ายลิสต์
        actions = [a for a in state.getLegalPacmanActions() if a != Directions.STOP]
        if self.last_action:
            rev = Directions.REVERSE[self.last_action]
            actions = sorted(actions, key=lambda a: a == rev)

        # move ordering: เรียง successor จากดี→แย่ ด้วย eval_แบบไว
        succs = []
        for act in actions:
            succ = state.generateSuccessor(0, act)
            GameState.countExpanded += 1
            succs.append((succ, act))
        succs.sort(key=lambda sa: self._eval_quick(sa[0]), reverse=True)

        best_value, best_action = -INF, None
        for succ, act in succs:
            value, _ = self._min(succ, depth, 1, alpha, beta)
            if value > best_value or (value == best_value and self._prefer(act, best_action)):
                best_value, best_action = value, act
            if best_value >= beta:    # beta-cut
                break
            alpha = max(alpha, best_value)
        return best_value, best_action

    # ---------- MIN (Ghosts) ----------
    def _min(self, state, depth, ghost_index, alpha, beta):
        if self._cutoff(state, depth):
            return self._eval(state), None

        # ผีไม่ต้องสนใจย้อนศร แต่ตัด STOP ทิ้ง
        actions = [a for a in state.getLegalActions(ghost_index) if a != Directions.STOP]
        if not actions:
            return self._eval(state), None

        # move ordering ฝั่ง MIN: เรียงจากแย่สำหรับ Pacman → ดี
        succs = []
        for act in actions:
            succ = state.generateSuccessor(ghost_index, act)
            GameState.countExpanded += 1
            succs.append((succ, act))
        succs.sort(key=lambda sa: self._eval_quick(sa[0]))  # ค่าน้อยก่อน

        best_value, best_action = INF, None
        last_ghost = (ghost_index == state.getNumAgents() - 1)
        for succ, act in succs:
            if last_ghost:
                value, _ = self._max(succ, depth + 1, alpha, beta)
            else:
                value, _ = self._min(succ, depth, ghost_index + 1, alpha, beta)

            if value < best_value or (value == best_value and self._prefer(act, best_action)):
                best_value, best_action = value, act
            if best_value <= alpha:   # alpha-cut
                break
            beta = min(beta, best_value)
        return best_value, best_action

    # ---------- cutoff ----------
    def _cutoff(self, state, depth):
        return depth >= self.depth_limit or state.isWin() or state.isLose()

    # ---------- heuristic (ละเอียด ใช้ตัดสินจริง) ----------
    def _eval(self, state):
        if state.isWin():  return 1e4
        if state.isLose(): return -1e4

        pacman = state.getPacmanPosition()
        ghosts = state.getGhostPositions()
        foods = state.getFood().asList()

        # ชนผี = แพ้
        if ghosts and any(pacman == g for g in ghosts):
            return -1e4

        # หนีผีแบบไม่ over-defensive
        ghost_bonus = 0.0
        if ghosts:
            dmin_g = min(self._manhattan(pacman, g) for g in ghosts)
            ghost_bonus = 2.5 * (dmin_g ** 0.5)

        # ดึงเข้าหาอาหาร + ลดจำนวนอาหารที่เหลือ
        near_food = -min(self._manhattan(pacman, f) for f in foods) if foods else 0.0
        food_pen  = -9.0 * len(foods)

        # กันวน: ถ้าเหยียบตำแหน่งซ้ำในช่วงสั้น ๆ ให้โทษ
        repeat_pen = -6.0 if pacman in self.pos_hist else 0.0

        # เร่งปิดเกมช่วงท้าย
        endgame = 5.0 if len(foods) <= 6 else 0.0

        return state.getScore() + ghost_bonus + near_food + food_pen + repeat_pen + endgame

    # ---------- heuristic แบบไว (ใช้สำหรับ move ordering เท่านั้น) ----------
    def _eval_quick(self, state):
        if state.isWin():  return 1e4
        if state.isLose(): return -1e4
        pac = state.getPacmanPosition()
        ghosts = state.getGhostPositions()
        foods = state.getFood().asList()
        ghost_term = 0.0
        if ghosts:
            dmin_g = min(self._manhattan(pac, g) for g in ghosts)
            ghost_term = (dmin_g ** 0.5)
        near_food = -min(self._manhattan(pac, f) for f in foods) if foods else 0.0
        return ghost_term + near_food - 2.0 * len(foods)

    # ---------- utilities ----------
    def _prefer(self, a, b):
        return b is None or (a is not None and self.order.index(a) < self.order.index(b))

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
