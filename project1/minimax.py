# minimax.py  (เวอร์ชันกันวน + นับโหนดตามสเปค)

from collections import deque
from pacman_module.game import Agent, Directions
from pacman_module.pacman import GameState

INF = float("inf")

class PacmanAgent(Agent):
    """
    Minimax (Depth-limited) + anti-loop tweaks + correct node counting
    - กันเดินวนด้วยประวัติตำแหน่งล่าสุด
    - เลี่ยง STOP และดัน 'ย้อนศร' ไปท้ายลิสต์
    - heuristic เบาๆ ที่ดึงเข้าหาอาหาร + หนีผีแบบพอดี
    """

    def __init__(self, depth: str = "2"):
        self.depth_limit = int(depth)
        self.pos_hist = deque(maxlen=8)  # ประวัติตำแหน่ง 8 ก้าวล่าสุด
        self.last_action = None
        self.dir_order = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST, Directions.STOP]

    # ---------- main ----------
    def get_action(self, state: GameState):
        GameState.resetNodeExpansionCounter()
        self.pos_hist.append(state.getPacmanPosition())

        best_score, best_action = -INF, None

        legal = [a for a in state.getLegalActions(0) if a != Directions.STOP]
        if self.last_action:
            rev = Directions.REVERSE[self.last_action]
            # ดันการย้อนศรไปท้าย (ถ้ามีทางอื่น)
            legal = sorted(legal, key=lambda a: a == rev)

        for action in legal:
            succ = state.generateSuccessor(0, action)
            GameState.countExpanded += 1
            score = self._min_layer(succ, depth=0, agent_index=1)
            if score > best_score or (score == best_score and self._prefer(action, best_action)):
                best_score, best_action = score, action

        self.last_action = best_action
        return best_action or Directions.STOP

    # ---------- MAX (Pacman) ----------
    def _max_layer(self, state: GameState, depth: int):
        if self._cutoff(state, depth):
            return self._evaluate(state)

        value = -INF
        legal = [a for a in state.getLegalActions(0) if a != Directions.STOP]
        if self.last_action:
            rev = Directions.REVERSE[self.last_action]
            legal = sorted(legal, key=lambda a: a == rev)

        for action in legal:
            succ = state.generateSuccessor(0, action)
            GameState.countExpanded += 1
            v = self._min_layer(succ, depth, 1)
            value = max(value, v)

        return value if value != -INF else self._evaluate(state)

    # ---------- MIN (Ghost i) ----------
    def _min_layer(self, state: GameState, depth: int, agent_index: int):
        if self._cutoff(state, depth):
            return self._evaluate(state)

        value = INF
        num_agents = state.getNumAgents()
        actions = [a for a in state.getLegalActions(agent_index) if a != Directions.STOP]
        if not actions:
            return self._evaluate(state)

        for action in actions:
            succ = state.generateSuccessor(agent_index, action)
            GameState.countExpanded += 1

            next_agent = (agent_index + 1) % num_agents
            next_depth = depth + 1 if next_agent == 0 else depth
            if next_agent == 0:
                v = self._max_layer(succ, next_depth)
            else:
                v = self._min_layer(succ, next_depth, next_agent)

            value = min(value, v)

        return value

    # ---------- cutoff + evaluation ----------
    def _cutoff(self, state: GameState, depth: int) -> bool:
        return depth >= self.depth_limit or state.isWin() or state.isLose()

    def _evaluate(self, state: GameState) -> float:
        if state.isWin():  return 1e4
        if state.isLose(): return -1e4

        pac = state.getPacmanPosition()
        ghosts = state.getGhostPositions()
        foods = state.getFood().asList()

        # ชนผี = แพ้แน่
        if ghosts and any(pac == g for g in ghosts):
            return -1e4

        # หนีผีแบบพอดี (ลดน้ำหนักลงเล็กน้อยเพื่อลดโอกาสวิ่งวน)
        ghost_term = 0.0
        if ghosts:
            dmin_g = min(self._manhattan(pac, g) for g in ghosts)
            ghost_term = 2.5 * (dmin_g ** 0.5)

        # ดึงเข้าหาอาหาร & ลดจำนวนอาหารคงเหลือ
        near_food = -min(self._manhattan(pac, f) for f in foods) if foods else 0.0
        food_pen  = -9.0 * len(foods)

        # ลงโทษการวน (ถ้าเหยียบตำแหน่งที่เพิ่งผ่านมา)
        repeat_pen = -6.0 if pac in self.pos_hist else 0.0

        # เร่งจบเกมเล็กน้อยเมื่อใกล้หมด
        endgame = 5.0 if len(foods) <= 6 else 0.0

        return state.getScore() + ghost_term + near_food + food_pen + repeat_pen + endgame

    # ---------- utils ----------
    def _prefer(self, a, b):
        return b is None or (a is not None and self._order_index(a) < self._order_index(b))

    def _order_index(self, a):
        try:
            return self.dir_order.index(a)
        except ValueError:
            return len(self.dir_order)

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
