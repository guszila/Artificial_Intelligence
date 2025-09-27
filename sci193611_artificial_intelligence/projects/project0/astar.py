from pacman_module.game import Agent
from pacman_module.pacman import Directions
import heapq
import itertools

# สร้าง key สำหรับ state เพื่อเช็คซ้ำ
def key(state):
    return (state.getPacmanPosition(), state.getFood(), tuple(state.getCapsules()))

# heuristic แบบ Manhattan distance
def manhattan_heuristic(pos, goals):
    if not goals:
        return 0
    return min(abs(pos[0]-g[0]) + abs(pos[1]-g[1]) for g in goals)

class PacmanAgent(Agent):
    """
    Pacman agent ใช้อัลกอริทึม A* search
    """
    def __init__(self, args):
        self.moves = []

    def get_action(self, state):
        if not self.moves:
            self.moves = self.astar(state)
            if self.moves is None:
                self.moves = []
        return self.moves.pop(0) if self.moves else Directions.STOP

    def astar(self, state):
        """
        คืนค่า list ของ moves เพื่อแก้ maze ด้วย A*
        """
        start_pos = state.getPacmanPosition()
        food_positions = state.getFood().asList()
        
        heap = []
        closed = set()
        counter = itertools.count()  # เพิ่ม counter เพื่อหลีกเลี่ยง error heap

        # push node แรกลง heap
        heapq.heappush(heap, (manhattan_heuristic(start_pos, food_positions), 0, next(counter), state, []))

        while heap:
            f, g, _, current, path = heapq.heappop(heap)

            if current.isWin():
                return path

            current_key = key(current)
            if current_key in closed:
                continue
            closed.add(current_key)

            for next_state, action in current.generatePacmanSuccessors():
                next_path = path + [action]
                next_pos = next_state.getPacmanPosition()
                next_food = next_state.getFood().asList()
                h = manhattan_heuristic(next_pos, next_food)
                heapq.heappush(heap, (g + 1 + h, g + 1, next(counter), next_state, next_path))

        return []  # failure
