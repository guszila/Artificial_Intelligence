from pacman_module.game import Agent
from pacman_module.pacman import Directions
import heapq
import itertools

# ---------- Utilities ----------

def state_key(state):
    """
    แปลงสถานะ (state) ให้เป็น key ที่ hash ได้ (สำหรับเก็บใน dict/set)
    - รวมข้อมูลที่จำเป็นต่อการแยกแยะสถานะ: 
        * ตำแหน่งปัจจุบันของ Pacman
        * อาหารที่ยังเหลือ (เก็บเป็น tuple ของตำแหน่งแบบ sort เพื่อไม่ให้ order มีผล)
        * แคปซูลที่ยังเหลือ
    """
    pos = state.getPacmanPosition()
    food = tuple(sorted(state.getFood().asList()))
    capsules = tuple(sorted(state.getCapsules()))
    return (pos, food, capsules)

def manhattan(a, b):
    """
    ระยะทางแมนฮัตตันระหว่างสองตำแหน่ง (|x1-x2| + |y1-y2|)
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def max_manhattan_to_food(pos, foods):
    """
    heuristic: ระยะทางแมนฮัตตันไปยัง 'อาหารที่ไกลที่สุด'
    - admissible: เพราะระยะจริงใน maze ≥ ระยะแมนฮัตตันเสมอ
    - เลือก max (ไกลสุด) แทน min (ใกล้สุด) เพื่อได้ bound ที่ตึงขึ้น
    """
    if not foods:
        return 0
    return max(manhattan(pos, f) for f in foods)

# ---------- A* Agent ----------

class PacmanAgent(Agent):
    """
    ตัวแทน (agent) ที่ใช้ A* search
    - ใช้ priority queue (heapq) โดยเรียงตาม f = g + h
    - เก็บ best_g[state_key]: ค่า g (cost เดินจริงจาก start) ที่ดีที่สุดของแต่ละสถานะ
      เพื่อหลีกเลี่ยงการขยาย state ที่เคยเจอด้วยเส้นทางที่สั้นกว่าแล้ว
    """
    def __init__(self, args):
        self.moves = []  # เก็บเส้นทางที่หาเจอ

    def get_action(self, state):
        """
        เรียกเมื่อ Pacman ต้องเลือกการกระทำครั้งถัดไป
        - ถ้ายังไม่มี moves → รัน A* เพื่อหา path ทั้งหมดก่อน
        - คืนค่า action ตัวแรก แล้ว pop ออกจาก list
        """
        if not self.moves:
            self.moves = self.astar(state) or []
        return self.moves.pop(0) if self.moves else Directions.STOP

    def astar(self, start_state):
        """
        A* Search Algorithm:
        - ใช้ heapq เก็บ node: (f, g, tie_breaker, state, path)
            * f = g + h
            * g = cost ที่เดินมาถึง state นี้
            * h = heuristic (max manhattan ถึงอาหารที่เหลือ)
            * tie_breaker = counter เพื่อกัน heapq เทียบ state object โดยตรง
            * state = ตัว state ของเกม
            * path = ลิสต์ของ actions ที่ใช้เดินมาถึง state นี้
        """
        # เตรียมข้อมูลเริ่มต้น
        start_pos = start_state.getPacmanPosition()
        start_foods = start_state.getFood().asList()
        start_h = max_manhattan_to_food(start_pos, start_foods)

        heap = []
        counter = itertools.count()  # ตัวนับ unique ID สำหรับ tie-breaking
        heapq.heappush(heap, (start_h, 0, next(counter), start_state, []))

        # best_g เก็บ "ค่า g ที่ดีที่สุด" ของแต่ละ state
        best_g = {}

        while heap:
            f, g, _, current, path = heapq.heappop(heap)

            # ถ้า state นี้คือ win → คืน path ที่ได้ทันที
            if current.isWin():
                return path

            ck = state_key(current)

            # ถ้าเราเคยเจอ state นี้ด้วย g ที่ดีกว่าแล้ว → ข้าม
            if g > best_g.get(ck, float('inf')):
                continue

            # อัปเดต g ที่ดีที่สุดของสถานะนี้
            best_g[ck] = g

            # ขยาย successor states ทั้งหมด
            for next_state, action in current.generatePacmanSuccessors():
                new_path = path + [action]
                new_g = g + 1  # ทุก move มี cost = 1

                # คำนวณ heuristic ของ successor
                npos = next_state.getPacmanPosition()
                nfoods = next_state.getFood().asList()
                h = max_manhattan_to_food(npos, nfoods)

                nk = state_key(next_state)

                # ถ้าเส้นทางนี้สั้นกว่า → อัปเดตและ push เข้าคิว
                if new_g < best_g.get(nk, float('inf')):
                    best_g[nk] = new_g
                    heapq.heappush(
                        heap,
                        (new_g + h, new_g, next(counter), next_state, new_path)
                    )

        # ถ้าไม่เจอเส้นทาง (ไม่น่าจะเกิดกับ layout ที่ปกติ)
        return []
