# bfs.py
from collections import deque
from pacman_module.game import Agent
from pacman_module.pacman import Directions


def key(state):
    """
    ฟังก์ชัน key: สร้างค่า hashable ที่แทนสถานะของเกม
    ใช้เพื่อเก็บใน set (closed) และป้องกันการเยี่ยม state เดิมซ้ำ
    """
    return (
        state.getPacmanPosition(),  # ตำแหน่งปัจจุบันของ Pacman
        state.getFood(),            # ตารางอาหารที่เหลืออยู่ (Grid)
        tuple(state.getCapsules())  # แปลง capsules ให้เป็น tuple เพื่อ hash ได้
    )


class PacmanAgent(Agent):
    """
    ตัวแทน (agent) ที่ใช้ Breadth-First Search (BFS)
    """

    def __init__(self, args):
        # เก็บเส้นทาง (list ของ moves) ที่หาเจอจาก BFS
        self.moves = []

    def get_action(self, state):
        """
        คืน action (ทิศทาง) ถัดไปให้ Pacman
        - ถ้ายังไม่มีเส้นทางใน self.moves ให้รัน BFS หาใหม่
        - ถ้ามีแล้วก็หยิบออกมาทีละก้าว
        """
        if not self.moves:
            self.moves = self.bfs(state) or []  # กันไม่ให้เป็น None
        return self.moves.pop(0) if self.moves else Directions.STOP

    def bfs(self, state):
        """
        อัลกอริทึม Breadth-First Search
        - ขยาย state ออกไปทีละ "ชั้น" (level)
        - หยุดเมื่อเจอสถานะชนะ (isWin)
        """
        fringe = deque()      # ใช้คิว (FIFO) สำหรับ BFS
        closed = set()        # เก็บ state ที่เคยเจอแล้ว
        fringe.append((state, []))  # เก็บ tuple (สถานะ, เส้นทางที่เดินมา)

        while fringe:
            current, path = fringe.popleft()  # เอา state ตัวหน้าออกมา

            if current.isWin():   # ถ้าเป็น state ชนะ
                return path       # คืนเส้นทางทันที

            k = key(current)
            if k in closed:       # ถ้า state นี้เคยเจอแล้ว ข้ามไป
                continue
            closed.add(k)

            # สร้าง successor (state ถัดไป + action ที่ทำให้ไปถึง)
            for succ, action in current.generatePacmanSuccessors():
                fringe.append((succ, path + [action]))

        return []  # ถ้าไม่มีทางออกเลย
