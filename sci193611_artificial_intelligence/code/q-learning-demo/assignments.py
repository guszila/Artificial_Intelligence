#!/usr/bin/env python3
"""
Q-Learning Assignment - Grid World Challenge
แบบฝึกหัด Q-learning สำหรับนักเรียน

นักเรียนสามารถดัดแปลงไฟล์นี้เพื่อทำ assignment ต่างๆ
"""

from simple_q_learning import SimpleGridWorld, SimpleQLearning

def assignment_1_basic():
    """
    Assignment 1: Basic Q-Learning
    ให้รัน Q-learning ใน Grid World 4x4 และตอบคำถาม
    """
    print("=== Assignment 1: Basic Q-Learning ===")
    print()
    
    # TODO: สร้าง environment และ agent
    env = SimpleGridWorld(size=4)
    agent = SimpleQLearning(
        n_states=16,
        n_actions=4,
        learning_rate=0.1, #Test 0.01 กับ 0.5
        discount=0.9,
        epsilon=0.1
    )   
    
    print("Grid World Setup:")
    env.print_grid()
    
    # TODO: ฝึก agent
    print("Training...")
    agent.train(env, episodes=500)
    
    # TODO: ทดสอบและวิเคราะห์ผล
    print("\nTesting trained agent:")
    reward, steps, path = agent.test(env, show_path=False)
    print(f"Total reward: {reward}")
    print(f"Steps taken: {steps}")
    
    # แสดง Q-table บางส่วน
    print("\nQ-Table (first 8 states):")
    print("State |   ↑   |   ↓   |   ←   |   →   ")
    print("-" * 40)
    for state in range(8):
        q_vals = agent.q_table[state]
        print(f"{state:5d} | {q_vals[0]:5.2f} | {q_vals[1]:5.2f} | {q_vals[2]:5.2f} | {q_vals[3]:5.2f}")
    
    print("\n--- Questions for Assignment 1 ---")
    print("1. อธิบายทำไม Q-value ของ state ที่ใกล้ goal มีค่าสูงกว่า")
    print("     เพราะจากสมการ Bellman, ค่ารางวัลเป้าหมาย (+10) ถูกส่งย้อนกลับไปยัง state ก่อนหน้า")
    print("      states ที่เดินอีกไม่กี่ก้าวถึง goal จึงมีค่าคาดหวังสูงกว่า")
    print("2. ทำไม epsilon-greedy policy สำคัญในการเรียนรู้")
    print("     epsilon-greedy = บังคับให้สุ่ม action บางครั้ง (exploration) ")
    print("     ทำให้ agent มีโอกาสลองเส้นทางใหม่ ๆ ป้องกันการ “คิดว่าเจอเส้นทางดีที่สุด ทั้งที่ยังไม่ได้ลองครบ")
    print("3. ลองเปลี่ยน learning rate เป็น 0.01 และ 0.5 แล้วเปรียบเทียบผล")
    print("     learning rate 0.01 Agent อัปเดต Q-value ทีละน้อย → การเรียนรู้ช้ามาก Q-table มีค่าเล็ก ๆ และค่อย ๆ โตขึ้น → เสถียร แต่ช้า")
    print("     Learning rate 0.5  Agent เรียนรู้เร็วมากในช่วงแรก → reward พุ่งถึง ~9.5 ได้เร็ว แต่ Q-value กระโดดขึ้นลงเยอะ ไม่ค่อยเสถียร")
def assignment_2_parameter_study():
    """
    Assignment 2: Parameter Study
    ศึกษาผลของพารามิเตอร์ต่างๆ ต่อการเรียนรู้
    """
    print("=== Assignment 2: Parameter Study ===")
    print()
    
    # ทดลองกับ learning rates ต่างๆ
    learning_rates = [0.01, 0.1, 0.3, 0.7]
    #epsilons = [0.01, 0.1, 0.3, 0.7
    #gammas = [0.5, 0.7, 0.9, 0.99]
    print("Testing different learning rates:")
    
    for lr in learning_rates:
        env = SimpleGridWorld(size=4)
        agent = SimpleQLearning(
            n_states=16,
            n_actions=4,
            learning_rate=lr,
            discount=0.9,
            epsilon=0.1
        )
        
        agent.train(env, episodes=500)
        reward, steps, _ = agent.test(env, show_path=False)
        
        print(f"Learning Rate {lr}: Final reward = {reward:.2f}, Steps = {steps}")
    
    print("\n--- Questions for Assignment 2 ---")
    print("1. Learning rate ไหนให้ผลดีที่สุด? ทำไม?")
    print("     - 0.01 → ช้ามากกว่าจะดีขึ้น")
    print("     - 0.7 → ขึ้นเร็วแต่ Q-values กระโดดเยอะ ไม่ค่อยเสถียร")
    print("     - 0.1 หรือ 0.3 ดีที่สุด → สมดุล ทั้งเร็วและเสถียร")
    print("2. ลองทดลองกับ epsilon values: 0.01, 0.1, 0.3, 0.7")
    print("     - 0.01: สำรวจน้อยเกินไป อาจพลาดเส้นทางที่ดีกว่า")
    print("     - 0.1: สมดุลดี (exploit 90%, explore 10%) ✅")
    print("     - 0.3: explore มากขึ้น เหมาะกับปัญหายากขึ้น")
    print("     - 0.7: explore เยอะเกินไป agent เดินมั่ว ไม่ค่อย exploit")
    print("3. ลองทดลองกับ discount factor: 0.5, 0.7, 0.9, 0.99")
    print("     - 0.5: สนใจแต่รางวัลใกล้ ๆ ไม่มุ่งไป goal")
    print("     - 0.7: สนใจอนาคตบ้าง แต่ยังไม่มาก")
    print("     - 0.9: สมดุล สนใจทั้งปัจจุบันและอนาคต ✅")
    print("     - 0.99: สนใจอนาคตมากเกิน เรียนรู้ช้าลง")

def assignment_3_environment_design():
    """
    Assignment 3: Environment Design
    ออกแบบ environment ใหม่และทดสอบ
    """
    print("=== Assignment 3: Environment Design ===")
    print()
    
    # TODO: ให้นักเรียนสร้าง environment ใหม่
    # ตัวอย่าง: Grid World ขนาดใหญ่กว่า หรือมีอุปสรรคมากกว่า
    # class CustomGridWorld(SimpleGridWorld):
    #     def __init__(self):
    #         super().__init__(size=6)   # ทำ grid 6x6
    #     # ออกแบบอุปสรรคเอง
    #         self.obstacles = [(1,1), (1,2), (2,3), (3,3), (4,2), (4,4)]
    #     # ปรับ reward/penalty
    #         self.goal_reward = 15
    #         self.obstacle_penalty = -8

    class CustomGridWorld(SimpleGridWorld):
        def __init__(self):
            super().__init__(size=5)
            # เพิ่มอุปสรรคใหม่
            self.obstacles = [(1, 1), (1, 2), (2, 1), (3, 3)]
            # เปลี่ยน reward structure
            self.goal_reward = 20
            self.obstacle_penalty = -10
    
    env = CustomGridWorld()
    print("Custom Grid World:")
    env.print_grid()
    
    agent = SimpleQLearning(
        n_states=25,
        # n_states=36,
        n_actions=4,
        learning_rate=0.1,
        discount=0.9,
        epsilon=0.2
    )
    
    agent.train(env, episodes=1000)
    reward, steps, _ = agent.test(env, show_path=False)
    print(f"Custom environment result: Reward = {reward:.2f}, Steps = {steps}")
    
    print("\n--- Tasks for Assignment 3 ---")
    print("1. ออกแบบ Grid World ของคุณเอง (ขนาด, อุปสรรค, rewards)")
    print("2. เปรียบเทียบผลการเรียนรู้กับ standard environment")
    print("     - 4×4: agent ใช้ 6 steps ถึง goal, reward ≈ 9.5")
    print("     - 6×6: agent ใช้ 10 steps ถึง goal, reward ≈ 9.1")
    print("     - แปลว่าเมื่อ กริดใหญ่ขึ้น → agent ต้องใช้ก้าวมากขึ้น, reward เลยลดลงเล็กน้อย")
    print("3. วิเคราะห์ว่า environment design ส่งผลต่อ learning อย่างไร")
    print("     -การออกแบบ environment ที่ใหญ่และซับซ้อนขึ้น ทำให้การเรียนรู้ยากขึ้น → ต้องใช้ episodes มากขึ้น, exploration สำคัญขึ้น")

def assignment_4_advanced():
    """
    Assignment 4: Advanced Modifications
    การปรับปรุง algorithm หรือเพิ่มฟีเจอร์ใหม่
    """
    print("=== Assignment 4: Advanced Modifications ===")
    print()
    
    # TODO: ให้นักเรียนเลือกหัวข้อที่สนใจ
    
    print("Choose one of the following topics:")
    print("1. Implement SARSA algorithm และเปรียบเทียบกับ Q-Learning")
    print("2. Add epsilon decay strategy ที่ซับซ้อนกว่า")
    print("3. Implement Double Q-Learning")
    print("4. Add experience replay")
    print("5. Create multi-goal environment")
    print("6. Implement priority sweeping")
    
    print("\nExample: Simple SARSA Implementation")
    
    class SARSAAgent(SimpleQLearning):
        """SARSA Agent - On-policy learning"""
        
        def train_episode(self, env, max_steps=100):
            state = env.reset()
            action = self.get_action(state)  # เลือก action แรก
            total_reward = 0
            
            for _ in range(max_steps):
                next_state, reward, done = env.step(action)
                next_action = self.get_action(next_state) if not done else 0
                
                # SARSA update: ใช้ actual next action แทน max
                target = reward + self.gamma * self.q_table[next_state][next_action]
                error = target - self.q_table[state][action]
                self.q_table[state][action] += self.lr * error
                
                total_reward += reward
                state, action = next_state, next_action
                
                if done:
                    break
            
            return total_reward, 0
    
    print("SARSA vs Q-Learning comparison example implemented above.")

    print("\n=== ทดลองเปรียบเทียบ SARSA vs Q-Learning ===")

    env = SimpleGridWorld(size=5, difficulty='normal')

    # Q-Learning agent
    q_agent = SimpleQLearning(
        n_states=env.size * env.size,
        n_actions=4,
        learning_rate=0.1,
        discount=0.9,
        epsilon=0.2
    )
    q_agent.train(env, episodes=800)
    q_reward, q_steps, _ = q_agent.test(env, show_path=False)
    print(f"Q-Learning -> Reward={q_reward:.2f}, Steps={q_steps}")

    # SARSA agent
    sarsa_agent = SARSAAgent(
        n_states=env.size * env.size,
        n_actions=4,
        learning_rate=0.1,
        discount=0.9,
        epsilon=0.2
    )
    sarsa_agent.train(env, episodes=800)
    s_reward, s_steps, _ = sarsa_agent.test(env, show_path=False)
    print(f"SARSA      -> Reward={s_reward:.2f}, Steps={s_steps}")

    # สรุปผล
    if s_reward > q_reward:
        print("\nSARSA ทำได้ดีกว่าใน environment นี้ ✅")
    elif q_reward > s_reward:
        print("\nQ-Learning ทำได้ดีกว่าใน environment นี้ ✅")
    else:
        print("\nผลออกมาสูสีกัน")


def bonus_visualization():
    """
    Bonus: Enhanced Visualization
    การแสดงผลที่สวยงามขึ้น (สำหรับนักเรียนที่สนใจ)
    """
    print("=== Bonus: Enhanced Visualization ===")
    print()
    
    # TODO: ใช้ matplotlib สร้างกราฟ learning curve
    # TODO: สร้าง animation ของการเรียนรู้
    # TODO: แสดง heatmap ของ Q-values

    import matplotlib.pyplot as plt
    import numpy as np


    """
    Bonus: Enhanced Visualization
    """
    print("=== Bonus: Enhanced Visualization ===\n")

    # สร้าง environment และ agent
    from simple_q_learning import SimpleGridWorld, SimpleQLearning
    env = SimpleGridWorld(size=4)
    agent = SimpleQLearning(
        n_states=env.size * env.size,
        n_actions=4,
        learning_rate=0.1,
        discount=0.9,
        epsilon=0.3
    )

    # ฝึก agent แล้วเก็บ reward
    episodes = 500
    agent.train(env, episodes=episodes)

    # 📈 Plot Learning Curve
    plt.figure(figsize=(8,5))
    plt.plot(agent.episode_rewards, label="Episode reward")
    plt.title("Q-Learning Performance (Learning Curve)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 🔥 Heatmap ของ Q-values (ค่า max Q ของแต่ละ state)
    q_values = [max(agent.q_table[s]) for s in range(agent.n_states)]
    q_grid = np.array(q_values).reshape(env.size, env.size)

    plt.figure(figsize=(6,6))
    plt.imshow(q_grid, cmap="YlGnBu", origin="upper")
    plt.colorbar(label="Max Q-value")
    plt.title("Heatmap of Max Q-values per State")
    for r in range(env.size):
        for c in range(env.size):
            plt.text(c, r, f"{q_grid[r,c]:.1f}",
                     ha="center", va="center", color="black")
    plt.show()

    print("\n✅ แสดงผล Learning Curve และ Heatmap เสร็จสิ้น!")

        
    print("Ideas for enhanced visualization:")
    print("1. Plot learning curves with matplotlib")
    print("2. Create heatmap of Q-values")
    print("3. Animate the learning process")
    print("4. Show value function as 3D surface")
    print("5. Create policy visualization with arrows")
    
    print("\nSample code for learning curve:")
    print("""
import matplotlib.pyplot as plt

def plot_learning_curve(episode_rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('Q-Learning Performance')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()
    """)

def main():
    """เลือก assignment ที่จะทำ"""
    print("🎓 Q-Learning Assignments")
    print("=========================")
    print()
    
    assignments = {
        '1': assignment_1_basic,
        '2': assignment_2_parameter_study,
        '3': assignment_3_environment_design,
        '4': assignment_4_advanced,
        '5': bonus_visualization
    }
    
    while True:
        print("Available assignments:")
        print("1. Basic Q-Learning (เริ่มต้น)")
        print("2. Parameter Study (ศึกษาพารามิเตอร์)")
        print("3. Environment Design (ออกแบบ environment)")
        print("4. Advanced Modifications (ขั้นสูง)")
        print("5. Bonus: Visualization (เสริม)")
        print("6. Exit")
        print()
        
        choice = input("เลือก assignment (1-6): ").strip()
        
        if choice in assignments:
            print()
            assignments[choice]()
            print("\n" + "="*50 + "\n")
        elif choice == '6':
            print("Good luck with your assignments! 🚀")
            break
        else:
            print("กรุณาเลือก 1-6")

if __name__ == "__main__":
    main()
