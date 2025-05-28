import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import matplotlib.colors as mcolors
from scipy.optimize import linear_sum_assignment

# === Классы ===

class Task:
    def __init__(self, task_id, position):
        self.id = task_id
        self.position = np.array(position, dtype=float)
        self.status = 'waiting'
        self.assigned_robot = None
        self.color = 'gray'

class Robot:
    def __init__(self, robot_id, position):
        self.id = robot_id
        self.position = np.array(position, dtype=float)
        self.target_task = None
        self.speed = 0.1
        self.color = 'blue'
        self.task_color = None

    def distance_to(self, task):
        return np.linalg.norm(self.position - task.position)

    def assign_task(self, task, color):
        self.target_task = task
        task.assigned_robot = self
        task.status = 'in_progress'
        self.task_color = color
        self.color = color
        task.color = color

    def move(self):
        if self.target_task:
            direction = self.target_task.position - self.position
            dist = np.linalg.norm(direction)
            if dist < 0.2:
                self.position = self.target_task.position.copy()
                self.target_task.status = 'done'
                self.target_task.color = 'lightgray'
                self.target_task.assigned_robot = None
                self.target_task = None
                self.color = 'blue'
            else:
                self.position += self.speed * direction / dist

class CentralManager:
    def __init__(self, robots, tasks, formation_center, grid_size=2.0):
        self.robots = robots
        self.tasks = tasks
        self.grid_size = grid_size
        self.phase = 'formation'
        self.color_pool = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
        self.used_colors = []
        self.formation_center = formation_center

    def form_grid(self):
        n = len(self.robots)
        rows = int(np.ceil(np.sqrt(n)))
        formation = []
        center = self.formation_center
        idx = 0
        for i in range(rows):
            for j in range(rows):
                if idx < n:
                    dx = (j - rows // 2) * self.grid_size
                    dy = (i - rows // 2) * self.grid_size
                    formation.append(center + np.array([dx, dy]))
                    idx += 1
        return formation

    def update_formation(self):
        formation = self.form_grid()
        done = True
        for i, robot in enumerate(self.robots):
            direction = formation[i] - robot.position
            dist = np.linalg.norm(direction)
            if dist > 0.2:
                done = False
                robot.position += robot.speed * direction / dist
        if done:
            self.phase = 'assignment'

    def assign_tasks_globally(self):
        waiting_tasks = [t for t in self.tasks if t.status == 'waiting']
        if not waiting_tasks:
            return

        cost_matrix = np.full((len(self.robots), len(waiting_tasks)), np.inf)

        for i, robot in enumerate(self.robots):
            for j, task in enumerate(waiting_tasks):
                if robot.target_task is None:
                    # Робот свободен: обычное расстояние
                    eta = robot.distance_to(task) / robot.speed
                elif robot.target_task and robot.target_task.status != 'done':
                    # Робот занят: учесть сначала завершение текущей задачи
                    to_current = robot.distance_to(robot.target_task)
                    from_current_to_new = np.linalg.norm(robot.target_task.position - task.position)
                    eta = (to_current + from_current_to_new) / robot.speed
                else:
                    eta = robot.distance_to(task) / robot.speed

                cost_matrix[i, j] = eta

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for i, j in zip(row_ind, col_ind):
            robot = self.robots[i]
            task = waiting_tasks[j]
            if task.status == 'waiting' and robot.target_task is None:
                color = robot.task_color if robot.task_color else self.get_unique_color()
                robot.assign_task(task, color)

    def get_unique_color(self):
        for c in self.color_pool:
            if c not in self.used_colors:
                self.used_colors.append(c)
                return c
        return 'black'

    def step(self):
        if self.phase == 'formation':
            self.update_formation()
        elif self.phase == 'assignment':
            self.assign_tasks_globally()
            self.phase = 'execution'
        elif self.phase == 'execution':
            waiting_tasks = [t for t in self.tasks if t.status == 'waiting']
            free_robots = [r for r in self.robots if r.target_task is None]
            if waiting_tasks and free_robots:
                self.assign_tasks_globally()
            for robot in self.robots:
                robot.move()

# === Настройки ===
cluster_distance = 20
num_robots = 5
num_tasks = 8

robots = [
    Robot('R1', [14, 14]),
    Robot('R2', [16, 14]),
    Robot('R3', [14, 16]),
    Robot('R4', [16, 16]),
    Robot('R5', [15, 15])
]

tasks = []
offsets = [[0, 1], [1, 0], [0, -1], [-1, 0]]
for i, (dx, dy) in enumerate(offsets):
    x1 = 15 + cluster_distance * dx
    y1 = 15 + cluster_distance * dy
    x2 = 15 + (cluster_distance - 5) * dx
    y2 = 15 + (cluster_distance - 5) * dy
    tasks.append(Task(f'T{i*2+1}', [x1, y1]))
    tasks.append(Task(f'T{i*2+2}', [x2, y2]))

# robots = [
#     Robot('R1', [13, 16]),
#     Robot('R2', [15, 16]),
#     Robot('R3', [17, 16]),
#     Robot('R4', [14, 14]),
#     Robot('R5', [16, 14]),
# ]
#
# tasks = [
#     Task('T1', [15 + 10 * np.cos(0),            15 + 10 * np.sin(0)]),           # 0°
#     Task('T2', [15 + 10 * np.cos(np.pi / 4),    15 + 10 * np.sin(np.pi / 4)]),   # 45°
#     Task('T3', [15 + 10 * np.cos(np.pi / 2),    15 + 10 * np.sin(np.pi / 2)]),   # 90°
#     Task('T4', [15 + 10 * np.cos(3 * np.pi / 4),15 + 10 * np.sin(3 * np.pi / 4)]),# 135°
#     Task('T5', [15 + 10 * np.cos(np.pi),        15 + 10 * np.sin(np.pi)]),       # 180°
#     Task('T6', [15 + 10 * np.cos(5 * np.pi / 4),15 + 10 * np.sin(5 * np.pi / 4)]),# 225°
#     Task('T7', [15 + 10 * np.cos(3 * np.pi / 2),15 + 10 * np.sin(3 * np.pi / 2)]),# 270°
#     Task('T8', [15 + 10 * np.cos(7 * np.pi / 4),15 + 10 * np.sin(7 * np.pi / 4)]),# 315°
# ]

formation_center = np.array([15, 15])
manager = CentralManager(robots, tasks, formation_center)

# === Визуализация ===

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
plt.title("Оптимальное распределение задач")

robot_dots = ax.scatter([], [], s=80, marker='o', edgecolors='black', linewidths=1.0, zorder=3)
task_dots = ax.scatter([], [], s=200, marker='o', edgecolors='black', linewidths=1.0, zorder=2)
task_texts = [ax.text(*t.position + 0.3, t.id, fontsize=9) for t in tasks]
speed_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=10)
center_cross = ax.scatter([formation_center[0]], [formation_center[1]], marker='x', c='red', s=150, zorder=1)

dt_scale = [1.0]
running = [True]

ax_up = plt.axes([0.7, 0.05, 0.08, 0.05])
ax_down = plt.axes([0.8, 0.05, 0.08, 0.05])
ax_pause = plt.axes([0.6, 0.05, 0.08, 0.05])

btn_up = Button(ax_up, '>>')
btn_down = Button(ax_down, '<<')
btn_pause = Button(ax_pause, 'Pause')

def increase(event):
    dt_scale[0] = min(5.0, dt_scale[0] + 0.5)

def decrease(event):
    dt_scale[0] = max(0.1, dt_scale[0] - 0.5)

def toggle(event):
    running[0] = not running[0]

btn_up.on_clicked(increase)
btn_down.on_clicked(decrease)
btn_pause.on_clicked(toggle)

def animate(frame):
    if running[0]:
        for robot in manager.robots:
            robot.speed = 0.1 * dt_scale[0]
        manager.step()

    robot_pos = np.array([r.position for r in manager.robots])
    task_pos = np.array([t.position for t in manager.tasks])
    robot_colors = [r.color for r in manager.robots]
    task_colors = [t.color for t in manager.tasks]

    robot_dots.set_offsets(robot_pos)
    robot_dots.set_facecolor(robot_colors)
    task_dots.set_offsets(task_pos)
    task_dots.set_facecolor(task_colors)
    speed_text.set_text(f"Скорость: x{dt_scale[0]:.1f}")

    return robot_dots, task_dots, center_cross, speed_text, *task_texts

ani = animation.FuncAnimation(fig, animate, frames=1000, interval=100, blit=True)
plt.show()
