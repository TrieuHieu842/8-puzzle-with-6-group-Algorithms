import pygame as pg
from pygame.locals import *
from collections import deque, defaultdict
import heapq
import random
import math

import numpy as np
import copy
from collections import namedtuple
import time
import sys
import threading
import matplotlib.pyplot as plt

# Khởi tạo Pygame
pg.init()

# Cài đặt màu sắc
WHITE = (245, 245, 245)
BLACK = (30, 30, 30)
GRAY = (120, 130, 140)
LIGHT_GRAY = (180, 190, 200)
BLUE = (50, 100, 255)
DARK_BLUE = (30, 70, 200)
RED = (255, 100, 100)
SHADOW = (100, 100, 100, 50)
GREEN = (0, 255, 0)

# Kích thước cửa sổ
GAME_WIDTH = 550
GAME_HEIGHT = 650
ALGO_WIDTH = 250
ALGO_HEIGHT = 650
GAP = 20

# Tạo cửa sổ
game_screen = pg.display.set_mode((GAME_WIDTH + ALGO_WIDTH + GAP, max(GAME_HEIGHT, ALGO_HEIGHT)))
pg.display.set_caption("8-Puzzle")

# Font
pg.font.init()
TITLE_FONT = pg.font.Font(None, 48)
BUTTON_FONT = pg.font.Font(None, 25)
LOG_FONT = pg.font.Font(None, 24)
TILE_FONT = pg.font.Font(None, 72)
SMALL_FONT = pg.font.Font(None, 24)

# Trạng thái ban đầu và đích
start_state = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
trang_thai_dich = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

# Log messages
log_messages = []
log_scroll_offset = 0
algorithm_runtime = None
sys.setrecursionlimit(2000)
comparison_data = {}

# Lớp Button
class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pg.Rect(x, y, width, height)
        self.text = text
        self.hovered = False

    def draw(self, surface):
        shadow_rect = self.rect.move(3, 3)
        pg.draw.rect(surface, SHADOW, shadow_rect, border_radius=8)
        color_top = BLUE if not self.hovered else LIGHT_GRAY
        color_bottom = DARK_BLUE if not self.hovered else GRAY
        for i in range(self.rect.height):
            alpha = i / self.rect.height
            color = tuple(int(color_top[j] * (1 - alpha) + color_bottom[j] * alpha) for j in range(3))
            pg.draw.line(surface, color,
                         (self.rect.x, self.rect.y + i),
                         (self.rect.right, self.rect.y + i))
        pg.draw.rect(surface, GREEN, self.rect, 2, 8)
        text_surf = BUTTON_FONT.render(self.text, True, GREEN)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == MOUSEBUTTONDOWN and self.hovered:
            return self.text
        return None

# Lớp Menu thuật toán
class AlgorithmMenu:
    # Thay thế hàm __init__ của class AlgorithmMenu
    def __init__(self, x, y, width, height):
        self.rect = pg.Rect(x, y, width, height)
        self.buttons = []
        algorithms = [
            "BFS", "DFS", "UCS", "IDS", "GBFS", "A*", "IDA*",
            "Simple HC", "Steepest HC", "Stochastic HC",
            "Simulated Annealing", "Beam Stochastic HC",
            "Genetic Algorithm", "AND-OR DFS", "Sensorless",
            "Partially Observable", "Backtracking",
            "Forward Checking","Min-Conflicts", "Q-learning"
        ]
        btn_width = width - 80
        btn_height = 25
        for i, algo in enumerate(algorithms):
            btn = Button(x + 40, y + 50 + i * (btn_height + 5),
                         btn_width, btn_height, algo)
            self.buttons.append(btn)
        self.log_y = y + len(algorithms) * (btn_height + 5) + 60
        self.log_height = height - self.log_y + y

    def draw(self, surface):
        shadow_rect = self.rect.move(3, 3)
        pg.draw.rect(surface, SHADOW, shadow_rect, border_radius=12)
        pg.draw.rect(surface, GRAY, self.rect, border_radius=12)
        title = TITLE_FONT.render("Algorithms", True, WHITE)
        title_rect = title.get_rect(center=(self.rect.centerx, self.rect.y + 25))
        surface.blit(title, title_rect)
        for button in self.buttons:
            button.draw(surface)
        log_rect = pg.Rect(self.rect.x + 10, self.log_y, self.rect.width - 20, self.log_height)
        pg.draw.rect(surface, LIGHT_GRAY, log_rect, border_radius=8)
        for i, msg in enumerate(log_messages[-10:]):
            text = LOG_FONT.render(msg, True, BLACK)
            surface.blit(text, (log_rect.x + 10, log_rect.y + 5 + i * 25))

    def handle_event(self, event):
        for button in self.buttons:
            result = button.handle_event(event)
            if result:
                return result
        return None

def add_log(message):
    log_messages.append(message)
    if len(log_messages) > 100:
        log_messages.pop(0)

def ve_khung(start_or_belief, goal_state, current_state, is_belief):
    game_screen.fill(WHITE)
    current_rect = pg.Rect(0, 0, GAME_WIDTH, GAME_HEIGHT)
    shadow_rect = current_rect.move(3, 3)
    pg.draw.rect(game_screen, SHADOW, shadow_rect, border_radius=12)
    pg.draw.rect(game_screen, LIGHT_GRAY, current_rect, border_radius=12)

    tile_size = 120
    margin = 15
    grid_x = (GAME_WIDTH - 3 * tile_size - 2 * margin) // 2
    grid_y = (GAME_HEIGHT - 3 * tile_size - 2 * margin) // 2

    current_title = TILE_FONT.render("8-Puzzle", True, BLACK)
    current_title_rect = current_title.get_rect(center=(GAME_WIDTH // 2 + 10, 30))
    game_screen.blit(current_title, current_title_rect)

    log_rect = pg.Rect(10, 130, GAME_WIDTH - 20, 80)
    log_rect_left = pg.Rect(10, 130, (GAME_WIDTH - 30) // 2, 80)
    log_rect_right = pg.Rect(log_rect_left.right + 10, 130, (GAME_WIDTH - 30) // 2, 80)
    max_log_display = 5

    pg.draw.rect(game_screen, WHITE, log_rect_left, border_radius=8)
    pg.draw.rect(game_screen, BLACK, log_rect_left, 1, 8)
    for i, msg in enumerate(log_messages[-max_log_display - log_scroll_offset:-log_scroll_offset or None]):
        if i >= max_log_display:
            break
        text = LOG_FONT.render(msg, True, BLACK)
        text_rect = text.get_rect(topleft=(log_rect_left.x + 10, log_rect_left.y + 5 + i * 25))
        if text_rect.bottom <= log_rect_left.bottom:
            game_screen.blit(text, text_rect)

    pg.draw.rect(game_screen, WHITE, log_rect_right, border_radius=8)
    pg.draw.rect(game_screen, BLACK, log_rect_right, 1, 8)
    runtime_text = f"Runtime: {algorithm_runtime:.10f}s" if algorithm_runtime is not None else "Runtime: N/A"
    runtime_surf = LOG_FONT.render(runtime_text, True, BLACK)
    runtime_rect = runtime_surf.get_rect(center=log_rect_right.center)
    game_screen.blit(runtime_surf, runtime_rect)

    for i in range(3):
        for j in range(3):
            x = grid_x + j * (tile_size + margin)
            y = grid_y + i * (tile_size + margin) + 100
            tile = current_state[i][j]
            tile_rect = pg.Rect(x, y, tile_size, tile_size)
            shadow_rect = tile_rect.move(2, 2)
            pg.draw.rect(game_screen, SHADOW, shadow_rect, border_radius=8)
            color = WHITE if tile is not None else GRAY
            pg.draw.rect(game_screen, color, tile_rect, border_radius=8)
            pg.draw.rect(game_screen, GREEN, tile_rect, 1, 8)
            if tile is not None and tile != 0:
                text = TILE_FONT.render(str(tile), True, BLACK)
                text_rect = text.get_rect(center=tile_rect.center)
                game_screen.blit(text, text_rect)

    small_tile_size = 25
    small_margin = 3
    start_x = 10
    start_y = 35

    start_state_display = start_or_belief if start_or_belief else [[None for _ in range(3)] for _ in range(3)]
    start_title = SMALL_FONT.render("Start", True, BLACK)
    start_title_rect = start_title.get_rect(
        center=(start_x + (3 * small_tile_size + 2 * small_margin) // 2, start_y - 20))
    game_screen.blit(start_title, start_title_rect)

    for i in range(3):
        for j in range(3):
            x = start_x + j * (small_tile_size + small_margin)
            y = start_y + i * (small_tile_size + small_margin)
            tile = start_state_display[i][j]
            tile_rect = pg.Rect(x, y, small_tile_size, small_tile_size)
            shadow_rect = tile_rect.move(1, 1)
            pg.draw.rect(game_screen, SHADOW, shadow_rect, border_radius=4)
            color = WHITE if tile is not None else GRAY
            pg.draw.rect(game_screen, color, tile_rect, border_radius=4)
            pg.draw.rect(game_screen, BLACK, tile_rect, 1, 4)
            if tile is not None and tile != 0:
                text = SMALL_FONT.render(str(tile), True, BLACK)
                text_rect = text.get_rect(center=tile_rect.center)
                game_screen.blit(text, text_rect)

    goal_x = start_x + 450
    goal_y = start_y
    goal_title = SMALL_FONT.render("Goal", True, BLACK)
    goal_title_rect = goal_title.get_rect(
        center=(goal_x + (3 * small_tile_size + 2 * small_margin) // 2, goal_y - 20))
    game_screen.blit(goal_title, goal_title_rect)

    for i in range(3):
        for j in range(3):
            x = goal_x + j * (small_tile_size + small_margin)
            y = goal_y + i * (small_tile_size + small_margin)
            tile = goal_state[i][j]
            tile_rect = pg.Rect(x, y, small_tile_size, small_tile_size)
            shadow_rect = tile_rect.move(1, 1)
            pg.draw.rect(game_screen, SHADOW, shadow_rect, border_radius=4)
            color = WHITE if tile != 0 else GRAY
            pg.draw.rect(game_screen, color, tile_rect, border_radius=4)
            pg.draw.rect(game_screen, BLACK, tile_rect, 1, 4)
            if tile != 0:
                text = SMALL_FONT.render(str(tile), True, BLACK)
                text_rect = text.get_rect(center=tile_rect.center)
                game_screen.blit(text, text_rect)

def get_zero_pos(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j
    return None

def is_valid_state(state):
    flat = [num for row in state for num in row]
    return sorted(flat) == list(range(9))

def move(state, direction):
    new_state = [row[:] for row in state]
    zero_pos = get_zero_pos(new_state)
    if zero_pos is None:
        print("LỖI: Trạng thái không hợp lệ, không tìm thấy số 0!")
        return None
    i, j = zero_pos
    if direction == "up" and i > 0:
        new_state[i][j], new_state[i - 1][j] = new_state[i - 1][j], new_state[i][j]
    elif direction == "down" and i < 2:
        new_state[i][j], new_state[i + 1][j] = new_state[i + 1][j], new_state[i][j]
    elif direction == "left" and j > 0:
        new_state[i][j], new_state[i][j - 1] = new_state[i][j - 1], new_state[i][j]
    elif direction == "right" and j < 2:
        new_state[i][j], new_state[i][j + 1] = new_state[i][j + 1], new_state[i][j]
    else:
        return None
    return new_state

def tao_buoc_moi(state):
    directions = ["up", "down", "left", "right"]
    new_states = []
    for direction in directions:
        new_state = move(state, direction)
        if new_state:
            new_states.append(new_state)
    return new_states

def state_to_string(state):
    return str([item for sublist in state for item in sublist])

def string_to_state(state_str):
    state_str = eval(state_str)
    return [[state_str[i * 3 + j] for j in range(3)] for i in range(3)]

def heuristic(state):
    total_distance = 0
    for i in range(3):
        for j in range(3):
            value = state[i][j]
            if value != 0:
                goal_i = (value - 1) // 3
                goal_j = (value - 1) % 3
                total_distance += abs(i - goal_i) + abs(j - goal_j)
    return total_distance

def belief_heuristic(belief_state):
    min_h = float('inf')
    for state_str in belief_state:
        state = string_to_state(state_str)
        h = heuristic(state)
        if h < min_h:
            min_h = h
    return min_h

def print_state(state):
    for row in state:
        print(row)
    print()

def print_solution(algo_name, path):
    print(f"{algo_name} Solution:")
    for i, state in enumerate(path):
        print(f"Step {i}:")
        print_state(state)

def generate_random_state():
    numbers = list(range(9))
    random.shuffle(numbers)
    return [[numbers[i * 3 + j] for j in range(3)] for i in range(3)]

# Thay thế hàm plot_comparison
def plot_comparison(data):
    import matplotlib.pyplot as plt  # Import trong hàm để đảm bảo thread an toàn
    plt.switch_backend('Agg')  # Backend không giao diện
    algorithms = list(data.keys())
    runtimes = [data[algo]['runtime'] for algo in algorithms]
    states = [data[algo]['states'] for algo in algorithms]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Biểu đồ thời gian thực hiện
    ax1.bar(algorithms, runtimes, color='skyblue')
    ax1.set_title('Thời gian thực hiện (giây)')
    ax1.set_ylabel('Thời gian (s)')
    ax1.tick_params(axis='x', rotation=45)

    # Biểu đồ không gian trạng thái
    ax2.bar(algorithms, states, color='lightgreen')
    ax2.set_title('Không gian trạng thái (số trạng thái khám phá)')
    ax2.set_ylabel('Số trạng thái')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('comparison.png', bbox_inches='tight')
    plt.close(fig)


def bfs(start_state):
    queue = deque([(start_state, [])])
    visited = set()
    visited.add(state_to_string(start_state))
    states_explored = 0
    while queue:
        current_state, path = queue.popleft()
        states_explored += 1
        if state_to_string(current_state) == state_to_string(trang_thai_dich):
            print_solution("BFS", path + [current_state])
            return path + [current_state], start_state, states_explored
        for new_state in tao_buoc_moi(current_state):
            if state_to_string(new_state) not in visited:
                visited.add(state_to_string(new_state))
                queue.append((new_state, path + [current_state]))
                states_explored += 1
    return None, start_state, states_explored

# Thay thế hàm dfs
def dfs(start_state, max_depth=50):
    stack = [(start_state, [])]
    visited = set()
    visited.add(state_to_string(start_state))
    states_explored = 0
    while stack:
        current_state, path = stack.pop()
        states_explored += 1
        if state_to_string(current_state) == state_to_string(trang_thai_dich):
            print_solution("DFS", path + [current_state])
            return path + [current_state], start_state, states_explored
        if len(path) < max_depth:
            for new_state in reversed(tao_buoc_moi(current_state)):
                if state_to_string(new_state) not in visited:
                    visited.add(state_to_string(new_state))
                    stack.append((new_state, path + [current_state]))
                    states_explored += 1
    return None, start_state, states_explored

# Thay thế hàm ucs
def ucs(start_state):
    counter = 0
    node_storage = {counter: (start_state, [])}
    pq = [(0, counter)]
    visited = set()
    visited.add(state_to_string(start_state))
    states_explored = 0
    while pq:
        cost, node_id = heapq.heappop(pq)
        current_state, path = node_storage[node_id]
        states_explored += 1
        if state_to_string(current_state) == state_to_string(trang_thai_dich):
            print_solution("UCS", path + [current_state])
            return path + [current_state], start_state, states_explored
        for new_state in tao_buoc_moi(current_state):
            if state_to_string(new_state) not in visited:
                visited.add(state_to_string(new_state))
                counter += 1
                node_storage[counter] = (new_state, path + [current_state])
                heapq.heappush(pq, (cost + 1, counter))
                states_explored += 1
    return None, start_state, states_explored

# Thay thế hàm gbfs
def gbfs(start_state):
    counter = 0
    node_storage = {counter: (start_state, [])}
    pq = [(heuristic(start_state), counter)]
    visited = set()
    visited.add(state_to_string(start_state))
    states_explored = 0
    while pq:
        h, node_id = heapq.heappop(pq)
        current_state, path = node_storage[node_id]
        states_explored += 1
        if state_to_string(current_state) == state_to_string(trang_thai_dich):
            print_solution("GBFS", path + [current_state])
            return path + [current_state], start_state, states_explored
        for new_state in tao_buoc_moi(current_state):
            if state_to_string(new_state) not in visited:
                visited.add(state_to_string(new_state))
                counter += 1
                node_storage[counter] = (new_state, path + [current_state])
                heapq.heappush(pq, (heuristic(new_state), counter))
                states_explored += 1
    return None, start_state, states_explored

# Thay thế hàm astar
def astar(start_state):
    counter = 0
    node_storage = {counter: (start_state, [], 0)}
    pq = [(heuristic(start_state), 0, counter)]
    visited = set()
    visited.add(state_to_string(start_state))
    states_explored = 0
    while pq:
        f, g, node_id = heapq.heappop(pq)
        current_state, path, g = node_storage[node_id]
        states_explored += 1
        if state_to_string(current_state) == state_to_string(trang_thai_dich):
            print_solution("A*", path + [current_state])
            return path + [current_state], start_state, states_explored
        for new_state in tao_buoc_moi(current_state):
            if state_to_string(new_state) not in visited:
                visited.add(state_to_string(new_state))
                new_g = g + 1
                new_f = new_g + heuristic(new_state)
                counter += 1
                node_storage[counter] = (new_state, path + [current_state], new_g)
                heapq.heappush(pq, (new_f, new_g, counter))
                states_explored += 1
    return None, start_state, states_explored

# Thay thế hàm iddfs
def ids(start_state):
    states_explored = 0
    for depth in range(50):
        stack = [(start_state, [], 0)]
        visited = set()
        visited.add(state_to_string(start_state) + f"_{depth}")
        while stack:
            current_state, path, current_depth = stack.pop()
            states_explored += 1
            if state_to_string(current_state) == state_to_string(trang_thai_dich):
                print_solution("IDDFS", path + [current_state])
                return path + [current_state], start_state, states_explored
            if current_depth < depth:
                for new_state in reversed(tao_buoc_moi(current_state)):
                    state_key = state_to_string(new_state) + f"_{depth}"
                    if state_key not in visited:
                        visited.add(state_key)
                        stack.append((new_state, path + [current_state], current_depth + 1))
                        states_explored += 1
    print("IDDFS: No solution found!")
    return None, start_state, states_explored
def ida_star(start_state):
    def search(state, g, path, threshold, visited):
        states_explored[0] += 1
        h = heuristic(state)
        f = g + h
        if f > threshold:
            return f, None
        if state_to_string(state) == state_to_string(trang_thai_dich):
            return f, path + [state]
        min_threshold = float('inf')
        for new_state in tao_buoc_moi(state):
            state_str = state_to_string(new_state)
            if state_str not in visited:
                visited.add(state_str)
                new_g = g + 1
                new_threshold, result = search(new_state, new_g, path + [state], threshold, visited)
                if result is not None:
                    return new_threshold, result
                min_threshold = min(min_threshold, new_threshold)
                visited.remove(state_str)
        return min_threshold, None

    states_explored = [0]  # Sử dụng list để cập nhật trong hàm lồng
    threshold = heuristic(start_state)
    visited = set()
    visited.add(state_to_string(start_state))
    while True:
        new_threshold, result = search(start_state, 0, [], threshold, visited)
        if result is not None:
            print_solution("IDA*", result)
            return result, start_state, states_explored[0]
        if new_threshold == float('inf'):
            print("IDA*: No solution found!")
            return None, start_state, states_explored[0]
        threshold = new_threshold


def simple_hill_climbing(start_state):
    current_state = [row[:] for row in start_state]
    path = [current_state]
    visited = set()
    visited.add(state_to_string(current_state))
    states_explored = 0
    while state_to_string(current_state) != state_to_string(trang_thai_dich):
        states_explored += 1
        neighbors = tao_buoc_moi(current_state)
        current_h = heuristic(current_state)
        next_state = None
        for neighbor in neighbors:
            if state_to_string(neighbor) not in visited:
                states_explored += 1
                if heuristic(neighbor) < current_h:
                    next_state = neighbor
                    break
        if next_state is None:
            print("Simple Hill Climbing: No solution found (local minimum reached)")
            return None, start_state, states_explored
        current_state = [row[:] for row in next_state]
        visited.add(state_to_string(current_state))
        path.append(current_state)
    print_solution("Simple Hill Climbing", path)
    return path, start_state, states_explored




def steepest_ascent_hill_climbing(start_state):
    current_state = [row[:] for row in start_state]
    path = [current_state]
    visited = set()
    visited.add(state_to_string(current_state))
    states_explored = 0

    while state_to_string(current_state) != state_to_string(trang_thai_dich):
        neighbors = tao_buoc_moi(current_state)
        current_h = heuristic(current_state)
        best_state = None
        best_h = float('inf')

        states_explored += 1
        for neighbor in neighbors:
            if state_to_string(neighbor) not in visited:
                states_explored += 1
                h = heuristic(neighbor)
                if h < best_h:
                    best_h = h
                    best_state = neighbor

        if best_state is None or best_h >= current_h:
            print("Steepest-Ascent Hill Climbing: No solution found (local minimum reached)")
            return None, start_state, states_explored  # Trả về 3 giá trị

        current_state = [row[:] for row in best_state]
        visited.add(state_to_string(current_state))
        path.append(current_state)

    print_solution("Steepest-Ascent Hill Climbing", path)
    return path, start_state, states_explored

def stochastic_hill_climbing(start_state):
    current_state = [row[:] for row in start_state]
    path = [current_state]
    visited = set()
    visited.add(state_to_string(current_state))
    states_explored = 0

    while state_to_string(current_state) != state_to_string(trang_thai_dich):
        neighbors = tao_buoc_moi(current_state)
        current_h = heuristic(current_state)
        better_neighbors = []
        states_explored += 1

        for neighbor in neighbors:
            if state_to_string(neighbor) not in visited:
                states_explored += 1
                h = heuristic(neighbor)
                if h < current_h:
                    better_neighbors.append(neighbor)

        if not better_neighbors:
            print("Stochastic Hill Climbing: No solution found (local minimum reached)")
            return None, start_state, states_explored  # Trả về 3 giá trị

        next_state = random.choice(better_neighbors)
        current_state = [row[:] for row in next_state]
        visited.add(state_to_string(current_state))
        path.append(current_state)

    print_solution("Stochastic Hill Climbing", path)
    return path, start_state, states_explored

def simulated_annealing(start_state, initial_temp=2000, cooling_rate=0.99, min_temp=0.1, max_iterations=5000):
    current_state = [row[:] for row in start_state]
    best_state = [row[:] for row in start_state]
    path = [current_state]
    temp = initial_temp
    iterations = 0
    best_h = heuristic(best_state)
    stagnation_count = 0
    states_explored = 0

    while temp > min_temp and iterations < max_iterations:
        if state_to_string(current_state) == state_to_string(trang_thai_dich):
            break

        neighbors = tao_buoc_moi(current_state)
        if not neighbors:
            break
        states_explored += 1
        states_explored += len(neighbors)
        current_h = heuristic(current_state)
        best_neighbor = min(neighbors, key=heuristic, default=current_state)
        next_state = best_neighbor if random.random() < 0.7 else random.choice(neighbors)
        next_h = heuristic(next_state)
        delta_h = next_h - current_h

        if delta_h <= 0 or (random.random() < math.exp(-delta_h / temp) and stagnation_count < 10):
            current_state = [row[:] for row in next_state]
            if next_h < best_h or state_to_string(next_state) == state_to_string(trang_thai_dich):
                best_h = next_h
                best_state = [row[:] for row in current_state]
                path.append(current_state)
                stagnation_count = 0
            else:
                stagnation_count += 1

        temp *= cooling_rate
        iterations += 1

    if state_to_string(best_state) == state_to_string(trang_thai_dich) or state_to_string(
            current_state) == state_to_string(trang_thai_dich):
        print_solution("Simulated Annealing", path)
        return path, start_state, states_explored
    else:
        print("Simulated Annealing: No solution found")
        return None, start_state, states_explored  # Trả về 3 giá trị

def beam_stochastic_hill_climbing(start_state, beam_width=3, max_iterations=1000):
    counter = 0
    node_storage = {counter: (start_state, [])}
    current_beam = [(heuristic(start_state), counter)]
    visited = set()
    visited.add(state_to_string(start_state))
    iterations = 0
    states_explored = 0

    while iterations < max_iterations:
        next_candidates = []
        states_explored += len(current_beam)
        for h, node_id in current_beam:
            current_state, path = node_storage[node_id]
            if state_to_string(current_state) == state_to_string(trang_thai_dich):
                print_solution("Beam Stochastic HC", path + [current_state])
                return path + [current_state], start_state, states_explored
            current_h = heuristic(current_state)
            neighbors = tao_buoc_moi(current_state)
            states_explored += len(neighbors)
            for neighbor in neighbors:
                neighbor_str = state_to_string(neighbor)
                if neighbor_str not in visited:
                    h = heuristic(neighbor)
                    if h < current_h:
                        counter += 1
                        node_storage[counter] = (neighbor, path + [current_state])
                        next_candidates.append((h, counter))

        if not next_candidates:
            print("Beam Stochastic HC: No solution found (local minimum reached)")
            return None, start_state, states_explored  # Trả về 3 giá trị

        next_candidates.sort(key=lambda x: x[0])
        top_candidates = next_candidates[:min(beam_width, len(next_candidates))]
        current_beam = []
        visited_count = 0

        while len(current_beam) < beam_width and top_candidates and visited_count < len(next_candidates):
            h, node_id = random.choice(top_candidates)
            selected_state, _ = node_storage[node_id]
            selected_str = state_to_string(selected_state)
            if selected_str not in visited:
                visited.add(selected_str)
                current_beam.append((h, node_id))
                top_candidates.remove((h, node_id))
            else:
                visited_count += 1

        if not current_beam:
            print("Beam Stochastic HC: No solution found (all top states visited)")
            return None, start_state, states_explored  # Trả về 3 giá trị

        iterations += 1

    print(f"Beam Stochastic HC: No solution found after {max_iterations} iterations")
    return None, start_state, states_explored  # Trả về 3 giá trị

def flatten_state(state):
    return [state[i][j] for i in range(3) for j in range(3)]

def unflatten_state(flat_state):
    return [[flat_state[i * 3 + j] for j in range(3)] for i in range(3)]

def fitness(state):
    h = heuristic(state)
    return 1 / (h + 1) if h > 0 else float('inf')

def crossover(parent1, parent2):
    flat_p1 = flatten_state(parent1)
    flat_p2 = flatten_state(parent2)
    point = random.randint(1, 7)
    child = flat_p1[:point] + flat_p2[point:]
    used = set(child)
    missing = [x for x in range(9) if x not in used]
    for i in range(len(child)):
        if child.count(child[i]) > 1:
            child[i] = missing.pop(0)
    return unflatten_state(child)

def mutate(state, mutation_rate=0.1):
    if random.random() < mutation_rate:
        flat_state = flatten_state(state)
        i, j = random.sample(range(9), 2)
        flat_state[i], flat_state[j] = flat_state[j], flat_state[i]
        return unflatten_state(flat_state)
    return [row[:] for row in state]

def genetic_algorithm(start_state, population_size=100, generations=500, mutation_rate=0.1):
    population = [generate_random_state() for _ in range(population_size - 1)]
    population.append([row[:] for row in start_state])
    states_explored = population_size

    for generation in range(generations):
        fitness_scores = [(fitness(ind), ind) for ind in population]
        fitness_scores.sort(reverse=True)
        states_explored += len(population)

        if fitness_scores[0][0] == float('inf'):
            best_state = fitness_scores[0][1]
            path = [start_state, best_state]
            print_solution("Genetic Algorithm", path)
            return path, start_state, states_explored

        elite_size = population_size // 2
        new_population = [ind for _, ind in fitness_scores[:elite_size]]

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(new_population[:elite_size], 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
            states_explored += 1
        population = new_population

    print("Genetic Algorithm: No solution found!")
    return None, start_state, states_explored

def and_or_tree_search_dfs(start_state, max_depth=50):
    def or_search(state, path, depth, visited, states_explored):
        states_explored[0] += 1
        state_str = state_to_string(state)
        if state_str in visited or depth >= max_depth:
            return None, states_explored
        visited.add(state_str)
        if state_to_string(state) == state_to_string(trang_thai_dich):
            return path + [state], states_explored
        # AND node: try all possible moves
        result, states_explored = and_search(state, path, depth, visited, states_explored)
        return result, states_explored

    def and_search(state, path, depth, visited, states_explored):
        neighbors = tao_buoc_moi(state)
        states_explored[0] += len(neighbors)
        # Explore all neighbors (AND node)
        for neighbor in neighbors:  # Không đảo ngược để giữ thứ tự tự nhiên
            result, states_explored = or_search(neighbor, path + [state], depth + 1, visited, states_explored)
            if result is not None:
                return result, states_explored
        return None, states_explored

    visited = set()
    states_explored = [0]  # Dùng list để cập nhật trong hàm lồng
    path, states_explored = or_search(start_state, [], 0, visited, states_explored)
    if path is not None:
        print_solution("AND-OR DFS", path)
        return path, start_state, states_explored[0]
    print("AND-OR DFS: No solution found!")
    return None, start_state, states_explored[0]

def sensorless_problem(initial_belief, max_actions=50, update_display_callback=None):
    initial_belief = set(initial_belief)
    explored = set()
    num_explored_states = 0
    belief_states_path = [list(initial_belief)]  # Lưu lịch sử các tập niềm tin
    total_steps = 0

    # Khởi tạo explored states
    for state in initial_belief:
        explored.add(state)
        num_explored_states += 1

    queue = deque([(initial_belief, [])])
    visited = set()
    actions = ["up", "down", "left", "right"]

    while queue and num_explored_states < 200:
        current_belief, action_sequence = queue.popleft()
        belief_state_tuple = frozenset(current_belief)
        num_explored_states += 1

        # Kiểm tra mục tiêu: Tất cả trạng thái trong belief_state phải là goal
        if all(state == state_to_string(trang_thai_dich) for state in current_belief):
            for initial_state in initial_belief:
                solution = bfs(string_to_state(initial_state))
                if not solution:
                    if update_display_callback:
                        update_display_callback(string_to_state(list(initial_belief)[0]), initial_belief,
                                               "Sensorless: Không tìm thấy giải pháp")
                    return None, explored, 0
                total_steps += len(solution) - 1
            belief_states_path.append([state_to_string(trang_thai_dich)] * len(initial_belief))
            if update_display_callback:
                update_display_callback(string_to_state(list(initial_belief)[0]), current_belief,
                                       "Tìm thấy giải pháp Sensorless")
            print(f"Final belief: {list(current_belief)}")
            return belief_states_path, explored, total_steps

        if belief_state_tuple in visited:
            continue
        visited.add(belief_state_tuple)

        if len(action_sequence) >= max_actions:
            continue

        for action in actions:
            new_belief = set()
            for state_str in current_belief:
                state = string_to_state(state_str)
                new_state = move(state, action)
                if new_state:
                    new_belief.add(state_to_string(new_state))
                    # Thêm trạng thái không xác định với xác suất 10%
                    if random.random() < 0.05:
                        i, j = None, None
                        for r in range(3):
                            for c in range(3):
                                if new_state[r][c] == 0:
                                    i, j = r, c
                                    break
                        directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # Lên, xuống, phải, trái
                        valid_directions = [(di, dj) for di, dj in directions if
                                            0 <= i + di < 3 and 0 <= j + dj < 3]
                        if valid_directions:
                            di, dj = random.choice(valid_directions)
                            ni, nj = i + di, j + dj
                            state_list = [list(row) for row in new_state]
                            state_list[i][j], state_list[ni][nj] = state_list[ni][nj], state_list[i][j]
                            uncertain_state = state_to_string(state_list)
                            new_belief.add(uncertain_state)
                else:
                    new_belief.add(state_str)

            # Thu hẹp belief_state: Giữ tối đa 3 trạng thái tốt nhất
            if new_belief:
                goal_str = state_to_string(trang_thai_dich)
                goal_states = [s for s in new_belief if s == goal_str]
                other_states = sorted([s for s in new_belief if s != goal_str],
                                      key=lambda s: heuristic(string_to_state(s)))
                new_belief = set(goal_states + other_states[:3 - len(goal_states)])
                for state in new_belief:
                    if state not in explored:
                        explored.add(state)
                        num_explored_states += 1
                queue.append((new_belief, action_sequence + [action]))
                belief_states_path.append(list(new_belief))  # Thêm tập niềm tin mới vào lịch sử
                if update_display_callback:
                    update_display_callback(string_to_state(list(new_belief)[0]), new_belief,
                                           f"Thử hành động {action}", action=action)

    if update_display_callback:
        update_display_callback(string_to_state(list(initial_belief)[0]), initial_belief,
                               "Sensorless: Không tìm thấy giải pháp")
    return None, explored, 0

def partially_observable_search(initial_belief, max_actions=100, update_display_callback=None):
    def apply_action(belief_state, action):
        new_belief = set()
        for state_str in belief_state:
            state = string_to_state(state_str)
            new_state = move(state, action)
            if new_state:
                if new_state[0][0] == 1:
                    new_belief.add(state_to_string(new_state))
                else:
                    new_belief.add(state_str)
            else:
                new_belief.add(state_str)
        return new_belief

    def belief_heuristic(belief):
        return min(heuristic(string_to_state(s)) for s in belief)  # Heuristic của tập niềm tin

    initial_belief = set(initial_belief)
    explored = set()
    num_explored_states = 0
    belief_states_path = [list(initial_belief)]
    total_steps = 0

    for state in initial_belief:
        explored.add(state)
        num_explored_states += 1

    queue = [(belief_heuristic(initial_belief), initial_belief, [])]  # Sử dụng priority queue
    heapq.heapify(queue)
    visited = set()
    max_queue_size = 5000

    if update_display_callback:
        update_display_callback(string_to_state(list(initial_belief)[0]), initial_belief,
                               "Bắt đầu Partially Observable từ niềm tin ban đầu")

    while queue and num_explored_states < max_queue_size:
        _, current_belief, action_sequence = heapq.heappop(queue)
        belief_state_tuple = frozenset(current_belief)
        num_explored_states += 1

        if all(state == state_to_string(trang_thai_dich) for state in current_belief):
            for initial_state in initial_belief:
                solution = bfs(string_to_state(initial_state))
                if not solution:
                    if update_display_callback:
                        update_display_callback(string_to_state(list(initial_belief)[0]), initial_belief,
                                               "Partially Observable: Không tìm thấy giải pháp")
                    return None, explored, 0
                total_steps += len(solution) - 1
            belief_states_path.append([state_to_string(trang_thai_dich)] * len(initial_belief))
            if update_display_callback:
                update_display_callback(string_to_state(list(initial_belief)[0]), current_belief,
                                       "Tìm thấy giải pháp Partially Observable")
            print(f"Final belief: {list(current_belief)}")
            return belief_states_path, explored, total_steps

        if belief_state_tuple in visited:
            continue
        visited.add(belief_state_tuple)

        if len(action_sequence) >= max_actions:
            continue
        actions = ["up", "down", "left", "right"]

        for action in actions:
            new_belief = apply_action(current_belief, action)
            if new_belief:
                new_belief = set(sorted(new_belief, key=lambda s: heuristic(string_to_state(s)))[:3])  # Thu hẹp xuống 3 trạng thái
                for state in new_belief:
                    if state not in explored:
                        explored.add(state)
                        num_explored_states += 1
                if frozenset(new_belief) not in visited:
                    heapq.heappush(queue, (belief_heuristic(new_belief), new_belief, action_sequence + [action]))
                    belief_states_path.append(list(new_belief))
                    if update_display_callback:
                        update_display_callback(string_to_state(list(new_belief)[0]), new_belief,
                                               f"Thử hành động {action}", action=action)

    if update_display_callback:
        update_display_callback(string_to_state(list(initial_belief)[0]), initial_belief,
                               "Partially Observable: Hoàn thành tìm kiếm")
    return None, explored, 0


def is_valid_assignment(state, pos, value):
    i, j = pos
    if i == 0 and j == 0 and value != 1:
        return False
    for r in range(3):
        for c in range(3):
            if (r, c) != pos and state[r][c] == value:
                return False
    if value != 0:
        if j > 0 and state[i][j - 1] is not None and state[i][j - 1] != value - 1:
            return False
        if j < 2 and state[i][j + 1] is not None and state[i][j + 1] != value + 1:
            return False
        if i > 0 and state[i - 1][j] is not None and state[i - 1][j] != value - 3:
            return False
        if i < 2 and state[i + 1][j] is not None and state[i + 1][j] != value + 3:
            return False
    return True

def is_solvable(state):
    flat = [num for row in state for num in row if num != 0]
    inversions = 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i] > flat[j]:
                inversions += 1
    return inversions % 2 == 0

def backtracking_search(update_display_callback=None):
    state = [[None for _ in range(3)] for _ in range(3)]
    used_values = set()
    path = [copy.deepcopy(state)]
    visited = set()
    states_explored = [0]  # Sử dụng list để cập nhật trong hàm lồng
    max_depth = 50

    def state_to_string(s):
        return str([[val if val is not None else 'N' for val in row] for row in s])

    def backtrack(current_state, assigned, pos_index):
        if pos_index >= max_depth:
            if update_display_callback:
                update_display_callback(current_state, f"Đạt giới hạn độ sâu ({max_depth})")
            return None, path

        if pos_index == 9:
            states_explored[0] += 1
            if current_state == trang_thai_dich:
                path.append(copy.deepcopy(current_state))
                return current_state, path
            return None, path

        i, j = divmod(pos_index, 3)
        if i >= 3 or j >= 3:
            return None, path

        state_str = state_to_string(current_state)
        if state_str in visited:
            return None, path
        visited.add(state_str)
        states_explored[0] += 1

        values = list(range(9))
        random.shuffle(values)
        for value in values:
            if value not in assigned:
                if is_valid_assignment(current_state, (i, j), value):
                    new_state = [row[:] for row in current_state]
                    new_state[i][j] = value
                    new_assigned = assigned | {value}
                    path.append(copy.deepcopy(new_state))
                    if update_display_callback:
                        update_display_callback(new_state, f"Gán {value} vào ({i},{j})")
                    found_solution, solution_path = backtrack(new_state, new_assigned, pos_index + 1)
                    if found_solution is not None:
                        return found_solution, solution_path
                    path.pop()
                    if update_display_callback:
                        update_display_callback(new_state, f"Quay lui từ ({i},{j})")

        return None, path

    if update_display_callback:
        update_display_callback(state, "Bắt đầu Backtracking từ ma trận rỗng...")
    start_time = time.time()
    timeout = 60

    found_solution, solution_path = backtrack(state, used_values, 0)

    if time.time() - start_time > timeout:
        if update_display_callback:
            update_display_callback(state, "Hết thời gian")
        return None, state, states_explored[0]

    if found_solution is None:
        if update_display_callback:
            update_display_callback(state, "Không tìm thấy giải pháp")
        return None, state, states_explored[0]

    if update_display_callback:
        update_display_callback(found_solution, "Tìm thấy giải pháp")
    return found_solution, state, states_explored[0]

def forward_checking(start_state, update_display_callback=None):
    def is_valid_state(state):
        flat = [state[i][j] for i in range(3) for j in range(3) if state[i][j] is not None]
        return len(set(flat)) == len(flat) and all(0 <= x <= 8 for x in flat)

    def is_complete_state(state):
        return all(state[i][j] is not None for i in range(3) for j in range(3))

    def is_solvable(state):
        if not is_complete_state(state):
            return True  # Trạng thái chưa hoàn chỉnh, giả định có thể giải được
        flat = [state[i][j] for i in range(3) for j in range(3) if state[i][j] != 0]
        inversions = 0
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                if flat[i] > flat[j]:
                    inversions += 1
        return inversions % 2 == 0  # Trạng thái khả nghiệm nếu số nghịch đảo là chẵn

    def backtrack_with_fc(state, assigned, positions, domains, path, visited, states_explored_count, depth_limit=9):
        if states_explored_count[0] > 1000 or time.time() - start_time > 10:
            return None

        if len(assigned) == 9 and is_complete_state(state):
            state_tuple = tuple(tuple(row) for row in state)
            if state_tuple == tuple(tuple(row) for row in trang_thai_dich) and is_solvable(state):
                if state not in path:
                    path.append([row[:] for row in state])
                return path
            return None

        if len(assigned) >= 7:
            temp_state = [row[:] for row in state]
            temp_assigned = assigned.copy()
            temp_positions = [p for p in positions if p not in assigned]
            temp_domains = {k: v[:] for k, v in domains.items()}
            for p in temp_positions:
                remaining_values = [v for v in range(9) if v not in temp_assigned.values()]
                if not remaining_values:
                    return None
                value = remaining_values[0]
                temp_state[p[0]][p[1]] = value
                temp_assigned[p] = value
                path.append([row[:] for row in temp_state])
                success, temp_domains = forward_check(temp_state, p, value, temp_domains, temp_assigned)
                if not success:
                    path.pop()
                    return None
            if is_complete_state(temp_state):
                state_tuple = tuple(tuple(row) for row in temp_state)
                if state_tuple == tuple(tuple(row) for row in trang_thai_dich) and is_solvable(temp_state):
                    return path
                path.pop()
            return None

        pos = select_mrv_variable(positions, domains, state)
        if pos is None:
            return None

        domain = get_domain(state, pos, set(assigned.values()))
        sorted_values = select_lcv_value(pos, domain, state, domains, assigned)

        state_tuple = tuple(tuple(row if row is not None else (-1, -1, -1)) for row in state)
        if state_tuple in visited:
            return None
        visited.add(state_tuple)
        states_explored_count[0] += 1

        for value in sorted_values:
            new_state = [row[:] for row in state]
            new_state[pos[0]][pos[1]] = value
            new_assigned = assigned.copy()
            new_assigned[pos] = value
            new_positions = [p for p in positions if p != pos]
            path.append([row[:] for row in new_state])
            if update_display_callback:
                update_display_callback([row[:] for row in new_state], None, f"Assigning {value} to position {pos}")
            success, new_domains = forward_check(new_state, pos, value, domains, new_assigned)
            if success:
                result = backtrack_with_fc(new_state, new_assigned, new_positions, new_domains, path, visited, states_explored_count, depth_limit)
                if result is not None:
                    return result
            path.pop()

        return None

    start_time = time.time()
    states_explored_count = [0]  # Sử dụng list để cập nhật
    empty_state = [[None for _ in range(3)] for _ in range(3)]
    positions = [(i, j) for i in range(3) for j in range(3)]
    domains = {(i, j): list(range(9)) for i in range(3) for j in range(3)}
    domains[(0, 0)] = [1]
    assigned = {}
    visited = set()
    path = []

    if update_display_callback:
        update_display_callback(empty_state, None, "Starting Forward Checking from empty state")

    result = backtrack_with_fc(empty_state, assigned, positions, domains, path, visited, states_explored_count)

    if result:
        if update_display_callback:
            update_display_callback(trang_thai_dich, None, "Forward Checking: Solution found")
        return result, empty_state, states_explored_count[0]
    else:
        if update_display_callback:
            update_display_callback(empty_state, None, "Forward Checking: No solution found")
        return None, empty_state, states_explored_count[0]


# Hàm Min-Conflicts
def count_conflicts(state):
    """Tính tổng xung đột của trạng thái, kết hợp Manhattan Distance, Misplaced Tiles và Linear Conflicts"""
    conflicts = 0
    goal_positions = {trang_thai_dich[i][j]: (i, j) for i in range(3) for j in range(3) if trang_thai_dich[i][j] != 0}

    # 1. Khoảng cách Manhattan
    manhattan = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:
                goal_i, goal_j = goal_positions.get(state[i][j], (i, j))
                manhattan += abs(i - goal_i) + abs(j - goal_j)

    # 2. Số ô sai vị trí
    misplaced = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0 and state[i][j] != trang_thai_dich[i][j]:
                misplaced += 1

    # 3. Xung đột tuyến tính
    linear_conflicts = 0
    for i in range(3):
        # Xung đột trong hàng
        row_tiles = [(state[i][j], j) for j in range(3) if state[i][j] != 0]
        for idx1 in range(len(row_tiles)):
            for idx2 in range(idx1 + 1, len(row_tiles)):
                tile1, col1 = row_tiles[idx1]
                tile2, col2 = row_tiles[idx2]
                goal_pos1 = goal_positions.get(tile1, (i, col1))
                goal_pos2 = goal_positions.get(tile2, (i, col2))
                if goal_pos1[0] == i and goal_pos2[0] == i:  # Cả hai ô thuộc hàng i trong trạng thái đích
                    if goal_pos1[1] > goal_pos2[1] and col1 < col2:  # Sai thứ tự
                        linear_conflicts += 2  # Cần ít nhất 2 di chuyển để sửa
        # Xung đột trong cột
        col_tiles = [(state[j][i], j) for j in range(3) if state[j][i] != 0]
        for idx1 in range(len(col_tiles)):
            for idx2 in range(idx1 + 1, len(col_tiles)):
                tile1, row1 = col_tiles[idx1]
                tile2, row2 = col_tiles[idx2]
                goal_pos1 = goal_positions.get(tile1, (row1, i))
                goal_pos2 = goal_positions.get(tile2, (row2, i))
                if goal_pos1[1] == i and goal_pos2[1] == i:  # Cả hai ô thuộc cột i trong trạng thái đích
                    if goal_pos1[0] > goal_pos2[0] and row1 < row2:  # Sai thứ tự
                        linear_conflicts += 2

    # Kết hợp các heuristic: Manhattan là chính, cộng thêm Misplaced và Linear Conflicts với trọng số
    conflicts = manhattan + misplaced + linear_conflicts
    return conflicts


def is_solvable(state):
    """Kiểm tra trạng thái có giải được không bằng cách đếm nghịch đảo"""
    flat = [num for row in state for num in row if num != 0]
    inversions = 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i] > flat[j]:
                inversions += 1
    return inversions % 2 == 0


def get_move_description(prev_state, curr_state):
    """Xác định hướng di chuyển từ trạng thái trước sang trạng thái hiện tại"""
    prev_zero = get_zero_pos(prev_state)
    curr_zero = get_zero_pos(curr_state)
    if prev_zero is None or curr_zero is None:
        return "Invalid move"
    if prev_zero[0] > curr_zero[0]:
        return "Up"
    elif prev_zero[0] < curr_zero[0]:
        return "Down"
    elif prev_zero[1] > curr_zero[1]:
        return "Left"
    elif prev_zero[1] < curr_zero[1]:
        return "Right"
    return "No move"


def min_conflicts_search(start_state, max_iterations=5000, update_display_callback=None):
    """Triển khai thuật toán Min-Conflicts cải tiến với heuristic nâng cao và lịch trình làm nguội"""
    global algorithm_runtime

    if not is_valid_state(start_state) or not is_solvable(start_state):
        add_log("Min-Conflicts: Trạng thái ban đầu không hợp lệ hoặc không thể giải")
        return None, start_state, 0

    goal = trang_thai_dich
    current = [row[:] for row in start_state]
    path = [current]
    states_explored = 1
    start_time = time.time()
    stuck_count = 0  # Đếm số lần bị kẹt
    max_stuck = 50  # Giới hạn số lần bị kẹt trước khi dừng

    # Lịch trình làm nguội
    initial_temp = 100.0
    cooling_rate = 0.995
    min_temp = 0.1
    temperature = initial_temp

    for step in range(max_iterations):
        if current == goal:
            algorithm_runtime = time.time() - start_time
            add_log(f"Min-Conflicts: Tìm thấy giải pháp sau {step} bước")
            return path, start_state, states_explored

        zero_pos = None
        for i in range(3):
            for j in range(3):
                if current[i][j] == 0:
                    zero_pos = (i, j)
                    break
            if zero_pos:
                break

        if not zero_pos:
            add_log("Min-Conflicts: Không tìm thấy ô trống")
            return None, start_state, states_explored

        i, j = zero_pos
        moves = []
        min_conflicts = float('inf')

        # Đánh giá các bước di chuyển có thể
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Lên, xuống, trái, phải
            ni, nj = i + di, j + dj
            if 0 <= ni < 3 and 0 <= nj < 3:
                new_state = [row[:] for row in current]
                new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
                if is_solvable(new_state):  # Chỉ xem xét trạng thái có thể giải
                    conflicts = count_conflicts(new_state)
                    states_explored += 1
                    moves.append((conflicts, new_state))
                    min_conflicts = min(min_conflicts, conflicts)

        if not moves:
            add_log("Min-Conflicts: Không có di chuyển hợp lệ")
            return None, start_state, states_explored

        # Chọn trạng thái tiếp theo
        current_conflicts = count_conflicts(current)
        best_moves = [m for c, m in moves if c < current_conflicts]

        if best_moves:
            stuck_count = 0  # Đặt lại khi tìm thấy di chuyển tốt
            # Chọn ngẫu nhiên từ các di chuyển tốt nhất
            next_state = random.choice(best_moves)
        else:
            stuck_count += 1
            if stuck_count >= max_stuck or temperature < min_temp:
                add_log(f"Min-Conflicts: Bị kẹt quá lâu hoặc nhiệt độ quá thấp sau {step} bước")
                algorithm_runtime = time.time() - start_time
                return None, start_state, states_explored
            # Chấp nhận di chuyển xấu dựa trên lịch trình làm nguội
            conflicts_list = [c for c, _ in moves]
            worst_conflicts = max(conflicts_list)
            delta_conflicts = worst_conflicts - current_conflicts
            if delta_conflicts <= 0 or random.random() < math.exp(-delta_conflicts / temperature):
                _, next_state = random.choice(moves)
                add_log(f"Bước {step + 1}: Chấp nhận di chuyển xấu (ΔC={delta_conflicts:.2f}, T={temperature:.2f})")
            else:
                continue  # Bỏ qua bước này và thử lại

        # Cập nhật trạng thái hiện tại
        prev_state = [row[:] for row in current]
        current = [row[:] for row in next_state]
        path.append(current)

        # Giảm nhiệt độ
        temperature = max(min_temp, temperature * cooling_rate)

        # Ghi log và gọi callback
        move_desc = get_move_description(prev_state, current)
        msg = f"Bước {step + 1}: Di chuyển {move_desc}, Xung đột {count_conflicts(current)}, Nhiệt độ {temperature:.2f}"
        add_log(msg)
        if update_display_callback:
            update_display_callback(current, None, msg)

    algorithm_runtime = time.time() - start_time
    add_log(f"Min-Conflicts: Không tìm thấy giải pháp sau {max_iterations} bước")
    return None, start_state, states_explored


def min_conflicts_algorithm(start_state, update_display_callback=None):
    """Giao diện chuẩn cho thuật toán Min-Conflicts"""
    return min_conflicts_search(start_state, update_display_callback=update_display_callback)


def get_action_from_direction(dx, dy):
    """Chuyển đổi hướng di chuyển thành hành động (0: lên, 1: xuống, 2: phải, 3: trái)."""
    if dx == 0 and dy == -1:
        return 0  # Lên
    elif dx == 0 and dy == 1:
        return 1  # Xuống
    elif dx == 1 and dy == 0:
        return 2  # Phải
    elif dx == -1 and dy == 0:
        return 3  # Trái
    return None

def get_direction_from_action(action):
    """Chuyển đổi hành động thành hướng di chuyển (dx, dy)."""
    if action == 0:  # Lên
        return 0, -1
    elif action == 1:  # Xuống
        return 0, 1
    elif action == 2:  # Phải
        return 1, 0
    elif action == 3:  # Trái
        return -1, 0
    return 0, 0
def manhattan_distance(state, goal_state):
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:
                value = state[i][j]
                goal_i, goal_j = (value - 1) // 3, (value - 1) % 3
                distance += abs(i - goal_i) + abs(j - goal_j)
    return distance

def hamming_distance(state, goal_state):
    """Tính số ô không đúng vị trí so với goal_state (trừ ô trống)."""
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0 and state[i][j] != goal_state[i][j]:
                distance += 1
    return distance

def get_neighbors(state):
    """Lấy các trạng thái lân cận bằng cách di chuyển ô trống."""
    i, j = None, None
    for r in range(3):
        for c in range(3):
            if state[r][c] == 0:
                i, j = r, c
                break
    neighbors = []
    directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # Lên, xuống, phải, trái
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < 3 and 0 <= nj < 3:
            new_state = [list(row) for row in state]
            new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
            neighbors.append(tuple(tuple(row) for row in new_state))
    return neighbors

# Định nghĩa q_table toàn cục
q_table = defaultdict(lambda: {a: 0.0 for a in range(4)})  # {state: {action: q_value}}

def q_learning_algorithm(start_state, update_display_callback=None, trang_thai_dich=[[1, 2, 3], [4, 5, 6], [7, 8, 0]]):
    global algorithm_runtime, comparison_data

    if start_state is None:
        start_state = generate_random_state()
    if not is_valid_state(start_state) or not is_solvable(start_state):
        add_log("Q-Learning: Trạng thái ban đầu không hợp lệ hoặc không thể giải")
        return None, start_state, 0

    alpha = 0.2
    gamma = 0.9
    epsilon = 0.3
    convergence_threshold = 0.01
    max_episodes = 10000
    max_steps = 500
    max_runtime = 10
    max_states_explored = 10000
    states_explored = 0
    start_time = time.time()

    # Đặt Q-value cho trạng thái mục tiêu về 0
    goal_state = tuple(tuple(row) for row in trang_thai_dich)
    q_table[goal_state] = {a: 0.0 for a in range(4)}

    for episode in range(max_episodes):
        if time.time() - start_time > max_runtime or states_explored > max_states_explored:
            add_log("Q-Learning: Dừng do vượt quá thời gian hoặc số trạng thái")
            break
        current_state = tuple(tuple(row) for row in start_state)
        max_delta = 0
        epsilon = max(0.05, epsilon * 0.95)
        prev_state = None
        prev_action = None

        for step in range(max_steps):
            state_tuple = current_state
            if state_tuple == tuple(tuple(row) for row in trang_thai_dich):
                add_log(f"Q-Learning: Đạt đích trong episode {episode + 1}, bước {step + 1}")
                if prev_state is not None and prev_action is not None:
                    old_value = q_table[prev_state][prev_action]
                    q_table[prev_state][prev_action] = old_value + alpha * (1000 - old_value)
                break  # Thoát vòng lặp step

            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = max(q_table[state_tuple], key=q_table[state_tuple].get) if q_table[state_tuple] else random.randint(0, 3)

            i, j = [(r, c) for r in range(3) for c in range(3) if current_state[r][c] == 0][0]
            dx, dy = get_direction_from_action(action)
            next_i, next_j = i + dy, j + dx
            neighbors = set(get_neighbors(current_state))
            next_state = current_state
            reward = -10

            if 0 <= next_i < 3 and 0 <= next_j < 3:
                next_state = [list(row) for row in current_state]
                next_state[i][j], next_state[next_i][next_j] = next_state[next_i][next_j], next_state[i][j]
                next_state_tuple = tuple(tuple(row) for row in next_state)
                if next_state_tuple in neighbors:
                    distance_before = manhattan_distance(current_state, trang_thai_dich)
                    distance_after = manhattan_distance(next_state, trang_thai_dich)
                    reward = -0.1 + (distance_before - distance_after) * 2
                    if next_state_tuple == tuple(tuple(row) for row in trang_thai_dich):
                        reward = 1000
                    next_state = next_state_tuple

            if state_tuple != tuple(tuple(row) for row in trang_thai_dich):  # Không cập nhật Q-value cho trạng thái mục tiêu
                old_value = q_table[state_tuple][action]
                max_future_q = max(q_table[next_state].values()) if q_table[next_state] else 0.0
                q_table[state_tuple][action] = old_value + alpha * (reward + gamma * max_future_q - old_value)
                max_delta = max(max_delta, abs(old_value - q_table[state_tuple][action]))

            states_explored += 1
            current_state = next_state
            prev_state = state_tuple
            prev_action = action

            if update_display_callback and (step % 50 == 0 or reward >= 1000):
                move_desc = get_move_description([list(row) for row in state_tuple], [list(row) for row in next_state])
                msg = f"Episode {episode + 1}, Step {step + 1}: Action {move_desc}, Reward {reward:.2f}, Epsilon {epsilon:.3f}"
                try:
                    update_display_callback([list(row) for row in next_state], None, msg)
                    add_log(msg)
                except Exception as e:
                    add_log(f"Error in callback: {e}")

        if state_tuple == tuple(tuple(row) for row in trang_thai_dich):
            break  # Thoát vòng lặp episode khi đạt mục tiêu

        if episode % 100 == 0:
            avg_q = sum(sum(q_values.values()) for q_values in q_table.values()) / (len(q_table) * 4) if q_table else 0
            add_log(f"Episode {episode + 1}: States explored {states_explored}, Avg Q-value {avg_q:.2f}")

        if max_delta < convergence_threshold:
            add_log(f"Q-Learning: Hội tụ sau {episode + 1} episodes")
            break

    if goal_state in q_table:
        add_log(f"Q-values for goal state: {q_table[goal_state]}")

    path = [[list(row) for row in start_state]]
    current_state = tuple(tuple(row) for row in start_state)
    visited = set([current_state])
    steps = 0
    max_path_steps = 50
    last_action = None

    while steps < max_path_steps:
        state_tuple = current_state
        if state_tuple == tuple(tuple(row) for row in trang_thai_dich):
            add_log("Q-Learning: Đã đạt trạng thái đích, dừng trích xuất đường đi")
            break
        if state_tuple not in q_table or not q_table[state_tuple]:
            add_log("Q-Learning: Không có chính sách hợp lệ trong q_table")
            path = None
            break
        q_values = q_table[state_tuple]
        if last_action is not None:
            reverse_action = {0: 1, 1: 0, 2: 3, 3: 2}.get(last_action)
            if reverse_action in q_values:
                q_values = q_values.copy()
                q_values[reverse_action] = float('-inf')  # Ngăn hành động ngược
        action = max(q_values, key=q_values.get)
        i, j = [(r, c) for r in range(3) for c in range(3) if current_state[r][c] == 0][0]
        dx, dy = get_direction_from_action(action)
        next_i, next_j = i + dy, j + dx
        neighbors = set(get_neighbors(current_state))
        if 0 <= next_i < 3 and 0 <= next_j < 3:
            next_state = [list(row) for row in current_state]
            next_state[i][j], next_state[next_i][next_j] = next_state[next_i][next_j], next_state[i][j]
            next_state_tuple = tuple(tuple(row) for row in next_state)
            if next_state_tuple in neighbors and next_state_tuple not in visited:
                path.append([list(row) for row in next_state_tuple])
                visited.add(next_state_tuple)
                current_state = next_state_tuple
                steps += 1
                last_action = action
                if update_display_callback:
                    move_desc = get_move_description([list(row) for row in state_tuple], [list(row) for row in next_state_tuple])
                    msg = f"Final Path Step {steps}: Action {move_desc}"
                    update_display_callback([list(row) for row in next_state_tuple], None, msg)
                    add_log(msg)
                if next_state_tuple == tuple(tuple(row) for row in trang_thai_dich):
                    add_log("Q-Learning: Đã đạt trạng thái đích, dừng trích xuất đường đi")
                    break
        else:
            add_log("Q-Learning: Hành động không hợp lệ, dừng")
            break

    algorithm_runtime = time.time() - start_time
    comparison_data["Q-learning"] = {
        'runtime': algorithm_runtime,
        'states': states_explored
    }

    if path and current_state == tuple(tuple(row) for row in trang_thai_dich):
        add_log(f"Q-Learning: Tìm thấy giải pháp trong {algorithm_runtime:.2f}s với {len(path)} bước")
        print_solution("Q-learning", path)
        return path, start_state, states_explored
    add_log(f"Q-Learning: Không tìm thấy giải pháp trong {algorithm_runtime:.2f}s")
    return None, start_state, states_explored

def get_move_description(prev_state, curr_state):
    prev_zero = get_zero_pos(prev_state)
    curr_zero = get_zero_pos(curr_state)
    if prev_zero is None or curr_zero is None:
        return "Invalid move"
    if prev_zero[0] > curr_zero[0]:
        return "Up"
    elif prev_zero[0] < curr_zero[0]:
        return "Down"
    elif prev_zero[1] > curr_zero[1]:
        return "Left"
    elif prev_zero[1] < curr_zero[1]:
        return "Right"
    return "No move"

def create_belief_state_window(belief_states, algorithm_func, title):
    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 600
    screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pg.display.set_caption(title)

    WHITE = (245, 245, 245)
    BLACK = (30, 30, 30)
    GRAY = (120, 130, 140)
    GREEN = (0, 255, 0)
    SHADOW = (100, 100, 100, 50)
    TILE_FONT = pg.font.Font(None, 48)
    BUTTON_FONT = pg.font.Font(None, 36)
    SMALL_FONT = pg.font.Font(None, 24)

    run_button = Button(WINDOW_WIDTH // 2 - 120, WINDOW_HEIGHT - 70, 100, 40, "Run")
    back_button = Button(WINDOW_WIDTH // 2 + 20, WINDOW_HEIGHT - 70, 100, 40, "Back")

    tile_size = 80
    belief_matrices = [string_to_state(s) for s in belief_states]
    num_matrices = min(len(belief_matrices), 3)  # Giới hạn tối đa 3 Belief States
    max_per_row = 3
    offset_x = (WINDOW_WIDTH - (tile_size * 3 + 20) * max_per_row) // 2  # Căn giữa ngang
    offset_y = (WINDOW_HEIGHT - (tile_size * 3 + 40) * ((num_matrices - 1) // max_per_row + 1)) // 2  # Căn giữa dọc
    belief_positions = []
    for idx in range(num_matrices):
        row = idx // max_per_row
        col = idx % max_per_row
        pos_x = offset_x + col * (tile_size * 3 + 40)
        pos_y = offset_y + row * (tile_size * 3 + 40)
        belief_positions.append((pos_x, pos_y))

    solutions = [None] * num_matrices  # Lưu đường đi cho từng Belief State
    solution_indices = [0] * num_matrices  # Chỉ số bước hiện tại
    error_message = None
    error_timer = 0
    running_solution = False
    action_sequence = []

    def draw_grid(state, offset_x, offset_y, tile_size):
        for i in range(3):
            for j in range(3):
                x = offset_x + j * (tile_size + 10)
                y = offset_y + i * (tile_size + 10)
                tile = state[i][j]
                tile_rect = pg.Rect(x, y, tile_size, tile_size)
                shadow_rect = tile_rect.move(2, 2)
                pg.draw.rect(screen, SHADOW, shadow_rect, border_radius=8)
                color = WHITE if tile != 0 else GRAY
                pg.draw.rect(screen, color, tile_rect, border_radius=8)
                pg.draw.rect(screen, GREEN, tile_rect, 1, 8)
                if tile != 0:
                    text = TILE_FONT.render(str(tile), True, BLACK)
                    text_rect = text.get_rect(center=tile_rect.center)
                    screen.blit(text, text_rect)

    def callback(state, belief, msg, action=None):
        nonlocal action_sequence
        if "Thử hành động" in msg and action:
            action_sequence.append(action)
            print(f"Added action: {action}")
        print(f"Callback belief: {list(belief)}")
        print(msg)

    running = True
    clock = pg.time.Clock()
    last_update_time = pg.time.get_ticks()

    while running:
        screen.fill(GRAY)

        for idx, (pos_x, pos_y) in enumerate(belief_positions):
            if solutions[idx] is not None and solution_indices[idx] < len(solutions[idx]):
                state = string_to_state(solutions[idx][solution_indices[idx]])
            else:
                state = belief_matrices[idx]
            text = BUTTON_FONT.render(f"Belief State {idx + 1} (h={heuristic(state)})", True, BLACK)
            screen.blit(text, (pos_x, pos_y - 30))
            draw_grid(state, pos_x, pos_y, tile_size)

        if error_message and pg.time.get_ticks() - error_timer < 1000:
            text = SMALL_FONT.render(error_message, True, BLACK)
            screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, WINDOW_HEIGHT - 100))

        run_button.draw(screen)
        back_button.draw(screen)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            result = back_button.handle_event(event)
            if result == "Back":
                running = False
                break
            if not running_solution:
                result = run_button.handle_event(event)
                if result == "Run":
                    running_solution = True
                    solutions = [None] * num_matrices
                    solution_indices = [0] * num_matrices
                    error_message = None
                    action_sequence = []
                    start_time = time.time()
                    try:
                        belief_states_path, explored_states, total_steps = algorithm_func(belief_states, max_actions=50, update_display_callback=callback)
                        if belief_states_path:
                            solutions = [[] for _ in range(num_matrices)]
                            for belief_set in belief_states_path:
                                for idx in range(min(num_matrices, len(belief_set))):
                                    solutions[idx].append(belief_set[idx])
                        else:
                            error_message = "No Solution Found!"
                            error_timer = pg.time.get_ticks()
                            running_solution = False
                    except Exception as e:
                        print(f"Error in {title}: {str(e)}")
                        error_message = str(e)
                        error_timer = pg.time.get_ticks()
                        running_solution = False

        if running_solution:
            current_time = pg.time.get_ticks()
            if current_time - last_update_time >= 50:  # Thời gian chờ 50ms
                all_finished = True
                for idx in range(num_matrices):
                    if solutions[idx] and solution_indices[idx] < len(solutions[idx]) - 1:
                        solution_indices[idx] += 1
                        all_finished = False
                last_update_time = current_time
                if all_finished:
                    running_solution = False

        pg.display.flip()
        clock.tick(60)

    pg.display.set_mode((GAME_WIDTH + ALGO_WIDTH + GAP, max(GAME_HEIGHT, ALGO_HEIGHT)))
    pg.display.set_caption("8-Puzzle")
    return None


def main():
    global log_scroll_offset, algorithm_runtime, comparison_data
    pg.init()
    clock = pg.time.Clock()
    running = True

    menu = AlgorithmMenu(GAME_WIDTH + GAP, 0, ALGO_WIDTH, ALGO_HEIGHT)
    current_state = [[None for _ in range(3)] for _ in range(3)]
    start_or_belief = start_state
    is_belief = False
    solution = None
    solution_index = 0
    last_update = 0
    update_interval = 400
    solving = False
    display_states = []
    fixed_belief_state = None
    last_drawn_state = None
    selected_algo = None
    # Create Compare button below "8-Puzzle" title
    compare_button = Button(GAME_WIDTH // 2 - 50, 70, 100, 40, "Compare")

    algo_dict = {
        "BFS": bfs,
        "DFS": dfs,
        "UCS": ucs,
        "IDS": ids,
        "GBFS": gbfs,
        "A*": astar,
        "IDA*": ida_star,
        "Simple HC": simple_hill_climbing,
        "Steepest HC": steepest_ascent_hill_climbing,
        "Stochastic HC": stochastic_hill_climbing,
        "Simulated Annealing": simulated_annealing,
        "Beam Stochastic HC": beam_stochastic_hill_climbing,
        "Genetic Algorithm": genetic_algorithm,
        "AND-OR DFS": and_or_tree_search_dfs,
        "Sensorless": sensorless_problem,
        "Partially Observable": partially_observable_search,
        "Backtracking": lambda x: backtracking_search(update_display_callback=lambda state, msg: display_states.append((state, None, msg))),
        "Forward Checking": forward_checking,
        "Min-Conflicts": lambda x: min_conflicts_algorithm(
        x,
        update_display_callback=lambda state, belief, msg: display_states.append((state, None, msg))
    ),
        "Q-learning": lambda x: q_learning_algorithm(
        x,
        update_display_callback=lambda state, belief, msg: display_states.append((state, belief, msg))
    )
    }

    while running:
        current_time = pg.time.get_ticks()

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.MOUSEWHEEL and not solving:
                if menu.rect.collidepoint(pg.mouse.get_pos()):
                    log_scroll_offset = max(0, min(log_scroll_offset - event.y, len(log_messages) - 10))
            elif not solving:
                result = compare_button.handle_event(event)
                if result == "Compare":
                    if comparison_data:
                        threading.Thread(target=plot_comparison, args=(comparison_data,), daemon=True).start()
                        pg.event.pump()
                        pg.display.flip()
                        add_log("Comparison chart saved as comparison.png in current directory")
                    else:
                        add_log("No data to compare. Run algorithms first.")
                        pg.event.pump()
                        pg.display.flip()
                # Handle algorithm menu
                selected_algo = menu.handle_event(event)
                if selected_algo:
                        try:
                            solving = True
                            log_messages.clear()
                            solution = None
                            solution_index = 0
                            display_states = []
                            fixed_belief_state = None
                            algorithm_runtime = None
                            log_scroll_offset = 0
                            last_drawn_state = None
                            current_state = [[None for _ in range(3)] for _ in range(3)]
                            start_or_belief = start_state
                            is_belief = False

                            add_log(f"Running {selected_algo}...")
                            pg.display.update()

                            start_time = time.time()
                            timeout = 60 if selected_algo == "Sensorless" else 30  # Tăng timeout cho Sensorless
                            if selected_algo in ["Sensorless", "Partially Observable"]:
                                initial_belief = (generate_initial_belief_state() if selected_algo == "Sensorless"
                                                  else generate_initial_belief_state_partially())
                                create_belief_state_window(initial_belief, algo_dict[selected_algo],
                                                           f"{selected_algo} Belief States")

                                if time.time() - start_time > timeout:
                                    raise TimeoutError("Algorithm timed out")
                                belief_states_path, explored_states, total_steps = (algo_dict[selected_algo](initial_belief, max_actions=30))
                                start_or_belief = string_to_state(list(initial_belief)[0])
                                fixed_belief_state = initial_belief
                                is_belief = True
                                if belief_states_path:
                                    solution = [string_to_state(belief_set[0]) for belief_set in belief_states_path]
                                    current_state = solution[0]
                                comparison_data[selected_algo] = {
                                    'runtime': time.time() - start_time,
                                    'states': len(explored_states)
                                }
                                solving = False  # Đặt lại trạng thái

                            elif solving and selected_algo == "Min-Conflicts":
                                if display_states and solution_index < len(display_states):
                                    state, _, msg = display_states[solution_index]
                                    current_state = state
                                    add_log(msg)
                                    solution_index += 1
                                    if solution_index >= len(display_states):
                                        solving = False
                                elif solution and solution_index < len(solution):
                                    current_state = solution[solution_index]
                                    if solution_index > 0:
                                        move = get_move_description(solution[solution_index - 1],
                                                                    solution[solution_index])
                                        add_log(f"Bước {solution_index}: {move}")
                                    solution_index += 1
                                    if solution_index >= len(solution):
                                        solving = False
                            elif selected_algo in ["Q-learning", "Backtracking", "Forward Checking"]:
                                if time.time() - start_time > timeout:
                                    raise TimeoutError("Algorithm timed out")
                                result = algo_dict[selected_algo](start_state)
                                if selected_algo == "Q-learning":
                                    solution, representative_state, states_explored = result
                                    start_or_belief = representative_state
                                    fixed_belief_state = None
                                    is_belief = True
                                    comparison_data[selected_algo] = {
                                        'runtime': time.time() - start_time,
                                        'states': states_explored[0] if isinstance(states_explored, list) else states_explored
                                    }
                                else:
                                    solution, representative_state, states_explored = result
                                    start_or_belief = representative_state
                                    fixed_belief_state = None
                                    is_belief = False
                                    comparison_data[selected_algo] = {
                                        'runtime': time.time() - start_time,
                                        'states': states_explored
                                    }
                            else:
                                if time.time() - start_time > timeout:
                                    raise TimeoutError("Algorithm timed out")
                                result = algo_dict[selected_algo](start_state)
                                solution, representative_state, states_explored = result
                                start_or_belief = representative_state
                                fixed_belief_state = None
                                is_belief = False
                                comparison_data[selected_algo] = {
                                    'runtime': time.time() - start_time,
                                    'states': states_explored
                                }

                            algorithm_runtime = time.time() - start_time
                            add_log(f"{selected_algo} completed in {algorithm_runtime:.2f}s")
                            pg.event.pump()
                            pg.display.flip()
                        except Exception as e:
                            algorithm_runtime = time.time() - start_time
                            add_log(f"Error in {selected_algo}: {str(e)}")
                            print(f"Error in {selected_algo}: {str(e)}")
                            start_or_belief = start_state
                            fixed_belief_state = None
                            is_belief = False
                            solving = False
                            selected_algo = None
                            pg.event.pump()
                            pg.display.flip()

        if solving:
            if display_states and solution_index < len(display_states):
                if current_time - last_update >= update_interval:
                    current_state, _, msg = display_states[solution_index]
                    add_log(msg)
                    solution_index += 1
                    last_update = current_time
                    if solution_index == len(display_states):
                        solving = False

            elif solution and solution_index < len(solution):
                if current_time - last_update >= update_interval:
                    current_state = [row[:] for row in solution[solution_index]]
                    if solution_index > 0:
                        move = get_move_description(solution[solution_index - 1], solution[solution_index])
                        add_log(f"Step {solution_index}: {move}")
                    else:
                        add_log("Initial state")
                    solution_index += 1
                    last_update = current_time
                    if solution_index == len(solution):
                        add_log("Completed!")
                        solving = False

        if current_state != last_drawn_state or not solving:
            ve_khung(start_or_belief, trang_thai_dich, current_state, is_belief)
            menu.draw(game_screen)
            compare_button.draw(game_screen)
            pg.display.update()
            last_drawn_state = [row[:] for row in current_state]

        clock.tick(60)

    pg.quit()

def generate_initial_belief_state():
    belief = set()
    base_states = [
        [[1, 2, 3], [4, 5, 6], [7, 0, 8]],
        [[1, 2, 3], [0, 4, 6], [7, 5, 8]],
        [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
    ]
    for state in base_states:
        if is_valid_state(state):
            belief.add(state_to_string(state))
    return belief

def generate_initial_belief_state_partially():
    belief = set()
    base_states = [
        [[1, 2, 3], [4, 5, 6], [7, 0, 8]],
        [[1, 2, 3], [0, 4, 6], [7, 5, 8]],
        [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
    ]
    for state in base_states:
        if state[0][0] == 1 and is_valid_state(state):
            belief.add(state_to_string(state))
    return belief

if __name__ == "__main__":
    main()