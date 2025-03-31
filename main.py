import pygame as pg
from pygame.locals import *
from collections import deque
import heapq
import random
import math

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
GAME_WIDTH = 600
GAME_HEIGHT = 700
ALGO_WIDTH = 300
ALGO_HEIGHT = 700
GAP = 20

# Tạo cửa sổ
game_screen = pg.display.set_mode((GAME_WIDTH + ALGO_WIDTH + GAP, max(GAME_HEIGHT, ALGO_HEIGHT)))
pg.display.set_caption("Giai cuu o trang voi 8-puzzle")

# Font
pg.font.init()
TITLE_FONT = pg.font.Font(None, 48)
BUTTON_FONT = pg.font.Font(None, 36)
LOG_FONT = pg.font.Font(None, 24)
TILE_FONT = pg.font.Font(None, 72)
SMALL_FONT = pg.font.Font(None, 24)

# Trạng thái ban đầu và đích
start_state = [[1, 2, 3], [4, 8, 5], [7, 0, 6]]  # Trạng thái ban đầu mới
trang_thai_dich = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

# Log messages
log_messages = []

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
    def __init__(self, x, y, width, height):
        self.rect = pg.Rect(x, y, width, height)
        self.buttons = []
        algorithms = ["BFS", "DFS", "UCS", "IDDFS", "GBFS", "A*", "IDA*",
                     "Simple HC", "Steepest HC", "Stochastic HC", "Simulated Annealing",
                     "Beam Stochastic HC", "Genetic Algorithm"]
        btn_width = width - 50
        btn_height = 30
        for i, algo in enumerate(algorithms):
            btn = Button(x + 30, y + 40 + i * (btn_height + 5),
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


def ve_khung(start_state, goal_state, current_state):
    game_screen.fill(WHITE)

    # Vẽ trạng thái hiện tại (lớn, trung tâm giao diện chính)
    current_rect = pg.Rect(0, 0, GAME_WIDTH, GAME_HEIGHT)
    shadow_rect = current_rect.move(3, 3)
    pg.draw.rect(game_screen, SHADOW, shadow_rect, border_radius=12)
    pg.draw.rect(game_screen, LIGHT_GRAY, current_rect, border_radius=12)

    tile_size = 150
    margin = 20
    grid_x = (GAME_WIDTH - 3 * tile_size - 2 * margin) // 2
    grid_y = (GAME_HEIGHT - 3 * tile_size - 2 * margin) // 2

    current_title = SMALL_FONT.render("Giai cuu o trang 8-puzzle", True, BLACK)
    current_title_rect = current_title.get_rect(center=(GAME_WIDTH // 2, 20))
    game_screen.blit(current_title, current_title_rect)

    for i in range(3):
        for j in range(3):
            x = grid_x + j * (tile_size + margin)
            y = grid_y + i * (tile_size + margin) + 100
            tile = current_state[i][j]
            tile_rect = pg.Rect(x, y, tile_size, tile_size)
            shadow_rect = tile_rect.move(2, 2)
            pg.draw.rect(game_screen, SHADOW, shadow_rect, border_radius=8)
            color = WHITE if tile != 0 else GRAY
            pg.draw.rect(game_screen, color, tile_rect, border_radius=8)
            pg.draw.rect(game_screen, GREEN, tile_rect, 1, 8)
            if tile != 0:
                text = TILE_FONT.render(str(tile), True, BLACK)
                text_rect = text.get_rect(center=tile_rect.center)
                game_screen.blit(text, text_rect)

    # Vẽ trạng thái ban đầu (nhỏ, góc trên trái, cố định)
    small_tile_size = 40
    small_margin = 10
    start_x = 0
    start_y = 40

    start_title = SMALL_FONT.render("Start", True, BLACK)
    start_title_rect = start_title.get_rect(
        center=(start_x + (3 * small_tile_size + 2 * small_margin) // 2, start_y - 20))
    game_screen.blit(start_title, start_title_rect)

    for i in range(3):
        for j in range(3):
            x = start_x + j * (small_tile_size + small_margin)
            y = start_y + i * (small_tile_size + small_margin)
            tile = start_state[i][j]
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

    # Vẽ trạng thái đích (nhỏ, cùng y nhưng dịch sang phải 200 thay vì 400)
    goal_x = start_x + 420
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


def move(state, direction):
    new_state = [row[:] for row in state]
    i, j = get_zero_pos(new_state)
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
    return tuple(tuple(row) for row in state)


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

def print_state(state):
    for row in state:
        print(row)
    print()


def print_solution(algo_name, path):
    print(f"{algo_name} Solution:")
    for i, state in enumerate(path):
        print(f"Step {i}:")
        print_state(state)


# Các thuật toán
def bfs(start_state):
    queue = deque([(start_state, [])])
    visited = set()
    visited.add(state_to_string(start_state))
    while queue:
        current_state, path = queue.popleft()
        if current_state == trang_thai_dich:
            print_solution("BFS", path + [current_state])
            return path + [current_state]
        for new_state in tao_buoc_moi(current_state):
            if state_to_string(new_state) not in visited:
                visited.add(state_to_string(new_state))
                queue.append((new_state, path + [current_state]))
    return None


def dfs(start_state, max_depth=50):
    stack = [(start_state, [])]
    visited = set()
    visited.add(state_to_string(start_state))
    while stack:
        current_state, path = stack.pop()
        if current_state == trang_thai_dich:
            print_solution("DFS", path + [current_state])
            return path + [current_state]
        if len(path) < max_depth:
            for new_state in reversed(tao_buoc_moi(current_state)):
                if state_to_string(new_state) not in visited:
                    visited.add(state_to_string(new_state))
                    stack.append((new_state, path + [current_state]))
    return None


def ucs(start_state):
    pq = [(0, start_state, [])]
    visited = set()
    visited.add(state_to_string(start_state))
    while pq:
        cost, current_state, path = heapq.heappop(pq)
        if current_state == trang_thai_dich:
            print_solution("UCS", path + [current_state])
            return path + [current_state]
        for new_state in tao_buoc_moi(current_state):
            if state_to_string(new_state) not in visited:
                visited.add(state_to_string(new_state))
                heapq.heappush(pq, (cost + 1, new_state, path + [current_state]))
    return None


def iddfs(start_state, max_depth=50):
    for depth in range(max_depth):
        stack = [(start_state, [])]
        visited = set()
        visited.add(state_to_string(start_state))
        while stack:
            current_state, path = stack.pop()
            if current_state == trang_thai_dich:
                print_solution("IDDFS", path + [current_state])
                return path + [current_state]
            if len(path) < depth:
                for new_state in reversed(tao_buoc_moi(current_state)):
                    if state_to_string(new_state) not in visited:
                        visited.add(state_to_string(new_state))
                        stack.append((new_state, path + [current_state]))
    return None


def gbfs(start_state):
    pq = [(heuristic(start_state), start_state, [])]
    visited = set()
    visited.add(state_to_string(start_state))
    while pq:
        _, current_state, path = heapq.heappop(pq)
        if current_state == trang_thai_dich:
            print_solution("GBFS", path + [current_state])
            return path + [current_state]
        for new_state in tao_buoc_moi(current_state):
            if state_to_string(new_state) not in visited:
                visited.add(state_to_string(new_state))
                heapq.heappush(pq, (heuristic(new_state), new_state, path + [current_state]))
    return None


def astar(start_state):
    pq = [(heuristic(start_state), 0, start_state, [])]
    visited = set()
    visited.add(state_to_string(start_state))
    while pq:
        f, g, current_state, path = heapq.heappop(pq)
        if current_state == trang_thai_dich:
            print_solution("A*", path + [current_state])
            return path + [current_state]
        for new_state in tao_buoc_moi(current_state):
            if state_to_string(new_state) not in visited:
                visited.add(state_to_string(new_state))
                new_g = g + 1
                new_f = new_g + heuristic(new_state)
                heapq.heappush(pq, (new_f, new_g, new_state, path + [current_state]))
    return None


def ida_star(start_state):
    def search(path, g, bound):
        current_state = path[-1]
        f = g + heuristic(current_state)
        if f > bound:
            return f
        if current_state == trang_thai_dich:
            print_solution("IDA*", path)
            return path
        min_cost = float('inf')
        for new_state in tao_buoc_moi(current_state):
            if new_state not in [state_to_string(s) for s in path]:
                path.append(new_state)
                result = search(path, g + 1, bound)
                if isinstance(result, list):
                    return result
                if result < min_cost:
                    min_cost = result
                path.pop()
        return min_cost

    bound = heuristic(start_state)
    path = [start_state]
    while True:
        result = search(path, 0, bound)
        if isinstance(result, list):
            return result
        if result == float('inf'):
            return None
        bound = result


def simple_hill_climbing(start_state):
    current_state = [row[:] for row in start_state]
    path = [current_state]
    visited = set()
    visited.add(state_to_string(current_state))

    while current_state != trang_thai_dich:
        neighbors = tao_buoc_moi(current_state)
        current_h = heuristic(current_state)
        next_state = None

        # Chọn neighbor đầu tiên cải thiện heuristic
        for neighbor in neighbors:
            if state_to_string(neighbor) not in visited:
                if heuristic(neighbor) < current_h:
                    next_state = neighbor
                    break

        if next_state is None:  # Không tìm thấy neighbor nào tốt hơn
            print("Simple Hill Climbing: No solution found (local minimum reached)")
            return None

        current_state = [row[:] for row in next_state]
        visited.add(state_to_string(current_state))
        path.append(current_state)

    print_solution("Simple Hill Climbing", path)
    return path


def steepest_ascent_hill_climbing(start_state):
    current_state = [row[:] for row in start_state]
    path = [current_state]
    visited = set()
    visited.add(state_to_string(current_state))

    while current_state != trang_thai_dich:
        neighbors = tao_buoc_moi(current_state)
        current_h = heuristic(current_state)
        best_state = None
        best_h = float('inf')

        # Tìm neighbor tốt nhất (heuristic nhỏ nhất)
        for neighbor in neighbors:
            if state_to_string(neighbor) not in visited:
                h = heuristic(neighbor)
                if h < best_h:
                    best_h = h
                    best_state = neighbor

        if best_state is None or best_h >= current_h:  #Không tìm thấy neighbor nào tốt hơn
            print("Steepest-Ascent Hill Climbing: No solution found (local minimum reached)")
            return None

        current_state = [row[:] for row in best_state]
        visited.add(state_to_string(current_state))
        path.append(current_state)

    print_solution("Steepest-Ascent Hill Climbing", path)
    return path


def stochastic_hill_climbing(start_state):
    current_state = [row[:] for row in start_state]
    path = [current_state]
    visited = set()
    visited.add(state_to_string(current_state))

    while current_state != trang_thai_dich:
        neighbors = tao_buoc_moi(current_state)
        current_h = heuristic(current_state)
        better_neighbors = []

        # Tìm tất cả neighbor tốt hơn
        for neighbor in neighbors:
            if state_to_string(neighbor) not in visited:
                h = heuristic(neighbor)
                if h < current_h:
                    better_neighbors.append(neighbor)

        if not better_neighbors:  #Không tìm thấy neighbor nào tốt hơn
            print("Stochastic Hill Climbing: No solution found (local minimum reached)")
            return None

        #Chọn ngẫu nhiên một neighbor từ các neighbor tốt hơn
        next_state = random.choice(better_neighbors)
        current_state = [row[:] for row in next_state]
        visited.add(state_to_string(current_state))
        path.append(current_state)

    print_solution("Stochastic Hill Climbing", path)
    return path

def simulated_annealing(start_state, initial_temp=2000, cooling_rate=0.99, min_temp=0.1, max_iterations=5000):
    current_state = [row[:] for row in start_state]
    best_state = [row[:] for row in start_state]
    path = [current_state]  # Chỉ lưu các bước cải thiện hoặc đích
    temp = initial_temp
    iterations = 0
    best_h = heuristic(best_state)
    stagnation_count = 0  # Đếm số bước không cải thiện

    while temp > min_temp and iterations < max_iterations:
        if current_state == trang_thai_dich:
            break

        neighbors = tao_buoc_moi(current_state)
        if not neighbors:
            break

        # Chọn neighbor tốt nhất trong số các neighbor, nhưng vẫn có xác suất ngẫu nhiên
        current_h = heuristic(current_state)
        best_neighbor = min(neighbors, key=heuristic, default=current_state)
        next_state = best_neighbor if random.random() < 0.7 else random.choice(neighbors)
        next_h = heuristic(next_state)
        delta_h = next_h - current_h

        # Chấp nhận neighbor nếu tốt hơn hoặc theo xác suất (giới hạn bước xấu)
        if delta_h <= 0 or (random.random() < math.exp(-delta_h / temp) and stagnation_count < 10):
            current_state = [row[:] for row in next_state]
            # Chỉ thêm vào path nếu cải thiện hoặc đạt đích
            if next_h < best_h or next_state == trang_thai_dich:
                best_h = next_h
                best_state = [row[:] for row in current_state]
                path.append(current_state)
                stagnation_count = 0  # Reset khi cải thiện
            else:
                stagnation_count += 1  # Tăng khi không cải thiện

        temp *= cooling_rate
        iterations += 1

    # Kiểm tra trạng thái tốt nhất
    if best_state == trang_thai_dich or current_state == trang_thai_dich:
        print_solution("Simulated Annealing", path)
        return path
    else:
        print(f"Simulated Annealing: No solution found")
        return None


def beam_stochastic_hill_climbing(start_state, beam_width=3, max_iterations=1000):
    # Khởi tạo beam với trạng thái ban đầu
    current_beam = [(heuristic(start_state), [row[:] for row in start_state], [])]
    visited = set()
    visited.add(state_to_string(start_state))
    iterations = 0

    while iterations < max_iterations:
        # Kiểm tra nếu bất kỳ trạng thái nào trong beam là đích
        for _, state, path in current_beam:
            if state == trang_thai_dich:
                print_solution("Beam Stochastic HC", path + [state])
                return path + [state]

        # Tạo tập hợp các neighbor từ tất cả trạng thái trong beam
        next_candidates = []
        for _, current_state, path in current_beam:
            current_h = heuristic(current_state)
            neighbors = tao_buoc_moi(current_state)
            for neighbor in neighbors:
                neighbor_str = state_to_string(neighbor)
                if neighbor_str not in visited:
                    h = heuristic(neighbor)
                    if h < current_h:  # Chỉ giữ các neighbor cải thiện heuristic
                        next_candidates.append((h, neighbor, path + [current_state]))

        if not next_candidates:  # Không tìm thấy neighbor nào tốt hơn
            print("Beam Stochastic HC: No solution found (local minimum reached)")
            return None

        # Sắp xếp theo heuristic và chọn ngẫu nhiên trong số beam_width trạng thái tốt nhất
        next_candidates.sort(key=lambda x: x[0])  # Sắp xếp theo heuristic tăng dần
        top_candidates = next_candidates[:min(beam_width, len(next_candidates))]
        current_beam = []
        visited_count = 0

        # Chọn ngẫu nhiên từ top candidates để tạo beam mới
        while len(current_beam) < beam_width and top_candidates and visited_count < len(next_candidates):
            selected = random.choice(top_candidates)
            selected_str = state_to_string(selected[1])
            if selected_str not in visited:
                visited.add(selected_str)
                current_beam.append(selected)
                top_candidates.remove(selected)
            else:
                visited_count += 1  # Đếm số trạng thái đã thăm để tránh vòng lặp vô hạn

        if not current_beam:  # Không còn trạng thái mới để khám phá
            print("Beam Stochastic HC: No solution found (all top states visited)")
            return None

        iterations += 1

    print(f"Beam Stochastic HC: No solution found after {max_iterations} iterations")
    return None

def flatten_state(state):
    """Chuyển ma trận 3x3 thành danh sách phẳng."""
    return [state[i][j] for i in range(3) for j in range(3)]


def unflatten_state(flat_state):
    """Chuyển danh sách phẳng thành ma trận 3x3."""
    return [[flat_state[i * 3 + j] for j in range(3)] for i in range(3)]


def fitness(state):
    """Hàm đánh giá: Nghịch đảo của heuristic (càng gần đích, fitness càng cao)."""
    h = heuristic(state)
    return 1 / (h + 1) if h > 0 else float('inf')


def generate_random_state():
    """Tạo một trạng thái ngẫu nhiên hợp lệ cho 8-puzzle."""
    numbers = list(range(9))
    random.shuffle(numbers)
    return unflatten_state(numbers)


def crossover(parent1, parent2):
    """Lai ghép hai trạng thái cha mẹ."""
    flat_p1 = flatten_state(parent1)
    flat_p2 = flatten_state(parent2)
    point = random.randint(1, 7)  # Điểm cắt giữa 1 và 7
    child = flat_p1[:point] + flat_p2[point:]

    # Sửa các giá trị trùng lặp
    used = set(child)
    missing = [x for x in range(9) if x not in used]
    for i in range(len(child)):
        if child.count(child[i]) > 1:
            child[i] = missing.pop(0)

    return unflatten_state(child)


def mutate(state, mutation_rate=0.1):
    """Đột biến: Hoán đổi ngẫu nhiên hai ô nếu xác suất đột biến được kích hoạt."""
    if random.random() < mutation_rate:
        flat_state = flatten_state(state)
        i, j = random.sample(range(9), 2)
        flat_state[i], flat_state[j] = flat_state[j], flat_state[i]
        return unflatten_state(flat_state)
    return [row[:] for row in state]


def genetic_algorithm(start_state, population_size=100, generations=500, mutation_rate=0.1):
    # Khởi tạo quần thể ban đầu
    population = [generate_random_state() for _ in range(population_size - 1)]
    population.append([row[:] for row in start_state])  # Thêm trạng thái bắt đầu

    for generation in range(generations):
        # Đánh giá fitness cho quần thể
        fitness_scores = [(fitness(ind), ind) for ind in population]
        fitness_scores.sort(reverse=True)  # Sắp xếp giảm dần theo fitness

        # Kiểm tra nếu đã tìm thấy đích
        if fitness_scores[0][0] == float('inf'):
            best_state = fitness_scores[0][1]
            # Tạo đường dẫn giả từ start_state đến best_state (GA không tạo path trực tiếp)
            path = [start_state, best_state]
            print_solution("Genetic Algorithm", path)
            return path

        # Chọn lọc: Giữ 50% cá thể tốt nhất
        elite_size = population_size // 2
        new_population = [ind for _, ind in fitness_scores[:elite_size]]

        # Lai ghép và đột biến để tạo đủ quần thể
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(new_population[:elite_size], 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

    print("Genetic Algorithm: No solution found!")
    return None

def get_move_description(prev_state, curr_state):
    prev_zero = get_zero_pos(prev_state)
    curr_zero = get_zero_pos(curr_state)
    if prev_zero[0] > curr_zero[0]:
        return "Up"
    elif prev_zero[0] < curr_zero[0]:
        return "Down"
    elif prev_zero[1] > curr_zero[1]:
        return "Left"
    elif prev_zero[1] < curr_zero[1]:
        return "Right"
    return ""


def main():
    clock = pg.time.Clock()
    running = True
    menu = AlgorithmMenu(GAME_WIDTH + GAP, 0, ALGO_WIDTH, ALGO_HEIGHT)
    current_state = [row[:] for row in start_state]
    solution = None
    solution_index = 0
    last_update = 0
    update_interval = 500
    solving = False

    while running:
        current_time = pg.time.get_ticks()

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif not solving:
                selected_algo = menu.handle_event(event)
                if selected_algo:
                    algo_dict = {
                        "BFS": bfs, "DFS": dfs, "UCS": ucs, "IDDFS": iddfs,
                        "GBFS": gbfs, "A*": astar, "IDA*": ida_star,
                        "Simple HC": simple_hill_climbing,
                        "Steepest HC": steepest_ascent_hill_climbing,
                        "Stochastic HC": stochastic_hill_climbing,
                        "Simulated Annealing": simulated_annealing,
                        "Beam Stochastic HC": beam_stochastic_hill_climbing,
                        "Genetic Algorithm": genetic_algorithm
                    }
                    if selected_algo in algo_dict:
                        current_state = [row[:] for row in start_state]
                        log_messages.clear()
                        solution = None
                        solution_index = 0
                        solving = True
                        add_log(f"Running {selected_algo}...")
                        print("Initial state:")
                        print_state(current_state)
                        solution = algo_dict[selected_algo](current_state)
                        last_update = current_time
                        if solution:
                            add_log(f"Solution found in {len(solution) - 1} steps!")
                        else:
                            add_log("No solution found!")
                            solving = False

        # Cập nhật giao diện
        ve_khung(start_state, trang_thai_dich, current_state)
        menu.draw(game_screen)

        # Hiển thị các bước giải
        if solving and solution and solution_index < len(solution):
            if current_time - last_update >= update_interval:
                if solution_index > 0:
                    move_desc = get_move_description(solution[solution_index - 1], solution[solution_index])
                    add_log(f"Step {solution_index}: Move {move_desc}")
                else:
                    add_log("Step 0: Initial state")
                current_state = [row[:] for row in solution[solution_index]]
                solution_index += 1
                last_update = current_time
                if solution_index == len(solution):
                    add_log("Completed!")
                    solving = False

        pg.display.update()
        clock.tick(60)

    pg.quit()


if __name__ == "__main__":
    main()