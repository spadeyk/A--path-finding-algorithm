import pygame
import math
from queue import PriorityQueue
import os
pygame.init()
# Set the screen width back to 600
WIDTH = 600
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding algorithm")

# Load the PNG image from the correct directory
icon_path = r"C:\Users\pc world\OneDrive\Desktop\Udemy python course\Data structures and algorithms\A star path finding algorithm\icon.png"  # Ensure this path matches your file

# Check if the file exists before trying to load it
if not os.path.isfile(icon_path):
    raise FileNotFoundError(f"File not found: {icon_path}")

icon = pygame.image.load(icon_path)

# Set the icon for the window
pygame.display.set_icon(icon)

# Define colors (start and end nodes have new colors)
RED = (255, 0, 0)       # Closed nodes
GREEN = (0, 255, 0)     # Open nodes
BLUE = (0, 0, 255)      # Path nodes
WHITE = (255, 255, 255) # Grid background
BLACK = (0, 0, 0)       # Obstacles (walls)
PINK = (255, 20, 147)   # Start node
CYAN = (0, 255, 255)    # End node
GREY = (128, 128, 128)  # Grid lines

# Class representing each cell (node) in the grid
class Cell:
    def __init__(self, row, col, size, total_rows):
        self.row = row
        self.col = col
        self.x = row * size
        self.y = col * size
        self.color = WHITE
        self.neighbors = []
        self.size = size
        self.total_rows = total_rows

    def get_position(self):
        return self.row, self.col

    def is_blocked(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == PINK

    def is_goal(self):
        return self.color == CYAN

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = PINK

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_goal(self):
        self.color = CYAN

    def make_path(self):
        self.color = BLUE

    # Keep the original drawing functionality
    def draw(self, window):
        """Draws a square representing the cell on the grid."""
        pygame.draw.rect(window, self.color, (self.x, self.y, self.size, self.size))

    # Update neighbors using a simplified list of directions
    def update_neighbors(self, grid):
        """Updates the list of valid neighboring cells (not barriers)."""
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_blocked():  # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])
        if self.row > 0 and not grid[self.row - 1][self.col].is_blocked():  # UP
            self.neighbors.append(grid[self.row - 1][self.col])
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_blocked():  # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])
        if self.col > 0 and not grid[self.row][self.col - 1].is_blocked():  # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False

# Heuristic function for calculating Manhattan distance
def estimate_cost(p1, p2):
    """Manhattan distance heuristic between two points p1 and p2."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# Build the path from end to start by retracing
def build_path(came_from, current, draw):
    """Rebuilds the path from the goal to the start by following `came_from`."""
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()

# A* algorithm using a priority queue
def find_path(draw, grid, start, goal):
    """Executes the A* algorithm to find the shortest path from start to goal."""
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}

    g_score = {cell: float('inf') for row in grid for cell in row}
    f_score = {cell: float('inf') for row in grid for cell in row}

    g_score[start] = 0
    f_score[start] = estimate_cost(start.get_position(), goal.get_position())

    open_set_hash = {start}

    while not open_set.empty():
        current = open_set.get()[1]
        open_set_hash.remove(current)

        if current == goal:
            build_path(came_from, goal, draw)
            goal.make_goal()
            return True

        current.update_neighbors(grid)
        for neighbor in current.neighbors:
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + estimate_cost(neighbor.get_position(), goal.get_position())
                if neighbor not in open_set_hash:
                    open_set.put((f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()
        if current != start:
            current.make_closed()

    return False  # Path not found

# Create the grid
def create_grid(rows, width):
    """Generates a grid of cells using a 2D list."""
    size = width // rows
    grid = []
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            cell = Cell(i, j, size, rows)
            grid[i].append(cell)
    return grid

# Draw grid lines (same as the tutorial)
def draw_gridlines(window, rows, width):
    """Draws the grid lines on the window."""
    size = width // rows
    for i in range(rows):
        pygame.draw.line(window, GREY, (0, i * size), (width, i * size))
        pygame.draw.line(window, GREY, (i * size, 0), (i * size, width))

# Draw all the elements (grid and cells)
def render(window, grid, rows, width):
    """Fills the window and renders the grid and its lines."""
    window.fill(WHITE)
    for row in grid:
        for cell in row:
            cell.draw(window)
    draw_gridlines(window, rows, width)
    pygame.display.update()

# Get the cell that was clicked
def get_clicked_cell(pos, rows, width):
    """Converts a clicked position into a row and column in the grid."""
    size = width // rows
    y, x = pos
    row = y // size
    col = x // size
    return row, col

# Main loop
def main(window, width):
    ROWS = 50  # Define the number of rows
    grid = create_grid(ROWS, width)

    start = None
    goal = None

    running = True
    while running:
        render(window, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if pygame.mouse.get_pressed()[0]:  # Left-click
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_cell(pos, ROWS, width)
                cell = grid[row][col]
                if not start and cell != goal:
                    start = cell
                    start.make_start()

                elif not goal and cell != start:
                    goal = cell
                    goal.make_goal()

                elif cell != goal and cell != start:
                    cell.make_barrier()

            elif pygame.mouse.get_pressed()[2]:  # Right-click
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_cell(pos, ROWS, width)
                cell = grid[row][col]
                cell.reset()
                if cell == start:
                    start = None
                elif cell == goal:
                    goal = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and goal:
                    for row in grid:
                        for cell in row:
                            cell.update_neighbors(grid)

                    find_path(lambda: render(window, grid, ROWS, width), grid, start, goal)

                if event.key == pygame.K_c:  # Clear the grid
                    start = None
                    goal = None
                    grid = create_grid(ROWS, width)

    pygame.quit()

# Run the main loop
main(WIN, WIDTH)
