import heapq
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pickle
import os


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if isinstance(other, Point):
            return np.isclose(self.x, other.x) and np.isclose(self.y, other.y)
        return False

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

    def __gt__(self, other):
        return (self.x, self.y) > (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))


path_cache = {}


def heuristic(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y)


def get_neighbors(board, point):
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nx, ny = int(point.x + dx), int(point.y + dy)
        if 0 <= nx < len(board) and 0 <= ny < len(board[0]) and np.isclose(board[nx][ny], 0):
            neighbors.append(Point(nx, ny))
    return neighbors


def a_star_search(board, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for neighbor in get_neighbors(board, current):
            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(goal, neighbor)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    if goal not in came_from:
        return None

    current, path = goal, []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()

    return path


def a_star_search_modified(board, start, goal):
    # Check if the path is in the cache
    if (start, goal) in path_cache:
        return path_cache[(start, goal)]

    # Check if the path is in the file
    if os.path.exists("TEST.pkl"):
        with open("TEST.pkl", "rb") as file:
            file_cache = pickle.load(file)
            if (start, goal) in file_cache:
                return file_cache[(start, goal)]

    # Otherwise, compute the path and store it in the cache
    path = a_star_search(board, start, goal)
    path_cache[(start, goal)] = path

    # Also save it to the file
    with open("TEST.pkl", "wb") as file:
        pickle.dump(path_cache, file)

    return path


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setGeometry(600, 300, 800, 800)
        self.setWindowTitle("Трассировка печатной платы")
        self.board = np.zeros((100, 100))

        self.plotWidget = pg.PlotWidget()
        self.setCentralWidget(self.plotWidget)

        self.plotWidget.setYRange(0, 100)
        self.plotWidget.setXRange(0, 100)

        self.points = [
            (Point(3, 44), Point(8, 45)),
            (Point(3, 40), Point(3, 32)),
            (Point(3, 32), Point(9, 30)),
            (Point(3, 26.8), Point(11, 19)),
            (Point(3, 20), Point(2, 10)),
            (Point(5, 10), Point(5, 4)),
            (Point(3, 20), Point(16.5, 20)),
            (Point(3, 14.8), Point(12, 19)),
            (Point(2, 10), Point(11, 14)),
            (Point(8, 45), Point(11, 45)),
            (Point(8, 38), Point(14, 45)),
            (Point(11, 45), Point(12, 19)),
            (Point(11, 38), Point(11, 19)),
            (Point(14, 45), Point(24, 45)),
            (Point(12, 19), Point(16, 15)),
            (Point(11, 19), Point(16.5, 20)),
            (Point(11, 19), Point(11, 10)),
            (Point(11, 10), Point(17, 10)),
            (Point(18, 29.5), Point(16, 17)),
            (Point(16, 17), Point(15, 10)),
            (Point(18, 20), Point(11, 19)),
            (Point(9, 30), Point(18, 29.5)),
            (Point(18, 29.5), Point(20, 29.5)),
            (Point(20, 29.5), Point(27, 29)),
            (Point(20, 29.5), Point(33, 18)),
            (Point(14, 38), Point(33, 32)),
            (Point(22, 45), Point(24, 38)),
            (Point(24, 45), Point(27, 33)),
            (Point(27, 33), Point(25, 19)),
            (Point(24, 45), Point(35, 44)),
            (Point(27, 45), Point(30, 45)),
            (Point(16, 38), Point(22, 38)),
            (Point(22, 38), Point(22, 14)),
            (Point(19, 38), Point(19, 33)),
            (Point(24, 38), Point(13, 30)),
            (Point(18, 20), Point(28, 19)),
            (Point(22, 10), Point(33, 32)),
            (Point(33, 45), Point(33, 30)),
            (Point(27, 38), Point(33, 38)),
            (Point(33, 38), Point(35, 40)),
            (Point(30, 38), Point(33, 28)),
            (Point(33, 28), Point(33, 21)),
        ]

        for point1, point2 in self.points:
            self.plotWidget.plot([point1.x, point2.x], [point1.y, point2.y], pen=None, symbol='o')
            path = a_star_search_modified(self.board, point1, point2)
            if path is not None:
                self.plotWidget.plot([point.x for point in path], [point.y for point in path], pen='r')

        self.pointsW = [
            (Point(3, 20), Point(16.5, 20)),
        ]

        for point1, point2 in self.pointsW:
            self.plotWidget.plot([point1.x, point2.x], [point1.y, point2.y], pen='r')

        self.pointsSolo = [
            Point(11, 12),
            Point(16, 45),
            Point(19, 45),
        ]

        for point in self.pointsSolo:
            self.plotWidget.plot([point.x], [point.y], pen=None, symbol='o')





if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = Window()
    window.show()

    app.exec_()
