import heapq
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PIL import Image
from PyQt5.QtWidgets import QMessageBox
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
    dx = abs(a.x - b.x)
    dy = abs(a.y - b.y)
    return dx + dy

def get_neighbors(board_array, point):
    neighbors = []
    dx_values = [-1, 0, 1, 0]  # Смещения по оси x
    dy_values = [0, 1, 0, -1]  # Смещения по оси y

    for dx, dy in zip(dx_values, dy_values):
        nx, ny = int(point.x + dx), int(point.y + dy)
        if 0 <= nx < len(board_array) and 0 <= ny < len(board_array[0]) and np.isclose(board_array[nx][ny], 0).any():
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

    if (start, goal) in path_cache:
        return path_cache[(start, goal)]


    if os.path.exists("TEST.pkl"):
        with open("TEST.pkl", "rb") as file:
            file_cache = pickle.load(file)
            if (start, goal) in file_cache:
                return file_cache[(start, goal)]


    path = a_star_search(board, start, goal)
    path_cache[(start, goal)] = path


    with open("TEST.pkl", "wb") as file:
        pickle.dump(path_cache, file)

    return path


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setGeometry(600, 300, 800, 800)
        self.setWindowTitle("Трассировка печатной платы")
        self.board = QtGui.QPixmap(100, 100)
        self.board.fill(QtGui.QColor(255, 255, 255))

        layout = QtWidgets.QGridLayout()

        # Создание всплывающих списков
        self.project_combobox = QtWidgets.QComboBox(self)
        self.project_combobox.addItem("Проект")
        self.project_combobox.addItem("Создать")
        self.project_combobox.addItem("Открыть")
        self.project_combobox.addItem("Сохранить")

        self.project_data_combobox = QtWidgets.QComboBox(self)
        self.project_data_combobox.addItem("Проектные данные")
        self.project_data_combobox.addItem("Данные о печатной плате")

        self.execute_combobox = QtWidgets.QComboBox(self)
        self.execute_combobox.addItem("Выполнить")
        self.execute_combobox.addItem("Выполнение трассировки")

        # Добавление всплывающих списков в макет
        layout.addWidget(self.project_combobox, 0, 0)
        layout.addWidget(self.project_data_combobox, 0, 1)
        layout.addWidget(self.execute_combobox, 0, 2)

        self.stacked_layout = QtWidgets.QStackedLayout()
        layout.addLayout(self.stacked_layout, 1, 0, 1, 3)

        self.plotWidget = pg.PlotWidget()
        self.stacked_layout.addWidget(self.plotWidget)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

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

        self.uniqpoints = [
            Point(18, 29.5), Point(35, 44), Point(22, 12), Point(19, 33), Point(27, 29),
            Point(14, 45), Point(16.5, 20), Point(15, 10), Point(33, 18), Point(27, 45), Point(33, 32),
            Point(13, 30), Point(8, 38), Point(11, 10), Point(11, 45), Point(27, 33), Point(3, 44),
            Point(3, 32), Point(2, 10), Point(11, 19), Point(5, 4), Point(20, 29.5), Point(12, 19),
            Point(33, 38), Point(8, 45), Point(3, 14.8), Point(16, 17), Point(18, 20), Point(30, 38),
            Point(24, 38), Point(24, 45), Point(3, 20), Point(22, 38), Point(22, 45), Point(3, 26.8),
            Point(27, 38), Point(11, 38), Point(24, 45), Point(22, 10), Point(3, 40), Point(11, 12),
            Point(16, 45), Point(27, 33), Point(16, 38), Point(33, 45), Point(9, 30), Point(20, 29.5),
            Point(19, 45), Point(11, 19), Point(18, 29.5), Point(33, 28), Point(27, 45), Point(25, 19),
            Point(14, 38), Point(22, 38), Point(27, 45),
        ]

        self.pointsW = [
            (Point(3, 20), Point(16.5, 20)),
        ]

        self.pointsSolo = [
            Point(11, 12),
            Point(16, 45),
            Point(19, 45),
        ]

        self.all_points = [point for pair in self.points for point in pair] + \
                          [point for pair in self.pointsW for point in pair] + \
                          self.pointsSolo + self.uniqpoints

        self.project_combobox.activated.connect(self.handle_project_combobox)
        self.project_data_combobox.activated.connect(self.handle_project_data_combobox)
        self.execute_combobox.activated.connect(self.handle_execute_combobox)

    def handle_project_combobox(self, index):
        if index == 1: # Создать
            self.create_points()
        if index == 2:  # Открыть
            self.open_points()
        elif index == 3:  # Сохранить
            self.save_points()

    def handle_project_data_combobox(self, index):
        if index == 1:  # Данные о печатной плате
            board_size = f"Размер печатной платы: {self.board.width()} x {self.board.height()}"
            QtWidgets.QMessageBox.information(self, "Информация о печатной плате", board_size)

    def handle_execute_combobox(self, index):
        if index == 1:  # Выполнение трассировки
            self.execute_tracing()

    def create_points(self):
        for point in self.all_points:
            self.plotWidget.plot([point.x], [point.y], pen=None, symbol='o')

        self.save_points_to_file(self.all_points, "points.pkl")

    def open_points(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выбрать файл", "", "Pickle Files (*.pkl)")
        if filename:
            self.display_points_from_file(filename)

    def execute_tracing(self):
        self.plotWidget.setYRange(0, 46)

        board_image = Image.fromqpixmap(self.board)
        board_array = np.array(board_image)
        board_array = board_array.reshape(self.board.height(), self.board.width(), 3)

        colors = ['r', 'g', 'b', 'c', 'm', 'y']

        for i, (point1, point2) in enumerate(self.points):
            self.plotWidget.plot([point1.x, point2.x], [point1.y, point2.y], pen=None, symbol='o')
            path = a_star_search_modified(board_array, point1, point2)
            if path is not None:
                # Use modulo operator to cycle through the colors list
                color = colors[i % len(colors)]
                pen = pg.mkPen(color=color, width=2)  # Create a pen with the specified color and width
                self.plotWidget.plot([point.x for point in path], [point.y for point in path], pen=pen)

        for point in self.pointsSolo:
            self.plotWidget.plot([point.x], [point.y], pen=None, symbol='o')

        for point1, point2 in self.pointsW:
            self.plotWidget.plot([point1.x, point2.x], [point1.y, point2.y], pen='g')

    def save_points_to_file(self, points, filename):
        with open(filename, "wb") as file:
            pickle.dump(points, file)

    def display_points_from_file(self, filename):
        with open(filename, "rb") as file:
            points = pickle.load(file)

        self.plotWidget.clear()  # Clear the plot before displaying points

        self.stacked_layout.setCurrentWidget(self.plotWidget)

        for point1, point2 in self.points:
            self.plotWidget.plot([point1.x, point2.x], [point1.y, point2.y], pen=None, symbol='o')

        for point1, point2 in self.pointsW:
            self.plotWidget.plot([point1.x, point2.x], [point1.y, point2.y], pen='r')

        for point in self.pointsSolo:
            self.plotWidget.plot([point.x], [point.y], pen=None, symbol='o')




if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = Window()
    window.show()

    app.exec_()
