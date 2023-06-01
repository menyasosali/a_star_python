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
        self.setWindowTitle("Test")
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
            (Point(27, 45), Point(26, 12.5)),
            (Point(22, 12), Point(26, 12.5)),
            (Point(16, 38), Point(22, 38)),
            (Point(22, 38), Point(22, 14)),
            (Point(19, 38), Point(19, 33)),
            (Point(24, 38), Point(13, 30)),
            (Point(18, 20), Point(28, 19)),
            (Point(27, 33), Point(27, 12.5)),
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
            (Point(22, 12), Point(26, 12.5)),
            (Point(27, 45), Point(26, 12.5)),
            (Point(27, 33), Point(27, 12.5)),
            (Point(3, 20), Point(16.5, 20)),
            (Point(11, 12), Point(11, 12)),
            (Point(16, 45), Point(16, 45)),
            (Point(19, 45), Point(19, 45)),
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

        delete_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self)
        delete_shortcut.activated.connect(self.delete_selected_shapes)

    def delete_selected_shapes(self):
        selected_shapes = []

        for shape in self.shapes:
            if shape.shape_item.isSelected():
                selected_shapes.append(shape)

        for shape in selected_shapes:
            self.plotWidget.removeItem(shape.shape_item)
            self.shapes.remove(shape)

    def save_shapes_and_close_event(self, event):
        self.save_shapes_to_file()
        event.accept()

    def save_shapes_to_file(self):
        shapes_data = []

        for shape in self.shapes:
            shape_item = shape.shape_item

            if isinstance(shape_item, pg.EllipseROI):
                shape_type = 'circle'
                shape_params = {
                    'center': shape_item.pos() + shape_item.size() / 2,
                    'radius': shape_item.size()[0] / 2
                }

            elif isinstance(shape_item, pg.RectROI):
                shape_type = 'rectangle'
                shape_params = {
                    'center': shape_item.pos() + shape_item.size() / 2,
                    'width': shape_item.size()[0],
                    'height': shape_item.size()[1]
                }

            shapes_data.append((shape_type, shape_params))

        with open("shapes_data.pkl", "wb") as file:
            pickle.dump(shapes_data, file)

        positions_data = []

        for shape in self.shapes:
            shape_item = shape.shape_item
            position = shape_item.pos()

            positions_data.append((shape_item, position))

        with open("positions_data.pkl", "wb") as file:
            pickle.dump(positions_data, file)

    def load_shapes_from_file(self):
        if os.path.exists("shapes_data.pkl"):
            with open("shapes_data.pkl", "rb") as file:
                shape_data = pickle.load(file)
                if shape_data is not None:
                    for shape_type, shape_params in shape_data:
                        if shape_type == 'circle':
                            shape_item = pg.EllipseROI(
                                pos=(shape_params['center'][0] - shape_params['radius'],
                                     shape_params['center'][1] - shape_params['radius']),
                                size=(2 * shape_params['radius'], 2 * shape_params['radius']),
                                movable=True,
                                rotatable=False,
                                resizable=False,
                                pen=pg.mkPen('y', width=2)
                            )

                        elif shape_type == 'rectangle':
                            shape_item = pg.RectROI(
                                pos=(shape_params['center'][0] - shape_params['width'] / 2,
                                     shape_params['center'][1] - shape_params['height'] / 2),
                                size=(shape_params['width'], shape_params['height']),
                                movable=True,
                                rotatable=False,
                                resizable=False,
                                pen=pg.mkPen('y', width=2)
                            )

                        else:
                            continue

                        shape = MovableShape(shape_item, shape_type, shape_params)
                        shape_item.sigRegionChangeFinished.connect(shape.move_shape)
                        self.shapes.append(shape)
                        self.plotWidget.addItem(shape_item)
                        shape_item.setPos(shape_params['center'])


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = Window()
    window.show()

    app.exec_()
