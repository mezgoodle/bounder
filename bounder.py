import random
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, QPushButton, QVBoxLayout,
    QLineEdit, QFileDialog, QGridLayout, QTableWidget, QTableWidgetItem,
    QSlider, QHBoxLayout
)
from PyQt5.QtCore import Qt
from typing import List, Set, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


class Point:
    def __init__(self, coordinates: List[float]):
        self.coordinates = coordinates

    def __eq__(self, other):
        return self.coordinates == other.coordinates

    def __hash__(self):
        return hash(tuple(self.coordinates))


def calculate_centre(points: Set[Point], key_point: Point, coordinates_count: int) -> Point:
    all_points = list(points) + [key_point]
    return Point([np.mean([point.coordinates[i] for point in all_points]) for i in range(coordinates_count)])


def calc_distance(point_a: Point, point_b: Point) -> float:
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(point_a.coordinates, point_b.coordinates)))


def calculate_point_offset(point: Point, centre: Point, mean_distance: float) -> float:
    return calc_distance(point, centre) / mean_distance


class ClusterBoundPointFinder:
    def __init__(self, input_vectors: List[Point], deviation: float, max_clusters=10):
        self.input_vectors = input_vectors
        self.deviation = deviation
        self.max_clusters = max_clusters
        self.coordinates_count = len(input_vectors[0].coordinates) if input_vectors else 0
        self.core_point_count = 5
        self.n_clusters, self.cluster_centers = self.find_cluster_centers()
        self.kmeans_clusters = None  # Додано для зберігання результату KMeans

    def calculate_bound_points(self):
        if self.kmeans_clusters is None:
            self.kmeans = KMeans(n_clusters=self.n_clusters)  # Зберігаємо kmeans як атрибут класу
            self.kmeans_clusters = self.kmeans.fit_predict(
                [[p.coordinates[i] for i in range(self.coordinates_count)] for p in self.input_vectors]
            )

        self.border_points = []
        for cluster_index in range(self.n_clusters):
            cluster_points = [
                point for i, point in enumerate(self.input_vectors) if self.kmeans_clusters[i] == cluster_index
            ]
            center = Point(self.kmeans.cluster_centers_[cluster_index].tolist())  # Використовуємо self.kmeans

            distances = {
                point: {
                    inner_point: calc_distance(point, inner_point)
                    for inner_point in cluster_points
                    if inner_point != point
                }
                for point in cluster_points
            }
            closest_distances = {
                point: dict(sorted(distances[point].items(), key=lambda item: item[1])[: self.core_point_count])
                for point in distances
            }
            mean_distances = {point: np.mean(list(dist.values())) for point, dist in closest_distances.items()}
            point_centres = {
                point: calculate_centre(set(closest.keys()), point, self.coordinates_count)
                for point, closest in closest_distances.items()
            }
            point_offsets = {
                point: calculate_point_offset(point, center, mean_distances[point])
                for point, centre in point_centres.items()
            }

            result = {point: offset for point, offset in point_offsets.items() if offset > self.deviation}
            self.border_points.append(result)
            print(f"  Cluster {cluster_index}: {len(result)} border points")

    def find_cluster_centers(self) -> Tuple[int, List[Point]]:
        """
        Знаходить центри кластерів за допомогою KMeans
        та визначає оптимальну кількість кластерів за методом "ліктя".
        """
        wcss = []
        for i in range(1, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit([[point.coordinates[0], point.coordinates[1]] for point in self.input_vectors])
            wcss.append(kmeans.inertia_)

        # Знаходимо оптимальну кількість кластерів
        # Використовуємо спрощений метод "ліктя"
        n_clusters = np.argmax(np.diff(wcss)) + 2

        # Повторно запускаємо KMeans з оптимальною кількістю кластерів
        # n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit([[p.coordinates[i] for i in range(self.coordinates_count)] for p in self.input_vectors])
        cluster_centers = [Point(center.tolist()) for center in kmeans.cluster_centers_]
        return n_clusters, cluster_centers


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Пошук граничних точок кластерів")
        self.dimension = 2  # За замовчуванням вибрано 2D
        self.input_vectors = []
        self.deviation = 0.15

        # Віджети для вибору розмірності
        self.dimension_label = QLabel("Оберіть розмірність простору ознак:")
        self.dimension_combobox = QComboBox(self)
        self.dimension_combobox.addItems(["2", "3"])
        self.dimension_button = QPushButton("Обрати", self)
        self.dimension_button.clicked.connect(self.on_dimension_select)

        # Віджети для введення даних
        self.input_label = QLabel("Введіть точки:")
        self.input_edit = QLineEdit(self)
        self.file_button = QPushButton("Обрати файл", self)
        self.file_button.clicked.connect(self.on_file_select)

        # Таблиця для відображення даних
        self.data_table = QTableWidget(self)
        self.data_table.setColumnCount(self.dimension)
        self.data_table.setHorizontalHeaderLabels([f"X{i + 1}" for i in range(self.dimension)])

        # Розміщення віджетів
        grid = QGridLayout()
        grid.addWidget(self.dimension_label, 0, 0)
        grid.addWidget(self.dimension_combobox, 0, 1)
        grid.addWidget(self.dimension_button, 0, 2)
        grid.addWidget(self.input_label, 1, 0)
        grid.addWidget(self.input_edit, 2, 0, 1, 3)
        grid.addWidget(self.file_button, 3, 0)
        grid.addWidget(self.data_table, 4, 0, 1, 3)  # Додаємо таблицю
        self.setLayout(grid)

        self.calculate_button = QPushButton("Розрахувати", self)
        self.calculate_button.clicked.connect(self.on_calculate_click)
        grid.addWidget(self.calculate_button, 5, 0)  # Додаємо кнопку

        self.add_button = QPushButton("Додати", self)
        self.add_button.clicked.connect(self.on_add_data_click)
        grid.addWidget(self.add_button, 3, 1)  # Додаємо кнопку "Додати"

        # Графік
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)  # Початково 2D графік

        # Повзунок
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(15)
        self.slider.setMaximum(100)
        self.slider.setValue(int(self.deviation * 100))
        self.slider.valueChanged.connect(self.on_slider_change)

        # Мітка для значення deviation
        self.deviation_label = QLabel(f"Deviation: {self.deviation:.2f}", self)

        # Розміщення віджетів
        hbox = QHBoxLayout()
        hbox.addWidget(self.canvas)
        vbox = QVBoxLayout()
        vbox.addWidget(self.slider)
        vbox.addWidget(self.deviation_label)
        hbox.addLayout(vbox)
        grid.addLayout(hbox, 6, 0, 1, 3)  # Додаємо графік та повзунок

    def on_dimension_select(self):
        self.dimension = int(self.dimension_combobox.currentText())
        print(f"Обрана розмірність: {self.dimension}")
        self.data_table.setColumnCount(self.dimension)
        self.data_table.setHorizontalHeaderLabels([f"X{i + 1}" for i in range(self.dimension)])

        # Оновлення графіка (видалення старого та створення нового)
        self.figure.clear()
        if self.dimension == 2:
            self.ax = self.figure.add_subplot(111)
        else:
            self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas.draw()

    def on_file_select(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Обрати файл", "", "Текстові файли (*.txt);;Всі файли (*)",
                                                   options=options)
        if file_name:
            self.load_data_from_file(file_name)

    def load_data_from_file(self, file_name):
        self.input_vectors = []
        with open(file_name, 'r') as f:
            for line in f:
                coordinates = [float(x) for x in line.split()]
                if len(coordinates) == self.dimension:
                    self.input_vectors.append(Point(coordinates))
        print(f"Дані завантажено з файлу: {file_name}")
        print(f"Кількість точок: {len(self.input_vectors)}")
        self.update_data_table()  # Оновлюємо таблицю після завантаження даних

    def update_data_table(self):
        self.data_table.setRowCount(len(self.input_vectors))
        for i, point in enumerate(self.input_vectors):
            for j, coord in enumerate(point.coordinates):
                item = QTableWidgetItem(str(coord))
                self.data_table.setItem(i, j, item)

    def on_calculate_click(self):
        self.cluster_bound_point_finder = ClusterBoundPointFinder(self.input_vectors,
                                                                  self.deviation)  # Зберігаємо cluster_bound_point_finder
        self.cluster_bound_point_finder.calculate_bound_points()

        self.draw_plot(self.cluster_bound_point_finder)  # Видаляємо kmeans_clusters з аргументів

    def on_slider_change(self):
        self.deviation = self.slider.value() / 100
        self.deviation_label.setText(f"Deviation: {self.deviation:.2f}")

        # Оновлюємо deviation в cluster_bound_point_finder
        self.cluster_bound_point_finder.deviation = self.deviation
        # Перераховуємо граничні точки
        self.cluster_bound_point_finder.calculate_bound_points()

        self.draw_plot(self.cluster_bound_point_finder)  # Перемальовуємо графік

    def on_add_data_click(self):
        self.input_vectors = [
            Point([random.uniform(0, 10) for _ in range(self.dimension)]) for _ in range(100)
        ]
        self.update_data_table()
        # text = self.input_edit.text()
        # try:
        #     point_strings = text.split(',')  # Розділяємо рядок на точки за допомогою коми
        #     for point_str in point_strings:
        #         coordinates = [float(x) for x in point_str.split()]  # Розділяємо координати пробілами
        #         if len(coordinates) == self.dimension:
        #             self.input_vectors.append(Point(coordinates))
        #         else:
        #             print(
        #                 f"Невірна кількість координат у точці '{point_str}'. Введіть {self.dimension} координати, розділені пробілами."
        #             )
        #     self.update_data_table()
        #     self.input_edit.clear()
        # except ValueError:
        #     print("Некоректний формат введення. Введіть числа, розділені пробілами та комами для розділення точок.")

    def draw_plot(self, cluster_bound_point_finder):
        self.ax.cla()  # Очищення графіка

        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

        if cluster_bound_point_finder.coordinates_count == 2:
            self.ax.scatter([p.coordinates[0] for p in cluster_bound_point_finder.input_vectors],
                            [p.coordinates[1] for p in cluster_bound_point_finder.input_vectors],
                            s=100, label="All points", c='grey')
        else:
            self.ax.scatter([p.coordinates[0] for p in cluster_bound_point_finder.input_vectors],
                            [p.coordinates[1] for p in cluster_bound_point_finder.input_vectors],
                            [p.coordinates[2] for p in cluster_bound_point_finder.input_vectors],
                            s=100, label="All points", c='grey')

        for cluster_index in range(len(cluster_bound_point_finder.cluster_centers)):
            center = cluster_bound_point_finder.cluster_centers[cluster_index]
            result = cluster_bound_point_finder.border_points[cluster_index]  # Беремо вже обчислені граничні точки

            if cluster_bound_point_finder.coordinates_count == 2:
                self.ax.scatter([p.coordinates[0] for p in result],
                                [p.coordinates[1] for p in result],
                                s=100, c=colors[cluster_index])
                self.ax.scatter(center.coordinates[0], center.coordinates[1], s=150, c='black', marker='X')
                self.ax.text(center.coordinates[0], center.coordinates[1], str(cluster_index), fontsize=12)
            else:
                self.ax.scatter([p.coordinates[0] for p in result],
                                [p.coordinates[1] for p in result],
                                [p.coordinates[2] for p in result],
                                s=100, c=colors[cluster_index])
                self.ax.scatter(center.coordinates[0], center.coordinates[1], center.coordinates[2], s=150,
                                c='black', marker='X')
                self.ax.text(center.coordinates[0], center.coordinates[1], center.coordinates[2], str(cluster_index),
                             fontsize=12)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        if cluster_bound_point_finder.coordinates_count == 3:
            self.ax.set_zlabel("Z")
        plt.grid(True)
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
