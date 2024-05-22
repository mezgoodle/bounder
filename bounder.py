import random
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, QPushButton, QVBoxLayout,
    QLineEdit, QFileDialog, QGridLayout, QTableWidget, QTableWidgetItem,
    QSlider, QHBoxLayout, QTextEdit
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
            self.kmeans = KMeans(n_clusters=self.n_clusters)
            self.kmeans_clusters = self.kmeans.fit_predict(
                [[p.coordinates[i] for i in range(self.coordinates_count)] for p in self.input_vectors]
            )

            # Оновлюємо центри кластерів з KMeans
            self.cluster_centers = [Point(center.tolist()) for center in self.kmeans.cluster_centers_]

        self.border_points = []
        for cluster_index in range(self.n_clusters):
            cluster_points = [
                point for i, point in enumerate(self.input_vectors) if self.kmeans_clusters[i] == cluster_index
            ]
            center = self.cluster_centers[cluster_index]  # Використовуємо оновлені центри

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
        та визначає оптимальну кількість кластерів.
        """
        threshold = 20  # Поріг для кількості точок

        if len(self.input_vectors) < threshold:
            n_clusters = 2  # Якщо точок мало, встановлюємо 2 кластери
        else:
            wcss = []
            for i in range(1, self.max_clusters + 1):
                kmeans = KMeans(n_clusters=i)
                kmeans.fit([[p.coordinates[j] for j in range(self.coordinates_count)] for p in self.input_vectors])
                wcss.append(kmeans.inertia_)
            n_clusters = np.argmax(np.diff(wcss)) + 2  # Якщо точок багато, використовуємо метод "ліктя"

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

        # Таблиця для відображення координат граничних точок
        self.border_points_table = QTableWidget(self)
        self.border_points_table.setColumnCount(self.dimension)
        self.border_points_table.setHorizontalHeaderLabels([f"X{i + 1}" for i in range(self.dimension)])

        # Мітки для таблиць
        self.data_table_label = QLabel("Список точок:", self)
        self.border_points_table_label = QLabel("Список border точок:", self)

        # Кнопка "Додати"
        self.add_button = QPushButton("Додати", self)
        self.add_button.clicked.connect(self.on_add_data_click)

        # Кнопка "Розрахувати"
        self.calculate_button = QPushButton("Розрахувати", self)
        self.calculate_button.clicked.connect(self.on_calculate_click)

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

        # Текстове поле для виведення інформації
        self.info_text = QTextEdit(self)
        self.info_text.setReadOnly(True)

        # Розміщення віджетів
        vbox_main = QVBoxLayout(self)  # Головний QVBoxLayout

        # --- Верхня частина ---
        grid_top = QGridLayout()
        grid_top.addWidget(self.dimension_label, 0, 0)
        grid_top.addWidget(self.dimension_combobox, 0, 1)
        grid_top.addWidget(self.dimension_button, 0, 2)
        grid_top.addWidget(self.input_label, 1, 0)
        grid_top.addWidget(self.input_edit, 2, 0, 1, 3)
        grid_top.addWidget(self.file_button, 3, 0)
        grid_top.addWidget(self.add_button, 3, 1)
        vbox_main.addLayout(grid_top)  # Додаємо grid_top до vbox_main

        # --- Таблиця точок ---
        vbox_table1 = QVBoxLayout()
        vbox_table1.addWidget(self.data_table_label)
        vbox_table1.addWidget(self.data_table)
        vbox_main.addLayout(vbox_table1)  # Додаємо vbox_table1 до vbox_main

        # --- Кнопка "Розрахувати" ---
        vbox_main.addWidget(self.calculate_button)

        # --- Нижня частина ---
        hbox_bottom = QHBoxLayout()

        # --- Графік і повзунок ---
        vbox_plot = QVBoxLayout()
        vbox_plot.addWidget(self.canvas)
        vbox_slider = QVBoxLayout()
        vbox_slider.addWidget(self.slider)
        vbox_slider.addWidget(self.deviation_label)
        vbox_plot.addLayout(vbox_slider)
        hbox_bottom.addLayout(vbox_plot)  # Додаємо vbox_plot до hbox_bottom

        # --- Текстове поле та таблиця граничних точок ---
        vbox_info = QVBoxLayout()
        vbox_info.addWidget(self.info_text)
        vbox_table2 = QVBoxLayout()
        vbox_table2.addWidget(self.border_points_table_label)
        vbox_table2.addWidget(self.border_points_table)
        vbox_info.addLayout(vbox_table2)
        hbox_bottom.addLayout(vbox_info)  # Додаємо vbox_info до hbox_bottom

        vbox_main.addLayout(hbox_bottom)  # Додаємо hbox_bottom до vbox_main

        # --- Збільшення відступів ---
        vbox_main.setContentsMargins(20, 20, 20, 20)  # Відступи навколо головного віджета
        grid_top.setContentsMargins(10, 10, 10, 10)  # Відступи навколо grid_top
        vbox_table1.setContentsMargins(10, 10, 10, 10)  # Відступи навколо таблиці точок
        vbox_table2.setContentsMargins(10, 10, 10, 10)  # Відступи навколо таблиці граничних точок

    def on_dimension_select(self):
        self.dimension = int(self.dimension_combobox.currentText())
        print(f"Обрана розмірність: {self.dimension}")

        # Очищуємо списки точок та граничних точок
        self.input_vectors = []
        if hasattr(self, 'cluster_bound_point_finder'):
            self.cluster_bound_point_finder.border_points = []

        # Оновлюємо кількість стовпців в обох таблицях
        self.data_table.setColumnCount(self.dimension)
        self.data_table.setHorizontalHeaderLabels([f"X{i + 1}" for i in range(self.dimension)])
        self.update_data_table()  # Оновлюємо таблицю точок

        self.border_points_table.setColumnCount(self.dimension)
        self.border_points_table.setHorizontalHeaderLabels([f"X{i + 1}" for i in range(self.dimension)])
        if hasattr(self, 'cluster_bound_point_finder'):  # Додано умову
            self.update_border_points_table()  # Оновлюємо таблицю граничних точок

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

        self.cluster_colors = {}
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for cluster_index in range(self.cluster_bound_point_finder.n_clusters):
            self.cluster_colors[cluster_index] = colors[cluster_index]

        self.draw_plot(self.cluster_bound_point_finder)
        self.update_border_points_table()  # Оновлюємо таблицю граничних точок

    def on_slider_change(self):
        self.deviation = self.slider.value() / 100
        self.deviation_label.setText(f"Deviation: {self.deviation:.2f}")

        # Оновлюємо deviation в cluster_bound_point_finder
        self.cluster_bound_point_finder.deviation = self.deviation
        # Перераховуємо граничні точки
        self.cluster_bound_point_finder.calculate_bound_points()

        self.draw_plot(self.cluster_bound_point_finder)  # Перемальовуємо графік
        self.update_border_points_table()  # Оновлюємо таблицю граничних точок

    def on_add_data_click(self):
        # Очищуємо списки точок та граничних точок
        self.input_vectors = []
        if hasattr(self, 'cluster_bound_point_finder'):
            self.cluster_bound_point_finder.border_points = []

        text = self.input_edit.text()
        if not text:  # Якщо текстове поле пусте, генеруємо випадкові точки
            self.input_vectors = [
                Point([random.uniform(0, 10) for _ in range(self.dimension)]) for _ in range(100)
            ]
        else:  # Інакше додаємо точки з текстового поля
            try:
                point_strings = text.split(',')
                for point_str in point_strings:
                    coordinates = [float(x) for x in point_str.split()]
                    if len(coordinates) == self.dimension:
                        self.input_vectors.append(Point(coordinates))
                    else:
                        print(
                            f"Невірна кількість координат у точці '{point_str}'. Введіть {self.dimension} координати, розділені пробілами."
                        )
                self.input_edit.clear()  # Очищуємо текстове поле
            except ValueError:
                print("Некоректний формат введення. Введіть числа, розділені пробілами та комами для розділення точок.")

        self.update_data_table()  # Оновлюємо таблицю

    def update_border_points_table(self):
        border_points = []
        for cluster_index in range(len(self.cluster_bound_point_finder.border_points)):
            border_points.extend(list(self.cluster_bound_point_finder.border_points[cluster_index].keys()))

        self.border_points_table.setRowCount(len(border_points))
        for i, point in enumerate(border_points):
            for j in range(self.dimension):  # Використовуємо self.dimension для кількості координат
                item = QTableWidgetItem(str(point.coordinates[j]))  # Виводимо j-ту координату
                self.border_points_table.setItem(i, j, item)

    def draw_plot(self, cluster_bound_point_finder):
        self.ax.cla()  # Очищення графіка

        if cluster_bound_point_finder.coordinates_count == 2:
            self.ax.scatter([p.coordinates[0] for p in cluster_bound_point_finder.input_vectors],
                            [p.coordinates[1] for p in cluster_bound_point_finder.input_vectors],
                            s=100, label="All points", c='grey')
        else:
            self.ax.scatter([p.coordinates[0] for p in cluster_bound_point_finder.input_vectors],
                            [p.coordinates[1] for p in cluster_bound_point_finder.input_vectors],
                            [p.coordinates[2] for p in cluster_bound_point_finder.input_vectors],
                            s=100, label="All points", c='grey')

        info_string = ""
        for cluster_index in range(len(cluster_bound_point_finder.cluster_centers)):
            center = cluster_bound_point_finder.cluster_centers[cluster_index]
            result = cluster_bound_point_finder.border_points[cluster_index]  # Беремо вже обчислені граничні точки
            cluster_color = self.cluster_colors[cluster_index]
            info_string += f"Кластер {cluster_index} ({cluster_color}): {len(result)} граничних точок\n"

            if cluster_bound_point_finder.coordinates_count == 2:
                self.ax.scatter([p.coordinates[0] for p in result],
                                [p.coordinates[1] for p in result],
                                s=100, c=self.cluster_colors[cluster_index])
                self.ax.scatter(center.coordinates[0], center.coordinates[1], s=150, c='black', marker='X')
                self.ax.text(center.coordinates[0], center.coordinates[1], str(cluster_index), fontsize=14, color='red')
            else:
                self.ax.scatter([p.coordinates[0] for p in result],
                                [p.coordinates[1] for p in result],
                                [p.coordinates[2] for p in result],
                                s=100, c=self.cluster_colors[cluster_index])
                self.ax.scatter(center.coordinates[0], center.coordinates[1], center.coordinates[2], s=150,
                                c='black', marker='X')
                self.ax.text(center.coordinates[0], center.coordinates[1], center.coordinates[2], str(cluster_index), fontsize=12, color='red')

        self.info_text.setText(info_string)
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
