import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QPushButton, QVBoxLayout
from typing import List, Set, Tuple
import numpy as np
import matplotlib.pyplot as plt
import random
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

    def calculate_bound_points(self):
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans_clusters = kmeans.fit_predict(
            [[p.coordinates[i] for i in range(self.coordinates_count)] for p in self.input_vectors])

        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

        for i in np.arange(self.deviation, 1, 0.05):
            if self.coordinates_count == 2:
                fig, ax = plt.subplots()
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

            if self.coordinates_count == 2:
                ax.scatter([p.coordinates[0] for p in self.input_vectors],
                           [p.coordinates[1] for p in self.input_vectors],
                           s=100, label="All points", c='grey')
            else:
                ax.scatter([p.coordinates[0] for p in self.input_vectors],
                           [p.coordinates[1] for p in self.input_vectors],
                           [p.coordinates[2] for p in self.input_vectors],
                           s=100, label="All points", c='grey')

            print(f"Deviation: {i:.2f}")

            for cluster_index in range(self.n_clusters):
                cluster_points = [point for i, point in enumerate(self.input_vectors) if
                                  kmeans_clusters[i] == cluster_index]
                center = Point(kmeans.cluster_centers_[cluster_index].tolist())

                distances = {point: {inner_point: calc_distance(point, inner_point) for inner_point in cluster_points if
                                     inner_point != point} for point in cluster_points}
                closest_distances = {
                    point: dict(sorted(distances[point].items(), key=lambda item: item[1])[:self.core_point_count]) for
                    point in distances}
                mean_distances = {point: np.mean(list(dist.values())) for point, dist in closest_distances.items()}
                point_centres = {point: calculate_centre(set(closest.keys()), point, self.coordinates_count) for
                                 point, closest in closest_distances.items()}
                point_offsets = {point: calculate_point_offset(point, centre, mean_distances[point]) for point, centre
                                 in point_centres.items()}

                result = {point: offset for point, offset in point_offsets.items() if offset > i}
                print(f"  Cluster {cluster_index}: {len(result)} border points")

                if self.coordinates_count == 2:
                    ax.scatter([p.coordinates[0] for p in result],
                               [p.coordinates[1] for p in result],
                               s=100, c=colors[cluster_index])
                    ax.scatter(center.coordinates[0], center.coordinates[1], s=150, c='black', marker='X')
                else:
                    ax.scatter([p.coordinates[0] for p in result],
                               [p.coordinates[1] for p in result],
                               [p.coordinates[2] for p in result],
                               s=100, c=colors[cluster_index])
                    ax.scatter(center.coordinates[0], center.coordinates[1], center.coordinates[2], s=150, c='black',
                               marker='X')

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            if self.coordinates_count == 3:
                ax.set_zlabel("Z")
            plt.grid(True)
            plt.show()

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
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit([[point.coordinates[0], point.coordinates[1]] for point in self.input_vectors])
        cluster_centers = [Point(center.tolist()) for center in kmeans.cluster_centers_]
        return n_clusters, cluster_centers


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Пошук граничних точок кластерів")
        self.dimension = 2  # За замовчуванням вибрано 2D

        self.dimension_label = QLabel("Оберіть розмірність простору ознак:")
        self.dimension_combobox = QComboBox(self)
        self.dimension_combobox.addItems(["2", "3"])
        self.dimension_button = QPushButton("Обрати", self)
        self.dimension_button.clicked.connect(self.on_dimension_select)

        vbox = QVBoxLayout()
        vbox.addWidget(self.dimension_label)
        vbox.addWidget(self.dimension_combobox)
        vbox.addWidget(self.dimension_button)
        self.setLayout(vbox)

    def on_dimension_select(self):
        self.dimension = int(self.dimension_combobox.currentText())
        print(f"Обрана розмірність: {self.dimension}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())