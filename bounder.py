import math
import random
import sys
from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QComboBox, QPushButton, QVBoxLayout,
                             QLineEdit, QFileDialog, QGridLayout, QTableWidget, QTableWidgetItem,
                             QSlider, QHBoxLayout, QTextEdit)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  # Adding PCA for projection


class Point:
    """
    A class that represents a point in a multidimensional space.
    """
    def __init__(self, coordinates: List[float]):
        """
        Initializes the Point object.

        :param coordinates: A list of point coordinates.
        """
        self.coordinates = coordinates

    def get_coordinates(self) -> List[float]:
        return self.coordinates

    def __eq__(self, other):
        """Overrides the method for comparing points."""
        return self.coordinates == other.coordinates

    def __hash__(self):
        """Overrides the method for using points in sets."""
        return hash(tuple(self.coordinates))


def calculate_centre(points: Set[Point], key_point: Point, coordinates_count: int) -> Point:
    """
    Calculates the centre of a set of points, including one key point.

    :param points: A set of points.
    :param key_point: The key point that is also taken into account when calculating the centre.
    :param coordinates_count: The number of coordinates in each point.
    :return: A point representing the centre of the set of points.
    """
    all_points = list(points)
    all_points.append(key_point)
    centre_coordinates = [
        sum(point.get_coordinates()[i] for point in all_points) / len(all_points)
        for i in range(coordinates_count)
    ]
    return Point(centre_coordinates)


def calc_distance(point_a: Point, point_b: Point) -> float:
    """
    Calculates the distance between two points.

    :param point_a: The first point.
    :param point_b: The second point.
    :return: The distance between the points.
    """
    return math.sqrt(
        sum(
            (point_a.get_coordinates()[i] - point_b.get_coordinates()[i]) ** 2
            for i in range(len(point_a.get_coordinates()))
        )
    )


def calculate_point_offset(point: Point, centre: Point, mean_distance: float) -> float:
    """
    Calculates the deviation of a point from the centre, normalised to the average distance.

    :param point: The point for which the deviation is calculated.
    :param centre: The centre of the cluster.
    :param mean_distance: The average distance from the point to its neighbours.
    :return: The normalised deviation of the point from the centre.
    """
    return calc_distance(point, centre) / mean_distance


class ClusterBoundPointFinder:
    """
    A class for finding boundary points in clusters.
    """
    def __init__(self, input_vectors: List[Point], deviation: float, max_clusters=10, core_point_count=20):
        """
        Initialises the ClusterBoundPointFinder object.

        :param input_vectors: A list of points to cluster.
        :param deviation: The deviation value for determining the boundary points.
        :param max_clusters: The maximum number of clusters to consider.
        """
        self.input_vectors = input_vectors
        self.deviation = deviation
        self.max_clusters = max_clusters
        self.coordinates_count = len(input_vectors[0].coordinates) if input_vectors else 0
        self.core_point_count = core_point_count * 4
        self.n_clusters, self.cluster_centers = self.find_cluster_centers()
        self.kmeans_clusters = None  # Saves the result of KMeans clustering
        self.border_points = []  # Stores boundary points for each cluster

    def calculate_bound_points(self):
        """
        Calculates the boundary points in each cluster.
        """
        if self.kmeans_clusters is None:
            self.kmeans = KMeans(n_clusters=self.n_clusters)
            self.kmeans_clusters = self.kmeans.fit_predict(
                [[p.coordinates[i] for i in range(self.coordinates_count)] for p in self.input_vectors]
            )

            # Updating cluster centres with KMeans
            self.cluster_centers = [Point(center.tolist()) for center in self.kmeans.cluster_centers_]

        self.border_points = []  # Clear the list of boundary points
        for cluster_index in range(self.n_clusters):
            # Обираємо точки, що належать до поточного кластера
            cluster_points = [
                point for i, point in enumerate(self.input_vectors) if self.kmeans_clusters[i] == cluster_index
            ]
            center = self.cluster_centers[cluster_index]

            # Calculate distances between points in a cluster
            distances = {
                point: {
                    inner_point: calc_distance(point, inner_point)
                    for inner_point in cluster_points
                    if inner_point != point
                }
                for point in cluster_points
            }

            # Find the closest points for each point in the cluster
            closest_distances = {
                point: dict(sorted(distances[point].items(), key=lambda item: item[1])[:self.core_point_count])
                for point in distances if point not in distances[point]
            }

            # Calculate the average distance to the nearest points
            mean_distances = {point: np.mean(list(dist.values())) for point, dist in closest_distances.items()}

            # Calculate the centre for each point and its neighbours
            point_centres = {
                point: calculate_centre(set(closest.keys()), point, self.coordinates_count)
                for point, closest in closest_distances.items()
            }

            # Обчислюємо відхилення точки від центру, нормалізоване до середньої відстані
            point_offsets = {
                point: calculate_point_offset(point, center, mean_distances[point])
                for point, centre in point_centres.items()
            }

            # Add a point to the list of boundary points if its deviation is greater than the specified threshold
            result = {point: offset for point, offset in point_offsets.items() if offset > self.deviation}
            self.border_points.append(result)
            print(f"  Cluster {cluster_index}: {len(result)} border points. Deviation: {self.deviation}")

    def find_cluster_centers(self) -> Tuple[int, List[Point]]:
        """
        Finds cluster centres using KMeans and determines the optimal number of clusters.

        :return: A tuple with the number of clusters and a list of cluster centres.
        """
        threshold = 30  # Threshold for the number of points

        if len(self.input_vectors) < threshold:
            n_clusters = 1
        else:
            wcss = []
            for i in range(1, self.max_clusters + 1):
                kmeans = KMeans(n_clusters=i)
                kmeans.fit([[p.coordinates[j] for j in range(self.coordinates_count)] for p in self.input_vectors])
                wcss.append(kmeans.inertia_)
            n_clusters = np.argmax(np.diff(wcss))

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit([[p.coordinates[i] for i in range(self.coordinates_count)] for p in self.input_vectors])
        cluster_centers = [Point(center.tolist()) for center in kmeans.cluster_centers_]
        return n_clusters, cluster_centers


class MainWindow(QWidget):
    """
    The main window of the program.
    """
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        """Initialises the user interface."""
        self.setWindowTitle("Search for cluster boundary points")
        self.dimension = 2 # Initial dimension of the feature space
        self.input_vectors = [] # List of input points
        self.deviation = 0.15 # Initial deviation value
        self.core_point_count = 5 # Initial value

        # Widgets for selecting the number of points
        self.core_point_label = QLabel("Core Point Count:", self)
        self.core_point_edit = QLineEdit(self)
        self.core_point_edit.setText(str(self.core_point_count))  # Set the initial value in the input field

        # Widgets for selecting a dimension
        self.dimension_label = QLabel("Select the dimension of the feature space:")
        self.dimension_combobox = QComboBox(self)
        self.dimension_combobox.addItems(["2", "3"])
        self.dimension_button = QPushButton("Select", self)
        self.dimension_button.clicked.connect(self.on_dimension_select)

        # Widgets for data entry
        self.input_label = QLabel("Enter the points:")
        self.input_edit = QLineEdit(self)
        self.file_button = QPushButton("Select file", self)
        self.file_button.clicked.connect(self.on_file_select)

        # Table for displaying data
        self.data_table = QTableWidget(self)
        self.data_table.setColumnCount(self.dimension)
        self.data_table.setHorizontalHeaderLabels([f"X{i + 1}" for i in range(self.dimension)])

        # Table for displaying the coordinates of boundary points
        self.border_points_table = QTableWidget(self)
        self.border_points_table.setColumnCount(self.dimension)
        self.border_points_table.setHorizontalHeaderLabels([f"X{i + 1}" for i in range(self.dimension)])

        # Labels for tables
        self.data_table_label = QLabel("List of points:", self)
        self.border_points_table_label = QLabel("List of limit points:", self)

        # Button ‘Add’
        self.add_button = QPushButton("Add points", self)
        self.add_button.clicked.connect(self.on_add_data_click)

        # Button ‘Calculate’
        self.calculate_button = QPushButton("Calculate", self)
        self.calculate_button.clicked.connect(self.on_calculate_click)

        # Schedule
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)  # Initially 2D graph

        # Slider to change the deviation value
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(15)
        self.slider.setMaximum(100)
        self.slider.setValue(int(self.deviation * 100))
        self.slider.valueChanged.connect(self.on_slider_change)

        # Мітка для значення deviation
        self.deviation_label = QLabel(f"Deviation: {self.deviation:.2f}", self)

        # Text field for displaying information
        self.info_text = QTextEdit(self)
        self.info_text.setReadOnly(True)

        # Placement of widgets
        vbox_main = QVBoxLayout(self)  # Main QVBoxLayout

        # --- Upper part ---
        grid_top = QGridLayout()
        grid_top.addWidget(self.dimension_label, 0, 0)
        grid_top.addWidget(self.dimension_combobox, 0, 1)
        grid_top.addWidget(self.dimension_button, 0, 2)
        grid_top.addWidget(self.input_label, 1, 0)
        grid_top.addWidget(self.input_edit, 2, 0, 1, 3)
        grid_top.addWidget(self.file_button, 3, 0)
        grid_top.addWidget(self.add_button, 3, 1)
        vbox_main.addLayout(grid_top)  # Add grid_top to vbox_main

        # --- Table of points ---
        vbox_table1 = QVBoxLayout()
        vbox_table1.addWidget(self.data_table_label)
        vbox_table1.addWidget(self.data_table)
        vbox_main.addLayout(vbox_table1)  # Add vbox_table1 to vbox_main

        # --- Button ‘Calculate’ ---
        vbox_main.addWidget(self.calculate_button)

        # --- Lower part ---
        hbox_bottom = QHBoxLayout()

        # --- Графік і повзунок ---
        vbox_plot = QVBoxLayout()
        vbox_plot.addWidget(self.canvas)
        vbox_slider = QVBoxLayout()
        vbox_slider.addWidget(self.slider)
        vbox_slider.addWidget(self.deviation_label)
        vbox_slider.addWidget(self.core_point_label)
        vbox_slider.addWidget(self.core_point_edit)
        vbox_plot.addLayout(vbox_slider)
        hbox_bottom.addLayout(vbox_plot)  # Add vbox_plot to hbox_bottom

        # --- Text box and limit point table ---
        vbox_info = QVBoxLayout()
        vbox_info.addWidget(self.info_text)
        vbox_table2 = QVBoxLayout()
        vbox_table2.addWidget(self.border_points_table_label)
        vbox_table2.addWidget(self.border_points_table)
        vbox_info.addLayout(vbox_table2)
        hbox_bottom.addLayout(vbox_info)  # Add vbox_info to hbox_bottom

        vbox_main.addLayout(hbox_bottom)  # Add hbox_bottom to vbox_main

        # --- Increasing the margins ---
        vbox_main.setContentsMargins(20, 20, 20, 20)  # Indents around the main widget
        grid_top.setContentsMargins(10, 10, 10, 10)  # Indents around the grid_top
        vbox_table1.setContentsMargins(10, 10, 10, 10)  # Indents around the point table
        vbox_table2.setContentsMargins(10, 10, 10, 10)  # Indents around the table of boundary points

    def on_dimension_select(self):
        """The choice of the dimensionality of the feature space is difficult."""
        self.dimension = int(self.dimension_combobox.currentText())
        print(f"Selected dimension: {self.dimension}")

        # Clear point and boundary point lists
        self.input_vectors = []
        if hasattr(self, 'cluster_bound_point_finder'):
            self.cluster_bound_point_finder.border_points = []

        # Update the number of columns in both tables
        self.data_table.setColumnCount(self.dimension)
        self.data_table.setHorizontalHeaderLabels([f"X{i + 1}" for i in range(self.dimension)])
        self.update_data_table()  # Update the point table

        self.border_points_table.setColumnCount(self.dimension)
        self.border_points_table.setHorizontalHeaderLabels([f"X{i + 1}" for i in range(self.dimension)])
        if hasattr(self, 'cluster_bound_point_finder'):
            self.update_border_points_table()  # Update the limit point table

        # Update the schedule (delete the old one and create a new one)
        self.figure.clear()
        if self.dimension == 2:
            self.ax = self.figure.add_subplot(111)
        else:
            self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas.draw()

    def on_file_select(self):
        """Handles the selection of a data file."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select file",
            "",
            "Data Files (*.data);;Всі файли (*)",
            options=options
        )
        if file_name:
            self.load_data_from_file(file_name)

    def load_data_from_file(self, file_name):
        """
        Loads data from a file.

        :param file_name: Path to the file with data.
        """
        self.input_vectors = []
        with open(file_name, 'r') as f:
            for line in f:
                coordinates = [float(x) for x in line.split()]
                self.input_vectors.append(Point(coordinates))

        # Project points immediately after downloading
        self.input_vectors = self.project_to_3d(self.input_vectors)

        print(f"Data loaded from file: {file_name}")
        print(f"Number of points: {len(self.input_vectors)}")
        self.update_data_table()  # Update the table after loading data

    def update_data_table(self):
        """Updates the table with data."""
        self.data_table.setRowCount(len(self.input_vectors))
        for i, point in enumerate(self.input_vectors):
            for j, coord in enumerate(point.coordinates):
                item = QTableWidgetItem(str(coord))
                self.data_table.setItem(i, j, item)

    def on_calculate_click(self):
        """Handles pressing the ‘Calculate’ button."""
        # Проектуємо точки на 3D простір, якщо потрібно
        try:
            self.core_point_count = int(self.core_point_edit.text())
        except ValueError:
            print("The Core Point Count value is incorrect. The default value (5) is used.")
        projected_points = self.project_to_3d(self.input_vectors)

        self.cluster_bound_point_finder = ClusterBoundPointFinder(
            projected_points, self.deviation, core_point_count=self.core_point_count
        )
        self.cluster_bound_point_finder.calculate_bound_points()

        # Define colours for each cluster
        self.cluster_colors = {}
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for cluster_index in range(self.cluster_bound_point_finder.n_clusters):
            self.cluster_colors[cluster_index] = colors[cluster_index]

        self.draw_plot(self.cluster_bound_point_finder)
        self.update_border_points_table()  # Update the limit point table

    def on_slider_change(self):
        """Handles changing the value of the deviation slider."""
        self.deviation = self.slider.value() / 100
        self.deviation_label.setText(f"Deviation: {self.deviation:.2f}")

        # Recalculate boundary points if the cluster_bound_point_finder object already exists
        if hasattr(self, 'cluster_bound_point_finder'):
            self.cluster_bound_point_finder.deviation = self.deviation
            self.cluster_bound_point_finder.calculate_bound_points()
            self.draw_plot(self.cluster_bound_point_finder)  # Перемальовуємо графік
            self.update_border_points_table()  # Оновлюємо таблицю граничних точок

    def on_add_data_click(self):
        """Handles pressing the ‘Add points’ button."""
        # Clear point and boundary point lists
        self.input_vectors = []
        if hasattr(self, 'cluster_bound_point_finder'):
            self.cluster_bound_point_finder.border_points = []

        text = self.input_edit.text()
        if not text:  # If the text field is empty, generate random points
            self.input_vectors = [
                Point([random.uniform(0, 10) for _ in range(self.dimension)]) for _ in range(100)
            ]
        else:  # Otherwise, add points from the text box
            try:
                point_strings = text.split(',')
                for point_str in point_strings:
                    coordinates = [float(x) for x in point_str.split()]
                    if len(coordinates) == self.dimension:
                        self.input_vectors.append(Point(coordinates))
                    else:
                        print(
                            f"Incorrect number of coordinates in a point '{point_str}'. Enter {self.dimension} coordinates separated by spaces."
                        )
                self.input_edit.clear()  # Очищуємо текстове поле
            except ValueError:
                print("The input format is incorrect. Enter numbers separated by spaces and commas to separate points.")

        # Project points immediately after adding them
        self.input_vectors = self.project_to_3d(self.input_vectors)

        self.update_data_table()  # Update the table

    def update_border_points_table(self):
        """Updates the table with boundary points."""
        border_points = []
        for cluster_index in range(len(self.cluster_bound_point_finder.border_points)):
            border_points.extend(list(self.cluster_bound_point_finder.border_points[cluster_index].keys()))

        self.border_points_table.setRowCount(len(border_points))
        for i, point in enumerate(border_points):
            for j in range(self.dimension):
                item = QTableWidgetItem(str(point.coordinates[j]))
                self.border_points_table.setItem(i, j, item)

    def project_to_3d(self, points: List[Point]) -> List[Point]:
        """
        Projects a list of points onto a 3D space using PCA.

        :param points: A list of points to project.
        :return: A list of points projected onto the 3D space.
        """
        if len(points[0].get_coordinates()) <= 3:
            return points  # No need for projection if dimension <= 3

        # Create a NumPy array from the coordinates of the points
        data = np.array([[p.coordinates[i] for i in range(self.dimension)] for p in points])

        # We create and train PCAs with 3 components
        pca = PCA(n_components=3)
        pca.fit(data)

        # Transform data and return a list of points
        transformed_data = pca.transform(data)
        return [Point(list(row)) for row in transformed_data]


    def draw_plot(self, cluster_bound_point_finder):
        """Draws a graph with points and boundary points."""
        self.ax.cla()  # Clearing the schedule

        # Project points onto 3D space
        projected_points = self.project_to_3d(cluster_bound_point_finder.input_vectors)

        # Draw all points in grey
        if cluster_bound_point_finder.coordinates_count == 2:
            self.ax.scatter(
                [p.coordinates[0] for p in projected_points],
                [p.coordinates[1] for p in projected_points],
                s=100, label="All points", c='grey'
            )
        else:
            self.ax.scatter(
                [p.coordinates[0] for p in projected_points],
                [p.coordinates[1] for p in projected_points],
                [p.coordinates[2] for p in projected_points],
                s=100, label="All points", c='grey'
            )

        info_string = ""
        for cluster_index in range(len(cluster_bound_point_finder.cluster_centers)):
            center = cluster_bound_point_finder.cluster_centers[cluster_index]
            result = cluster_bound_point_finder.border_points[cluster_index]  # Take the already calculated boundary points
            cluster_color = self.cluster_colors[cluster_index]
            info_string += f"Cluster {cluster_index} ({cluster_color}): {len(result)} boundary points\n"

            if cluster_bound_point_finder.coordinates_count == 2:
                # Drawing boundary points for the current cluster
                self.ax.scatter([p.coordinates[0] for p in result],
                                [p.coordinates[1] for p in result],
                                s=100, c=cluster_color)
                # Draw the centre of the cluster
                self.ax.scatter(center.coordinates[0], center.coordinates[1], s=150, c='black', marker='X')
                # Add text with the cluster number
                self.ax.text(center.coordinates[0], center.coordinates[1], str(cluster_index), fontsize=14, color='red')
            else:
                # Drawing boundary points for the current cluster
                self.ax.scatter([p.coordinates[0] for p in result],
                                [p.coordinates[1] for p in result],
                                [p.coordinates[2] for p in result],
                                s=100, c=cluster_color)
                # Draw the centre of the cluster
                self.ax.scatter(center.coordinates[0], center.coordinates[1], center.coordinates[2], s=150,
                                c='black', marker='X')
                # Add text with the cluster number
                self.ax.text(center.coordinates[0], center.coordinates[1], center.coordinates[2], str(cluster_index),
                            fontsize=12, color='red')

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