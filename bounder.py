from typing import List, Set, Tuple
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D  # Імпорт для 3D графіків


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


if __name__ == '__main__':
    # Генерація 100 випадкових точок (2D або 3D)
    dimension = 3  # Оберіть 2 для 2D або 3 для 3D
    input_vectors = [Point([random.uniform(0, 10) for _ in range(dimension)]) for _ in range(100)]

    cluster_bound_point_finder = ClusterBoundPointFinder(input_vectors, 0.15)
    cluster_bound_point_finder.calculate_bound_points()
