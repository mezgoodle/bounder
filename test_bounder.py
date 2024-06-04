import pytest
from typing import List, Set
from bounder import Point, calculate_centre, calc_distance, calculate_point_offset, ClusterBoundPointFinder


def test_point_initialization():
    coordinates = [1.0, 2.0, 3.0]
    point = Point(coordinates)
    assert point.get_coordinates() == coordinates


def test_point_equality():
    point1 = Point([1.0, 2.0, 3.0])
    point2 = Point([1.0, 2.0, 3.0])
    point3 = Point([4.0, 5.0, 6.0])
    assert point1 == point2
    assert point1 != point3


def test_point_hash():
    point1 = Point([1.0, 2.0, 3.0])
    point2 = Point([1.0, 2.0, 3.0])
    point_set = {point1, point2}
    assert len(point_set) == 1  # They should be considered the same point


def test_calculate_centre():
    points: Set[Point] = {Point([1.0, 1.0]), Point([3.0, 3.0])}
    key_point = Point([2.0, 2.0])
    centre = calculate_centre(points, key_point, 2)
    assert centre.get_coordinates() == [2.0, 2.0]


def test_calc_distance():
    point1 = Point([1.0, 2.0])
    point2 = Point([4.0, 6.0])
    distance = calc_distance(point1, point2)
    assert distance == pytest.approx(5.0, 0.0001)


def test_calculate_point_offset():
    point = Point([1.0, 1.0])
    centre = Point([4.0, 5.0])
    mean_distance = 5.0
    offset = calculate_point_offset(point, centre, mean_distance)
    assert offset == pytest.approx(1.0, 0.0001)


def test_find_cluster_centers():
    points = [Point([1.0, 1.0]), Point([2.0, 2.0]), Point([3.0, 3.0])]
    cluster_finder = ClusterBoundPointFinder(points, deviation=0.1)
    n_clusters, centers = cluster_finder.find_cluster_centers()
    assert n_clusters == 1  # In this case, all points should belong to one cluster
    assert len(centers) == 1
    assert centers[0].get_coordinates() == [2.0, 2.0]


def test_calculate_bound_points():
    points = [Point([1.0, 1.0]), Point([2.0, 2.0]), Point([3.0, 3.0]), Point([10.0, 10.0])]
    cluster_finder = ClusterBoundPointFinder(points, deviation=0.1, max_clusters=2)
    cluster_finder.calculate_bound_points()
    border_points = cluster_finder.border_points
    assert len(border_points) == 1  # Since we expect 2 clusters, we should have 2 lists of border points
    assert len(border_points[0]) == 4  # Four border point in the first cluster
