import pytest

from bounder import (
    Point,
    calc_distance,
    calculate_centre,
    calculate_point_offset,
)


# Fixture to create points for testing
@pytest.fixture
def points():
    return [Point([1, 2]), Point([4, 6]), Point([7, 8])]


# Tests for Point class
def test_point_equality():
    point_a = Point([1, 2, 3])
    point_b = Point([1, 2, 3])
    point_c = Point([3, 2, 1])
    assert point_a == point_b
    assert point_a != point_c


def test_point_hash():
    point_set = {Point([1, 2]), Point([3, 4])}
    assert Point([1, 2]) in point_set
    assert Point([5, 6]) not in point_set


# Tests for calculate_centre function
def test_calculate_centre(points):
    key_point = Point([10, 12])
    centre = calculate_centre(set(points), key_point, 2)
    expected_centre = Point([5.5, 7.0])  # Calculated manually
    assert centre == expected_centre


# Tests for calc_distance function
def test_calc_distance():
    point_a = Point([0, 0])
    point_b = Point([3, 4])
    distance = calc_distance(point_a, point_b)
    assert distance == 5  # 3-4-5 triangle
