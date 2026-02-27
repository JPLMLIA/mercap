import pytest
import numpy as np

from mercap.utils.storm_data_proc import get_binary_mask_polygon


# Run by cd'ing to test directory and running:
# pytest .


# Define some simple diamond polygons in row/col coordinates
poly_1 = np.array([[1798, 2], [1799, 1], [1799, 2], [1799, 3], [1800, 2]])
poly_2 = np.array([[0, 3598], [1, 3597], [1, 3598], [1, 3599], [2, 3598]])

broken_poly_points = np.array([[1798, 0], [1798, 1], [1799, 0], [1799, 1],
                               [1798, 3598], [1798, 3599], [1799, 3598], [1799, 3599]])

contour_1 = np.array([[1798, 2], [1799, 1], [1799, 3], [1800, 2]])
contour_2 = np.array([[0, 3598], [1, 3597], [1, 3599], [2, 3598]])
contour_broken = np.array([[1798, 3598], [1798, 3601], [1799, 3598], [1799, 3601]])

# Define the binary masks
# TODO: Could also store as test files
EMPTY_IMAGE = np.zeros((1801, 3600), dtype=bool)
ONE_POLYGON_IMAGE = np.zeros((1801, 3600), dtype=bool)
SPLIT_POLYGON_IMAGE = np.zeros((1801, 3600), dtype=bool)
TWO_POLYGONS_IMAGE = np.zeros((1801, 3600), dtype=bool)

ONE_POLYGON_IMAGE[poly_1[:, 0], poly_1[:, 1]] = True
SPLIT_POLYGON_IMAGE[broken_poly_points[:, 0], broken_poly_points[:, 1]] = True
TWO_POLYGONS_IMAGE[poly_1[:, 0], poly_1[:, 1]] = True
TWO_POLYGONS_IMAGE[poly_2[:, 0], poly_2[:, 1]] = True


@pytest.mark.parametrize("binary_mask, xy_offset, poly_extraction_epsilon, expected", [
    (EMPTY_IMAGE, (0, 0), 0.001, pytest.raises(ValueError)),  # Should raise an error
    #(TWO_POLYGONS_IMAGE, (0, 0), 0.001, pytest.raises(ValueError)),  # Should raise an error
])
def test_get_binary_mask_polygon_errors(binary_mask, xy_offset, poly_extraction_epsilon, expected):
    with expected:
        get_binary_mask_polygon(binary_mask, xy_offset, poly_extraction_epsilon)


@pytest.mark.parametrize("binary_mask, xy_offset, poly_extraction_epsilon, erode_dilate, expected", 
                         [(ONE_POLYGON_IMAGE, (0, 0), 0.001, False, contour_1), 
                          (SPLIT_POLYGON_IMAGE, (0, 0), 0.001, False, contour_broken)])
def test_get_binary_mask_polygon(binary_mask, xy_offset, poly_extraction_epsilon, erode_dilate, expected):
    result = get_binary_mask_polygon(binary_mask, xy_offset, poly_extraction_epsilon, erode_dilate)

    # Convert from x/y (with 0,0 at TL corner) to row/col
    result[:, [1, 0]] = result[:, [0, 1]]
    result = result.round().astype(int)
    
    result_poly_point_set = set(map(tuple, result))
    expected_poly_point_set = set(map(tuple, expected))

    # Check if both sets of points are equal
    assert result_poly_point_set == expected_poly_point_set, "The sets of points are not equal."