import pytest
import pandas as pd

from mercap.utils.viz_utils import (smooth_control_profiles_two_step_median, 
                                    smooth_control_profiles_one_step_median)


def test_smooth_control_profiles_two_step_median_equal_weighting():
    """Test that two-step median gives equal weight to each Mars year.
    
    This test verifies that the function performs a two-step median:
    1. Median within each Mars year
    2. Median across Mars years
    
    If it were just taking the median across all profiles, a Mars year with
    many profiles would dominate. This test ensures each year contributes equally.
    """
    # Create test data where:
    # - Mars Year 1 has 100 profiles with T=100
    # - Mars Year 2 has 10 profiles with T=200
    # If we took median across all profiles, result would be ~100 (dominated by MY1)
    # With two-step median, result should be 150 (median of 100 and 200)
    
    n_profiles_my1 = 100
    n_profiles_my2 = 10
    sol = 0
    level = 10
    
    # Create profiles for Mars Year 1
    my1_data = {
        'mars_year': [1] * n_profiles_my1,
        'rel_sol_int': [sol] * n_profiles_my1,
        'level': [level] * n_profiles_my1,
        'T': [100.0] * n_profiles_my1,
        'Dust': [1e-3] * n_profiles_my1,
        'Dust_column': [0.01] * n_profiles_my1,
        'T_surf': [300.0] * n_profiles_my1,
    }
    
    # Create profiles for Mars Year 2 (double the values of MY1)
    my2_data = {
        'mars_year': [2] * n_profiles_my2,
        'rel_sol_int': [sol] * n_profiles_my2,
        'level': [level] * n_profiles_my2,
        'T': [200.0] * n_profiles_my2,
        'Dust': [2e-3] * n_profiles_my2,
        'Dust_column': [0.02] * n_profiles_my2,
        'T_surf': [320.0] * n_profiles_my2,
    }
    
    control_profiles_df = pd.concat([
        pd.DataFrame(my1_data),
        pd.DataFrame(my2_data)
    ], ignore_index=True)
    
    # Create minimal storm_profiles_df for sol range
    storm_profiles_df = pd.DataFrame({'rel_sol_int': [sol]})
    
    # Apply smoothing with window size 1 (no temporal smoothing for this test)
    result = smooth_control_profiles_two_step_median(
        control_profiles_df, 
        storm_profiles_df,
        smoothing_window_size=1
    )
    
    # Filter to the specific sol and level we're testing
    result_filtered = result[(result['rel_sol_int'] == sol) & (result['level'] == level)]
    
    assert len(result_filtered) == 1, "Should have exactly one result for this sol/level"
    
    # With two-step median:
    # Step 1: Median within MY1 = 100, Median within MY2 = 200
    # Step 2: Median across years = median(100, 200) = 150
    expected_t = 150.0
    expected_dust = 1.5e-3
    expected_dust_column = 0.015
    expected_t_surf = 310.0
    
    assert result_filtered['T'].iloc[0] == pytest.approx(expected_t, rel=1e-6), \
        f"Expected T={expected_t} (median of MY medians), got {result_filtered['T'].iloc[0]}"
    assert result_filtered['Dust'].iloc[0] == pytest.approx(expected_dust, rel=1e-6), \
        f"Expected Dust={expected_dust} (median of MY medians), got {result_filtered['Dust'].iloc[0]}"
    assert result_filtered['Dust_column'].iloc[0] == pytest.approx(expected_dust_column, rel=1e-6), \
        f"Expected Dust_column={expected_dust_column} (median of MY medians), got {result_filtered['Dust_column'].iloc[0]}"
    assert result_filtered['T_surf'].iloc[0] == pytest.approx(expected_t_surf, rel=1e-6), \
        f"Expected T_surf={expected_t_surf} (median of MY medians), got {result_filtered['T_surf'].iloc[0]}"


def test_smooth_control_profiles_one_step_median():
    """Test the old aggregate-then-smooth method."""
    n_profiles_my1 = 100
    n_profiles_my2 = 10
    sol = 0
    level = 10
    
    my1_data = {
        'mars_year': [1] * n_profiles_my1,
        'rel_sol_int': [sol] * n_profiles_my1,
        'level': [level] * n_profiles_my1,
        'T': [100.0] * n_profiles_my1,
        'Dust': [1e-3] * n_profiles_my1,
        'Dust_column': [0.01] * n_profiles_my1,
        'T_surf': [300.0] * n_profiles_my1,
    }
    
    my2_data = {
        'mars_year': [2] * n_profiles_my2,
        'rel_sol_int': [sol] * n_profiles_my2,
        'level': [level] * n_profiles_my2,
        'T': [200.0] * n_profiles_my2,
        'Dust': [2e-3] * n_profiles_my2,
        'Dust_column': [0.02] * n_profiles_my2,
        'T_surf': [320.0] * n_profiles_my2,
    }
    
    control_profiles_df = pd.concat([
        pd.DataFrame(my1_data),
        pd.DataFrame(my2_data)
    ], ignore_index=True)
    
    storm_profiles_df = pd.DataFrame({'rel_sol_int': [sol]})
    
    result = smooth_control_profiles_one_step_median(
        control_profiles_df,
        storm_profiles_df,
        smoothing_window_size=1
    )
    
    result_filtered = result[(result['rel_sol_int'] == sol) & (result['level'] == level)]
    
    # The original smoothing method takes mean across all profiles first, then applies rolling median
    expected_t = (100.0 * n_profiles_my1 + 200.0 * n_profiles_my2) / (n_profiles_my1 + n_profiles_my2)
    expected_dust = (1e-3 * n_profiles_my1 + 2e-3 * n_profiles_my2) / (n_profiles_my1 + n_profiles_my2)
    expected_dust_column = (0.01 * n_profiles_my1 + 0.02 * n_profiles_my2) / (n_profiles_my1 + n_profiles_my2)
    expected_t_surf = (300.0 * n_profiles_my1 + 320.0 * n_profiles_my2) / (n_profiles_my1 + n_profiles_my2)
    
    assert result_filtered['T'].iloc[0] == pytest.approx(expected_t, rel=1e-6), \
        f"Expected T≈{expected_t:.2f} (mean across all profiles), got {result_filtered['T'].iloc[0]}"
    assert result_filtered['Dust'].iloc[0] == pytest.approx(expected_dust, rel=1e-6), \
        f"Expected Dust≈{expected_dust:.6f} (mean across all profiles), got {result_filtered['Dust'].iloc[0]}"
    assert result_filtered['Dust_column'].iloc[0] == pytest.approx(expected_dust_column, rel=1e-6), \
        f"Expected Dust_column≈{expected_dust_column:.4f} (mean across all profiles), got {result_filtered['Dust_column'].iloc[0]}"
    assert result_filtered['T_surf'].iloc[0] == pytest.approx(expected_t_surf, rel=1e-6), \
        f"Expected T_surf≈{expected_t_surf:.2f} (mean across all profiles), got {result_filtered['T_surf'].iloc[0]}"