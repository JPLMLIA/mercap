import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from mars_time.constants import seconds_per_sol, sols_per_year
from mercap.utils.profile_search import add_rel_sol_columns


@pytest.fixture
def storm_dt():
    """Fixture providing a reference storm datetime"""
    return datetime(2000, 1, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize("test_name,time_offset,expected_rel_sol", [
    ("one MY + one sol prior", -(sols_per_year + 1) * seconds_per_sol, -1),
    ("one MY - one sol prior", -(sols_per_year - 1) * seconds_per_sol, 1),
    ("one sol prior", -1 * seconds_per_sol, -1),
    ("same exact time", 0, 0),
    ("one sol forward", 1 * seconds_per_sol, 1),
    ("one MY - one sol forward", (sols_per_year - 1) * seconds_per_sol, -1),
    ("one MY + one sol forward", (sols_per_year + 1) * seconds_per_sol, 1),
])
def test_add_rel_sol_columns(storm_dt, test_name, time_offset, expected_rel_sol):
    """Test add_rel_sol_columns with various time offsets relative to storm date
    
    This test verifies that relative sol calculations correctly handle:
    - Time offsets before and after the reference storm date
    - Year wrapping (zero_out_year=True)
    - Different magnitudes of time offsets
    """
    test_time = storm_dt.timestamp() + time_offset
    test_datetime = datetime.fromtimestamp(test_time, tz=timezone.utc)
    
    df = pd.DataFrame({'dt': [test_datetime]})
    df = add_rel_sol_columns(df, storm_dt, zero_out_year=True, day_or_night_rounding=False)
    
    actual_rel_sol = df['rel_sol'].iloc[0]
    
    assert np.isclose(actual_rel_sol, expected_rel_sol, atol=1e-6), \
        f"{test_name}: expected {expected_rel_sol}, got {actual_rel_sol}"
