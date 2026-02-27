import logging

import pandas as pd
from sqlalchemy import text
import numpy as np
import datetime

from mars_time import marstime_to_datetime, datetime_to_marstime, MarsTime, MarsTimeDelta
from mars_time.constants import seconds_per_sol, sols_per_year
from mcstools import L2Loader
from mercap.config import INT_SOLS_PER_MY, DROP_INVALID_DDR1


logging.basicConfig(level=logging.INFO,
                    datefmt='%H:%M:%S',
                    format='%(asctime)s | %(lineno)d | %(levelname)-5s | %(module)-15s | %(message)s')


def get_storms_with_min_conf(connection, min_conf, data_source='mdssd', verbose=False):

    # Construct SQL statement
    text_statement = text(f'''
                          SELECT storms.id 
                          FROM {data_source}_table storms
                          WHERE storms.conflev >= {min_conf} 
                          ''')
    # Execute the search and get results as a Dataframe
    storm_db_ids = pd.read_sql(text_statement, connection)

    if verbose:
        logging.info(f'SQL call:\n\n{text_statement}')

        logging.info(f'\nReturned storm DB IDs:')
        logging.info(str(storm_db_ids.info()))

    return storm_db_ids


def get_storm_metadata_by_database_id(connection, db_id, data_source='mdssd', verbose=False):
    """Gets a storm by its primary key in the database. NOTE: Different than MDSSD's `storm_id`"""

    # Construct SQL statement
    text_statement = text(f'''
                          SELECT storms.* 
                          FROM {data_source}_table storms
                          WHERE storms.id = {db_id}
                          ''')
    # Execute the search and get results as a Dataframe
    storm_metadata = pd.read_sql(text_statement, connection)

    if verbose:
        logging.info(f'SQL call:\n\n{text_statement}')

        logging.info(f'\nReturned storm data:')
        logging.info(str(storm_metadata.info()))

    return storm_metadata


def get_storm_metadata_by_storm_id(connection, storm_id, data_source='mdssd', verbose=False):

    # Construct SQL statement
    text_statement = text(f'''
                          SELECT storms.* 
                          FROM {data_source}_table storms
                          WHERE storms.storm_id = :storm_id 
                          ''')
    bind_params = {'storm_id': str(storm_id)}

    # Execute the search and get results as a Dataframe
    storm_metadata = pd.read_sql(text_statement, connection, params=bind_params)

    if verbose:
        logging.info(f'SQL call:\n\n{text_statement}')

        logging.info(f'\nReturned storm data:')
        logging.info(str(storm_metadata.info()))

    return storm_metadata


def get_storm_metadata_by_seqid(connection, storm_seqid, data_source='mdssd', verbose=False):

    # Construct SQL statement
    text_statement = text(f'''
                          SELECT storms.* 
                          FROM {data_source}_table storms
                          WHERE storms.seq_id = :storm_seqid 
                          ''')
    bind_params = {'storm_seqid': str(storm_seqid)}

    # Execute the search and get results as a Dataframe
    storm_metadata = pd.read_sql(text_statement, connection, params=bind_params)
    if verbose:
        logging.info(f'SQL call:\n\n{text_statement}')

        logging.info(f'\nReturned storm data:')
        logging.info(str(storm_metadata.info()))

    return storm_metadata


def get_dt_for_ls_diff(storm_dt, ls_diff):
    """Helper to get the datetime given a MY, Ls and an Ls difference (positive or negative)
    
    Note, we expect this to be slower and less accurate than using datetime and seconds directly (e.g., `get_dt_for_sol_diff`)
    """

    mt = datetime_to_marstime(storm_dt)

    new_my = mt.year
    new_ls = mt.solar_longitude + ls_diff  # Taking manual approach as MarsTimeDelta doesn't handle Ls differences

    # Handle wrapping around years
    if new_ls < 0:
        new_ls += 360
        new_my -= 1
    elif new_ls > 360:
        new_ls -= 360
        new_my += 1

    if new_my > 360 or new_my < 0:
        raise ValueError(f"New Mars year {new_my} is out of range. Are you trying to span multiple years?")

    return marstime_to_datetime(MarsTime.from_solar_longitude(new_my, new_ls))


def get_dt_for_sol_diff(storm_dt, sol_diff):
    """Helper to get the datetime given a MY, sol, and a sol difference (positive or negative)

    Note, we expect this to be faster and more accurate than using solar longitudes (e.g., `get_dt_for_ls_diff`)
    """

    return storm_dt + datetime.timedelta(seconds=sol_diff * seconds_per_sol)


def add_rel_sol_columns(df, storm_dt, zero_out_year=False, day_or_night_rounding=False):
    """
    Add a column to a DataFrame that calculates the relative sol (float and int) 
    of each profile relative to the storm of interest
    
    We want to work in MarsTime here so that we have the option to drop the MY difference 
    (and get only the seasonal difference). This is valuable for control matching
    """
    
    # Ensure the 'dt' column is timezone-aware and in UTC
    if df['dt'].dt.tz is None:
        # Assume naive datetime is in UTC (matching MCS data)
        df['dt'] = df['dt'].dt.tz_localize('UTC')
    else:
        df['dt'] = df['dt'].dt.tz_convert('UTC')
        
    # Compute the time difference in seconds as a numpy array/Series
    time_diff_sec = (df['dt'] - storm_dt).dt.total_seconds()
    sol_diff = time_diff_sec / seconds_per_sol

    # XXX: Using the mod and looking for min(abs(diff)) works as long as we assume a small window (here, +/- 20 sols)
    if zero_out_year:
        # Compute sol differences to within a year, but not adjusted for year crossings
        sol_diff_unadjusted = (sol_diff % sols_per_year).to_numpy()[:, np.newaxis]
        sol_diff_pos_neg = np.hstack([sol_diff_unadjusted, sol_diff_unadjusted - sols_per_year, sol_diff_unadjusted + sols_per_year])

        # Identify whether the nearest year crossing is closer storm_dt, and use it if it is
        nearest_index = np.argmin(np.abs(sol_diff_pos_neg), axis=1)
        sol_diff = sol_diff_pos_neg[np.arange(len(nearest_index)), nearest_index]

    df['rel_sol'] = sol_diff
    
    '''
    # Original approach using MarsTimeDelta
    storm_mt = datetime_to_marstime(storm_dt)

    # Loop over rows, calculate the MarsTime object for each row, subtract the storm's time to get a MarsTimeDelta
    mt_diff = df.apply(lambda row: (MarsTime(row['mars_year'], row['sol']) - storm_mt), axis=1)

    # For controls, we will likely want to discard the difference in Mars years. (Effectively, "mod" by one Mars Year)
    if zero_out_year:
        rel_sol_list = []
        for temp_mt_diff in mt_diff:
            # Won't know if we want year prior or year after, so test both and take the min
            # NOTE: Working with solar longitude could be slightly more accurate, but involves an optimization step, which is computationally expensive
            sol_diffs = [MarsTimeDelta(year=-1, sol=temp_mt_diff.sol).sols, 
                         MarsTimeDelta(year=0, sol=temp_mt_diff.sol).sols]
            min_idx = np.argmin(np.abs(sol_diffs))
            sol_diff = sol_diffs[min_idx]

            rel_sol_list.append(sol_diff)
    else:
        rel_sol_list = [temp_mt_diff.sols for temp_mt_diff in mt_diff]
    df['rel_sol'] = rel_sol_list
    '''
    
    if day_or_night_rounding:
        # TODO: could rename the column to something like `rel_sol_discretized` since we no longer are using integers
        rel_sol_orig = np.array(df['rel_sol'])
        df['rel_sol_int'] = np.round(rel_sol_orig * 2) / 2
    else:
        # Round to nearest sol. Since orbits are consistently 
        df['rel_sol_int'] = df['rel_sol'].round().astype(int)

    return df


def find_ddr1_storm_profile_matches(connection, storm_db_id, storm_dt, time_window_bounds, data_source, ltst_start=None, ltst_end=None, good_obs_quals=None, data_source_date_start=None, data_source_date_end=None, verbose=False):
    """Identify profiles intersecting with storm in space and time.""" 
    # Setup some optional filters for the search statement

    # If desired, filter by time window
    time_filter = ""
    bind_params_update = {}

    if time_window_bounds[0] >= time_window_bounds[1]:
        raise ValueError('First element in time_window_bounds must be less than the second. Got: {time_window_bounds}')

    # Calculate the start and end datetimes for the time window
    start_dt = get_dt_for_sol_diff(storm_dt, time_window_bounds[0])
    end_dt = get_dt_for_sol_diff(storm_dt, time_window_bounds[1])

    # Generate the SQL filter statement and update bind parameters
    time_filter = 'AND mcs.dt BETWEEN :start_dt AND :end_dt '
    bind_params_update = {'start_dt': start_dt, 'end_dt': end_dt}

    # Add data source date range filter
    data_source_date_filter = ''
    if data_source_date_start and data_source_date_end:
        data_source_date_filter = 'AND mcs.dt BETWEEN :data_source_date_start AND :data_source_date_end '
        bind_params_update['data_source_date_start'] = data_source_date_start
        bind_params_update['data_source_date_end'] = data_source_date_end

    # Add local true solar time filter and observation quality filter
    ltst_start_filter = f'AND mcs.ltst >= {ltst_start} ' if ltst_start else ''
    ltst_end_filter = f'AND mcs.ltst < {ltst_end} ' if ltst_end else ''
    obs_qual_filter = f'AND mcs.obs_qual IN {good_obs_quals} ' if good_obs_quals else ''

    # Construct SQL statement
    text_statement = text(f'''
                          SELECT mcs.* 
                          FROM mcs_profiles_2d mcs 
                          JOIN {data_source}_table storm ON ST_Within(mcs.profile_loc, storm.storm_polygon) 
                          WHERE storm.id = :storm_db_id
                          {time_filter}
                          {data_source_date_filter}
                          {ltst_start_filter}
                          {ltst_end_filter}
                          {obs_qual_filter}
                          ''')
    # Execute the search and get results as a Dataframe
    bind_params = {'storm_db_id': str(storm_db_id)}
    bind_params.update(bind_params_update)
    storm_profile_hits = pd.read_sql(text_statement, connection, params=bind_params)

    if verbose:
        logging.info(f'Storm SQL query call:\n\n{text_statement}')
        logging.info(f'\nReturned profile Dataframe:')
        logging.info(str(storm_profile_hits.info()))

    return storm_profile_hits


def find_ddr1_control_profile_matches(connection, storm_db_id, storm_dt, data_source, ls_tolerance=None, time_window_bounds=None, ltst_start=None, ltst_end=None, good_obs_quals=None, control_exclusions_dict=None, data_source_date_start=None, data_source_date_end=None, verbose=False):
    """Identify profiles that would serve as controls for a known storm.
    
    Use either `ls_tolerance` (a float) or `time_window_bounds` (a tuple of ints representing sols) to specify the desired time window.
    
    Parameters
    ----------
    connection : SQLAlchemy connection
        Database connection
    storm_db_id : int
        Database ID of the storm to find controls for
    storm_dt : int
        Datetime of the storm of interest. Control profiles will never come from the same Mars year as the storm of interest.
    data_source : str
        Data source name (mdssd or mdad)
    ls_tolerance : float, optional
        Solar longitude tolerance for matching controls. If used, must be positive. 
        We only care about seasonal changes here, so the Mars Years component of the difference is dropped. 
        Specify either this or `time_window_bounds`.
    time_window_bounds : tuple of int, optional
        Sol range before and after the storm to search for controls (e.g., (-10, 11)). Note last sol is not included. 
        We only care about seasonal changes here, so the Mars Years component of the difference is dropped. 
        Specify either this or `ls_tolerance`.
    ltst_start : float, optional
        Minimum local true solar time to include
    ltst_end : float, optional
        Maximum local true solar time to include
    good_obs_quals : list, optional
        List of acceptable observation quality values
    control_exclusions_dict : dict, optional
        Dictionary containing Mars years as keys and list of integer sols as values to exclude from the search.
        These should correspond to known storm overlaps/confounders from other Mars years that need to be removed from the search results.
    data_source_date_start : datetime, optional
        Earliest allowed date for data source
    data_source_date_end : datetime, optional
        Latest allowed date for data source
    verbose : bool, optional
        Whether to print verbose output
        
    Returns
    -------
    pandas.DataFrame
        DataFrame of matching control profiles
    """ 

    #########################################################
    # Setup a time filter. We want to exclude profiles that are within the exact time window of the storm
    # Calculate time window bounds using either ls_tolerance (Ls) or time_window_bounds (sols)
    if ls_tolerance and time_window_bounds:
        raise ValueError('Cannot use both ls_tolerance and time_window_bounds')

    if ls_tolerance:
        if ls_tolerance < 0:
            raise ValueError(f'ls_tolerance must be positive. Got: {ls_tolerance}')
        start_dt = get_dt_for_ls_diff(storm_dt, -ls_tolerance)
        end_dt = get_dt_for_ls_diff(storm_dt, ls_tolerance)
    elif time_window_bounds:
        if time_window_bounds[0] >= time_window_bounds[1]:
            raise ValueError('Time window bound first element must be less than second. Got: {time_window_bounds}')
        start_dt = get_dt_for_sol_diff(storm_dt, time_window_bounds[0])
        end_dt = get_dt_for_sol_diff(storm_dt, time_window_bounds[1])
    else:
        raise ValueError('Must provide either ls_tolerance or time_window_bounds')

    time_filter = 'AND NOT mcs.dt BETWEEN :start_dt AND :end_dt '  # Exclude profiles that exactly match the storm's time window
    bind_params = {'start_dt': start_dt, 'end_dt': end_dt}

    #########################################################
    # Setup a seasonal filter. We want to include profiles with matching Ls values
    start_ls = datetime_to_marstime(start_dt).solar_longitude
    end_ls = datetime_to_marstime(end_dt).solar_longitude
    
    if start_ls < 0 or end_ls < 0:
        raise ValueError('Start and end Ls must be positive. Got: {start_ls}, {end_ls}')
    if start_ls >= 360 or end_ls >= 360:
        raise ValueError('Start and end Ls must be less than 360. Got: {start_ls}, {end_ls}')

    if start_ls <= end_ls:
        # Normal case: no wrapping around 360
        season_filter = "AND mcs.l_s BETWEEN :start_ls AND :end_ls "
    else:
        # Wrapping case: season crosses 360/0 boundary
        season_filter = "AND (mcs.l_s >= :start_ls OR mcs.l_s <= :end_ls) "

    bind_params['start_ls'] = start_ls
    bind_params['end_ls'] = end_ls
    
    #########################################################
    # Setup some optional filters for the local true solar time and observation quality
    ltst_start_filter = f'AND mcs.ltst >= {ltst_start} ' if ltst_start else ''
    ltst_end_filter = f'AND mcs.ltst < {ltst_end} ' if ltst_end else ''
    obs_qual_filter = f'AND mcs.obs_qual IN {good_obs_quals} ' if good_obs_quals else ''

    #########################################################
    # Add data source date range filter
    data_source_date_filter = ''
    if data_source_date_start and data_source_date_end:
        data_source_date_filter = 'AND mcs.dt BETWEEN :data_source_date_start AND :data_source_date_end '
        bind_params['data_source_date_start'] = data_source_date_start
        bind_params['data_source_date_end'] = data_source_date_end

    #########################################################
    # Add storm confounder exclusion criteria. This excludes profiles that overlapped with OTHER storms
    exclusion_filter_list = []
    if control_exclusions_dict:
        for exclusion_my, exclusion_sols in control_exclusions_dict.items():
            if not all(isinstance(sol, int) for sol in exclusion_sols):
                raise ValueError(f"All sols in an exclusion list must be integers. Found non-integer in list: {exclusion_sols}")
            exclusion_filter_list.append(f"NOT (mcs.mars_year = {exclusion_my} AND mcs.sol::INTEGER IN {exclusion_sols})")

    exclusion_filter = f"AND {' AND '.join(exclusion_filter_list)} " if exclusion_filter_list else ""

    #########################################################
    # Construct SQL statement
    text_statement = text(f'''
                          SELECT mcs.*, storm.mars_year AS storm_mars_year, storm.ls AS storm_ls, storm.storm_id, storm.seq_id AS storm_seq_id
                          FROM mcs_profiles_2d AS mcs
                          JOIN {data_source}_table AS storm ON ST_Within(mcs.profile_loc, storm.storm_polygon)
                          WHERE storm.id = :storm_db_id
                          {time_filter}
                          {season_filter}
                          {data_source_date_filter}
                          {ltst_start_filter}
                          {ltst_end_filter}
                          {obs_qual_filter}
                          {exclusion_filter}
                          ''')
    bind_params['storm_db_id'] = str(storm_db_id)
    control_profile_hits = pd.read_sql(text_statement, connection, params=bind_params)

    if verbose:
        logging.info(f'Control SQL query call:\n\n{text_statement}')
        logging.info(f'\nReturned profile Dataframe:')
        logging.info(str(control_profile_hits.info()))

    return control_profile_hits


def load_ddr1_from_profile_matches(profile_hits, verbose=False):
    """Load DDR1 data for a selected group of profiles identified with
    ``find_ddr1_storm_profile_matches`` or ``find_ddr1_control_profile_matches``.

    This is the DDR1-only variant of ``load_ddr2_from_profile_matches``.
    It loads profile-level metadata (column-integrated quantities, coordinates,
    quality flags, etc.) from MCS flat files without loading per-level DDR2 data.

    Parameters
    ----------
    profile_hits : pd.DataFrame
        Output of a profile query.  Required columns: ``["dt", "Date", "UTC",
        "mars_year", "sol", "rel_sol", "rel_sol_int"]``.
    verbose : bool
        Whether to print extra info to command line.

    Returns
    -------
    pd.DataFrame
        Matched DDR1 data with one row per profile.
    """
    if 'rel_sol' not in profile_hits.columns or 'rel_sol_int' not in profile_hits.columns:
        raise RuntimeError(
            'profile_hits must contain `rel_sol` and `rel_sol_int`. '
            'Try applying utils.profile_search.add_rel_sol_columns() first.')

    l2 = L2Loader()

    ddr1 = l2.load_from_datetimes("DDR1", profile_hits['dt'], add_cols=['dt'],
                                  verbose=verbose)

    if DROP_INVALID_DDR1:
        ddr1 = ddr1.loc[ddr1['1'] == 0].drop(columns='1')
    else:
        ddr1 = ddr1.drop(columns='1')

    matched_ddr1 = pd.merge(
        ddr1,
        profile_hits[['Date', 'UTC', 'mars_year', 'sol', 'rel_sol', 'rel_sol_int']],
        on=['Date', 'UTC'], how='inner')

    if verbose:
        logging.info('Loaded DDR1 data for %d profiles',
                     matched_ddr1['Profile_identifier'].nunique())

    return matched_ddr1


def load_ddr2_from_profile_matches(profile_hits, min_dust_permitted=None, verbose=False):
    """
    Loads DDR1 and DDR2 data for a selected group of profiles identified with 
    `find_ddr1_storm_profile_matches` or `find_ddr1_control_profile_matches`

    Note: profile datetimes in MCS database are rounded to the second,
    but the flat files are at the microsececond level. So, first load
    all 4-hour L2 DDR1 data containing the datetimes, then merge on
    the "Date" and "UTC" consistent between the database and flat files.
    Then use unique profile label added by mcstools package to load
    DDR2 only from the specific profiles of a given datetime. Merges
    DDR2, DDR1, and storm data (DDR1 and storm data duplicataed at each
    pressure level of an individual profile).
    
    Parameters
    ----------
    strom_profile_hits: output of profile query
        required cols: ["dt", "Date", "UTC"]
    min_dust_permitted: float
        Minimum dust permitted in DDR2 data. If None, no filtering is done.
    verbose: bool
        Whether or not to print extra info to command line

    Returns
    -------
    merged_ddr: DDR1, DDR2, strom data (one row per pressure level)
    """

    # Initialize L2 loader from mcstools (configured with env variables)
    l2 = L2Loader()

    # Load DDR1 info (containing profile metadata) for all possible profiles that match our datetimes
    ddr1 = l2.load_from_datetimes("DDR1", profile_hits['dt'], add_cols=['dt'], verbose=verbose)
    
    # If desired, keep only rows where '1' column == 0 (valid data). Otherwise, discard it
    if DROP_INVALID_DDR1:
        ddr1 = ddr1.loc[ddr1['1'] == 0].drop(columns='1')
    else:
        ddr1 = ddr1.drop(columns='1')
        
    if 'rel_sol' not in profile_hits.columns or 'rel_sol_int' not in profile_hits.columns:
        raise RuntimeError('Need to include `rel_sol` or `rel_sol_int`. Try applying utils.profile_search.add_rel_sol_columns()')

    # Filter down to just the (Date, UTC) matches, merge DDR1 and storm data
    matched_ddr1 = pd.merge(ddr1, profile_hits[['Date', 'UTC', 'mars_year', 'sol', 'rel_sol', 'rel_sol_int']], 
                            on=['Date', 'UTC'], how='inner')

    # Get unique profile identifier and then load DDR2 info (with atmospheric data)
    ddr2 = l2.load("DDR2", profiles=matched_ddr1["Profile_identifier"], verbose=verbose)

    # Combine both types of information into one Dataframe
    merged_ddr = l2.merge_ddrs(ddr2, matched_ddr1, verbose=verbose)
    if verbose:
        logging.info(f'Loaded DDR2 data for {merged_ddr["Profile_identifier"].nunique()} profiles')

    # If dust is below threshold, set to NaN
    if min_dust_permitted:
        merged_ddr.loc[merged_ddr['Dust'] < min_dust_permitted, 'Dust'] = np.nan

    return merged_ddr


def generate_control_exclusions_dict(storm_years_str, storm_sols_str, sol_exclusion_window_size):
    """Helper funtion to create a dictionary of exclusions from a list of storm years and sols
    
    Parameters
    ----------
    storm_years_str : str
        String of comma separated storm years that will help determine storms to exclude
    storm_sols_str : str
        String of comma separated storm sols that will help determine storms to exclude
    sol_exclusion_window_size : tuple of int
        +/- this many sols will be excluded around each (MY, sol) value. The second value
        should be inclusive of the last desired sol. For example, an argument like 
        (10, 11) would search for -/+ 10 sols before/after the storm sol.
        
    Returns
    -------
    exclusion_dict : dict
        Dictionary of storm years as keys tuples of sols as values. If no storms are provided, returns None.
    """

    # Extract the integer values from a comma separated string
    storm_years = [int(y) for y in storm_years_str.split(',')] if storm_years_str else None
    storm_sols = [int(s) for s in storm_sols_str.split(',')] if storm_sols_str else None

    # No exclusions, return None
    if not storm_years and not storm_sols:
        return None

    # Check that the two lists are the same length
    if len(storm_years) != len(storm_sols):
        raise ValueError('storm_years and storm_sols must be the same length')

    # Create the exclusion dictionary
    exclusion_dict = {}
    for storm_year, storm_sol in zip(storm_years, storm_sols):
        # Make sure we don't fall outside the range [0, Max Mars sols]
        sol_exclusions = list(range(max(0, storm_sol - sol_exclusion_window_size[0]), 
                                    min(INT_SOLS_PER_MY + 1, storm_sol + sol_exclusion_window_size[1])))
        
        if storm_year not in exclusion_dict:
            exclusion_dict[storm_year] = sol_exclusions
        else:
            exclusion_dict[storm_year].extend(sol_exclusions)

    # Convert values to unique tuples using sets. This makes sure we don't have duplicates
    for storm_year, storm_sols in exclusion_dict.items():
        exclusion_dict[storm_year] = tuple(sorted(set(storm_sols)))

    return exclusion_dict
