from pathlib import Path
import logging
from datetime import datetime as dt
from datetime import timezone

import cv2
import numpy as np
import scipy.io as sio
import netCDF4 as nc
import sqlalchemy
from shapely.ops import split
from shapely.affinity import translate
from shapely.geometry import Polygon, MultiPolygon, LineString
from sqlalchemy.exc import SQLAlchemyError
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_config

from mars_time import datetime_to_marstime
from mcstools.util.time import sols_elapsed
from mcstools.util.geom import haversine_dist

from mercap.config import INT_SOLS_PER_MY, MARCI_DT_FORMAT, MARS_GEOID_A


logging.basicConfig(level=logging.INFO,
                    datefmt='%H:%M:%S',
                    format='%(asctime)s | %(lineno)d | %(levelname)-5s | %(module)-15s | %(message)s')


def read_netcdf_to_dict(fpath):
    """
    Reads a NetCDF file and returns a dictionary with variable names as keys
    and their corresponding data as values.

    Parameters
    ----------
    fpath : str
        Path to the NetCDF file.

    Returns
    -------
    dict
        A dictionary where each key is a variable name from the NetCDF file
        and each value is the data associated with that variable.

    Example
    -------
    >>> fpath = 'example.nc'
    >>> netcdf_data = read_netcdf_to_dict(fpath)
    >>> print(netcdf_data['dayList'])  # Assuming 'dayList' is a variable in the file
   """
    data_dict = {}

    with nc.Dataset(fpath, 'r') as dataset:
        for var_name in dataset.variables:
            data_dict[var_name] = dataset.variables[var_name][:]

    return data_dict


def read_mdssd_idl(mdssd_filepath):
    """Helper to read an IDL file and extract storm boundary"""
    idl = sio.readsav(mdssd_filepath)
    results = idl['results']
    output = results[0]
    names = results.dtype.names

    return output, names


def get_polygon_properties(contour):
    '''Helper to extract some metrics for a polygon that describe its shape'''

    properties = {}

    # Aspect ratio
    _, _, w, h = cv2.boundingRect(contour)
    properties['aspect_ratio'] = w/h

    # Solidity - indicates how much of the outer boundary is concave
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    properties['solidity'] = area / hull_area

    # Eccentricity -- indicates how circular the boundary is
    #XXX Some confusion about whether it's major/minor or minor/major
    '''
    if len(contour) > 4:  # Open CV needs at least 5 points to fit an ellipse
        _, (major, minor), _ = cv2.fitEllipse(contour)
        properties['ellipse_eccentricity'] = np.sqrt(1 - (major / minor) ** 2)
    '''
    # Compactness -- also indicates how circular the boundary is
    perimeter = cv2.arcLength(contour, True)
    properties['compactness'] = 4 * np.pi * area / (perimeter ** 2)

    return properties


def average_longitude(longitudes):
    """
    Computes the average longitude for a list of longitudes, correctly handling 
    the wrap-around at the antemeridian.
    """
    sum_x = sum(np.cos(np.radians(lon)) for lon in longitudes)
    sum_y = sum(np.sin(np.radians(lon)) for lon in longitudes)

    mean_lon = np.degrees(np.arctan2(sum_y, sum_x))

    adjust_longitude = ((mean_lon + 180) % 360) - 180

    return adjust_longitude


def poly_fix_antimeridian_crossing(poly):
    """Helper to detect if a polygon wraps around a significant chunk of the globe instead of spanning the antimeridian"""

    lons, lats = np.array(poly.exterior.xy)

    candidate_poly = Polygon(list(zip(lons, lats)))

    # Case shouldn't arise, but catch it just in case
    if np.any(lons > 180) and np.any(lons < -180):
        raise RuntimeError('Polygon error, lon vals should only extend beyond [-180, 180] on one edge')

    if np.any(lons > 180) and not np.any(lons < -180):
        # Construct appriopriate edge of the antimeridian
        easterly_split_line = LineString([(180, 91), (180, -91)])
        # Split the polygon at the antimeridian
        split_polys = [geom for geom in split(candidate_poly, easterly_split_line).geoms]

        # Determine which polygon hangs beyond the [-180, 180] long bound, and shift it
        for si, split_poly in enumerate(split_polys):
            if np.max(split_poly.exterior.xy[0]) > 180:
                split_polys[si] = translate(split_poly, xoff=-360)

        # Replace the polygon with a multipolygon containing both sides of the pixel
        if len(split_polys) == 1:
            return split_polys[0]

        return MultiPolygon(split_polys)

    if not np.any(lons > 180) and np.any(lons < -180):
        # Construct appriopriate edge of the antimeridian
        westerly_split_line = LineString([(-180, 90), (-180, -90)])
        # Split the polygon at the antimeridian
        split_polys = [geom for geom in split(candidate_poly, westerly_split_line).geoms]

        # Determine which polygon hangs beyond the [-180, 180] long bound, and shift it
        for si, split_poly in enumerate(split_polys):
            if np.min(split_poly.exterior.xy[0]) < -180:
                split_polys[si] = translate(split_poly, xoff=+360)

        # Replace the polygon with a multipolygon containing both sides of the pixel
        if len(split_polys) == 1:
            return split_polys[0]
        return MultiPolygon(split_polys)

    # No modification needed
    return candidate_poly


def get_binary_mask_polygon(binary_storm_mask, xy_offset=(0, 0), poly_extraction_epsilon=0.001,
                            erode_dilate=True, logger=logging.getLogger('mercap_logger')):
    """Helper to convert binary storm masks (from MDGMs) into polygons"""
    if np.all(binary_storm_mask == 0):
        raise ValueError('Storm mask is completely blank.')

    # Extract all contours in an image mask
    mask = binary_storm_mask.astype(np.uint8)  # Cast to uint8 for contour finding
    if erode_dilate:
        # Extract all contours in an image mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(cv2.erode(mask, kernel), kernel)

        if np.sum(mask) == 0:
            logger.error('After morphological operations, mask has no positive pixels')
            return

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, 
                                   offset=xy_offset)
    contours = list(contours)
    contour_areas = np.array([cv2.contourArea(cont) for cont in contours])

    # Check that we don't get multiple storm masks per object
    if len(contours) > 1:
        contour_areas_normed_orig = contour_areas / np.sum(contour_areas)
        
        # Handle broken storm contour (case where single mask doesn't capture at least 95% of the storm's area)
        if np.max(contour_areas_normed_orig) < 0.95:
            logger.warning("Broken storm contour. Proportions: %s", str(contour_areas_normed_orig))
            repeated_mask = np.tile(mask, (1, 2))  # Tile the MDGM mask twice so we see both sides of the antimeridian
            contours_tiled, _ = cv2.findContours(repeated_mask, cv2.RETR_EXTERNAL, 
                                                 cv2.CHAIN_APPROX_SIMPLE, offset=xy_offset)
            
            # Note: contours could have longitude values > 180 at this point. 
            contour_areas_new = np.array([cv2.contourArea(cont) for cont in contours_tiled])
            contour_areas_normed = contour_areas_new / np.sum(contour_areas_new)
            contour_ind = np.argmax(contour_areas_normed)  # Take the largest contour
            contour = contours_tiled[contour_ind].astype(np.float32) # Must be float 32 for poly approximation

        # Relatively good storm contour (case where a single mask DOES capture at least 95% of the storm's area)
        else:
            contour_ind = np.argmax(contour_areas_normed_orig)
            contour = contours[contour_ind].astype(np.float32) # Must be float 32 for poly approximation

        selected_contour_area_normed = cv2.contourArea(contour) / np.sum(contour_areas)
        logger.warning(f"Consolidating {len(contours)} found contours. Captured proportion: {selected_contour_area_normed:0.4f}")
    else:
        contour = contours[0]

    # Check that we get at least 4 corners per contour mask
    if len(contour) < 4:
        raise RuntimeError('Not enough points identified to create a polygon')

    polygon = cv2.approxPolyDP(contour, poly_extraction_epsilon, True)
    polygon = polygon.squeeze()  # Remove extra dimension inserted by OpenCV

    return polygon


def write_storms_csv(df, csv_output_fpath):
    """Helper to write MDAD/MDSSD data to CSV"""

    logging.info('Saving output to CSV: %s', csv_output_fpath)
    df.to_csv(Path(csv_output_fpath))
    logging.info('CSV save complete.')


def write_storms_db(df, dataset_name, db_url, db_column_types, 
                    db_existing_behavior='replace'):
    """Helper to write MDAD/MDSSD data to a database"""

    logging.info('Saving output to database: %s', db_url)

    # Dump to SQL using specified data types
    df.to_sql(f'{dataset_name}_table', db_url, index=True, index_label='id', 
                 if_exists=db_existing_behavior, dtype=db_column_types)

    # Add geospatial indices for faster querying
    logging.info('Attempting to add indices to the %s DB...', dataset_name)
    engine = sqlalchemy.create_engine(db_url)
    with engine.begin() as conn:
        try:
            ind_1 = sqlalchemy.text(f'CREATE INDEX IF NOT EXISTS polygon_index ON {dataset_name}_table USING GIST (storm_polygon);') 
            ind_2 = sqlalchemy.text(f'CREATE INDEX IF NOT EXISTS dt_index ON {dataset_name}_table (dt);') 
            conn.execute(ind_1)
            conn.execute(ind_2)
            # Explicitly commit the transaction
            conn.commit()
            logging.info('Geospatial indices added successfully. Database write complete.')
        except SQLAlchemyError as e:
            # The transaction is automatically rolled back if an exception is raised
            logging.error(f"An error occurred: {e}")


def find_intersecting_storms(df, temporal_overlap_size, polygon_col='storm_polygon', id_col='StormID', year_col='Year'):
    """
    Find all storm polygons that intersect with each other and record their IDs, years, and sols.
    
    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing storm data with polygon geometries
    temporal_overlap_size : int
        +/- this many sols checked when searching for storms that overlap in time
    polygon_col : str, optional
        Name of the column containing polygon geometries, by default 'storm_polygon'
    id_col : str, optional
        Name of the column containing storm IDs, by default 'StormID'
    year_col : str, optional
        Name of the column containing Mars years, by default 'Year'
    
    Returns
    -------
    pandas.DataFrame
        Original DataFrame with three new columns: 'intersecting_storm_ids', 
        'intersecting_storm_years', and 'intersecting_storm_sols'
    """
    # Convert to GeoDataFrame with the polygon column as geometry
    gs = gpd.GeoSeries.from_wkt(df[polygon_col])
    gdf = gpd.GeoDataFrame(df, geometry=gs)

    # Initialize lists for storing intersection info
    intersecting_storm_ids, intersecting_storm_years, intersecting_storm_sols = [], [], []

    # Iterate through each row and find intersections
    for gi, row in tqdm(gdf.iterrows(), total=len(gdf), 
                        desc='Finding storm spatial intersections for control deconfounding'):
        poly_of_interest = row.geometry

        # Handle empty polygons
        if poly_of_interest == 'POLYGON EMPTY':
            intersecting_storm_ids.append('')
            intersecting_storm_years.append('')
            intersecting_storm_sols.append('')
            continue

        # Find which polygons intersect with the polygon of interest on the same sol in other years
        spatial_intersections = gdf.geometry.intersects(poly_of_interest)
        year_non_intersections = gdf[year_col] != row[year_col]
        sol_exclusion_search_window = range(max(0, row.sol_int - temporal_overlap_size), 
                                            min(INT_SOLS_PER_MY + 1, row.sol_int + temporal_overlap_size + 1))
        sol_intersections = gdf['sol_int'].isin(sol_exclusion_search_window)

        # Find storms that geospatially intersect a storm of interest on the same sol in other years
        intersecting_storm_info = gdf.loc[spatial_intersections & year_non_intersections & sol_intersections, 
                                          [id_col, year_col, 'sol_int']]

        # Convert to strings and append to lists
        intersecting_storm_ids.append(','.join(intersecting_storm_info[id_col].astype(str)))
        intersecting_storm_years.append(','.join(intersecting_storm_info[year_col].astype(str)))
        intersecting_storm_sols.append(','.join(intersecting_storm_info['sol_int'].astype(str)))

    # Create a new dataframe with only the intersection information
    intersection_df = pd.DataFrame({'intersecting_storm_ids': intersecting_storm_ids, 
                                    'intersecting_storm_years': intersecting_storm_years, 
                                    'intersecting_storm_sols': intersecting_storm_sols})

    return intersection_df


def load_marci_cumindex(cumindex_fpath):
    """
    Load and process MARCI cumindex.tab file.
    
    Parameters
    ----------
    cumindex_fpath : Path
        Path to the cumindex.tab file
        
    Returns
    -------
    pd.DataFrame
        Processed cumindex dataframe with Path (filename only) as index, 
        and lon, Ls columns
    """
    # Load cumindex file, keep only columns 1, 4, 15, 45 (0-indexed here, 1-indexed in PDS labelfile)
    cumindex_df = pd.read_csv(cumindex_fpath, header=None, 
                              usecols=[1, 4, 15, 45, 50], names=['Path', 'image_time', 'lon', 'Ls', 'orbit_num'])

    # Extract filename only from Path column and set as index
    cumindex_df['Path'] = cumindex_df['Path'].apply(lambda x: Path(x).name)
    cumindex_df.set_index('Path', inplace=True)

    return cumindex_df


def calculate_marci_precise_timing(storm_df, marci_lists_dir, cumindex_df, mdgm_col='MDGM'):
    """
    Calculate precise Ls values for each storm using MARCI image data.
    
    Parameters
    ----------
    storm_df : pd.DataFrame
        Storm dataframe containing mdgm and lon columns
    marci_lists_dir : Path
        Directory containing MARCI lists for each phase
    cumindex_df : pd.DataFrame
        Processed cumindex dataframe with image info
    mdgm_col : str, optional
        Name of the column containing MDGM IDs, by default 'MDGM'
        
    Returns
    -------
    image_time_precise: list
        List of datetime values for each MARCI swath corresponding to a storm
    ls_precise: list
        List of ls_precise values for each MARCI swath corresponding to a storm
    orbit_nums: list
        List of orbit numbers corresponding to each MARCI swath
    """
    image_time_precise = []
    ls_precise = []
    orbit_nums = []


    # TODO: could make more efficienty by creating two loops: one over MDGMs and an inner loop over storm_IDs
    for _, row in tqdm(storm_df.iterrows(), total=len(storm_df), desc='Calculating precise Ls values'):
        try:
            # Extract phase and day from MDGM column (e.g., "P12" and "day09" from "P12day09")
            mdgm = row[mdgm_col].replace('_', '')  # MDAD includes an underscore between the phase and day (MDSSD does not)
            phase = mdgm[:3]  # First 3 characters (e.g., "P12", "D01")
            day = mdgm[3:]

            # Construct path to list file
            list_file_path = Path(marci_lists_dir) / phase / f"{phase}_{day}.list"

            if not list_file_path.exists():
                logging.warning(f'MARCI list file not found: {list_file_path}')
                image_time_precise.append(np.nan)
                ls_precise.append(np.nan)
                orbit_nums.append(np.nan)
                continue

            # Read the list file to get MARCI image filenames
            with open(list_file_path, 'r') as f:
                marci_filenames = [line.strip() for line in f if line.strip()]

            # Find matching entries in cumindex dataframe
            marci_lons = []
            marci_image_times = []
            marci_ls_values = []
            marci_orbit_numbers = []

            for filename in marci_filenames:
                if filename in cumindex_df.index:
                    marci_lons.append(cumindex_df.loc[filename, 'lon'])
                    marci_image_times.append(cumindex_df.loc[filename, 'image_time'])
                    marci_ls_values.append(cumindex_df.loc[filename, 'Ls'])
                    marci_orbit_numbers.append(cumindex_df.loc[filename, 'orbit_num'])

            if not marci_lons:
                logging.warning(f'No matching MARCI images found in cumindex for {mdgm}')
                image_time_precise.append(np.nan)
                ls_precise.append(np.nan)
                orbit_nums.append(np.nan)
                continue

            # Convert to arrays for longitude offset calculation
            # MARCI is originally in west longitude. Convert to east [0, 360) and then to correct interval: [-180, 180)
            marci_lons_east = -1 * np.array(marci_lons) + 360
            marci_lons_east_adjusted = ((marci_lons_east + 180) % 360) - 180

            marci_image_times = np.array(marci_image_times)
            marci_ls_values = np.array(marci_ls_values)
            marci_orbit_numbers = np.array(marci_orbit_numbers)
            
            # Calculate longitude distances handling wrapping if diffs > 180 degrees
            storm_lon = row['lon']
            lon_diffs = np.abs(marci_lons_east_adjusted - storm_lon)
            lon_diffs_adjusted = np.minimum(lon_diffs, 360 - lon_diffs)
            if np.any(lon_diffs_adjusted < 0):
                raise RuntimeError('Error in calculating closest MARCI swath. Generated negative longitude difference')

            # Find index with minimum distance and store values from cumulative index file
            min_idx = np.argmin(lon_diffs_adjusted)

            # All MARCI datetimes are in UTC, so set the timezone here
            marci_start_dt = dt.strptime(marci_image_times[min_idx], MARCI_DT_FORMAT).replace(tzinfo=timezone.utc)
            image_time_precise.append(marci_start_dt)
            ls_precise.append(marci_ls_values[min_idx])
            orbit_nums.append(marci_orbit_numbers[min_idx])

        except Exception as e:
            logging.warning(f'Error calculating ls_precise for {row.get(mdgm_col, "unknown")}: {e}')
            image_time_precise.append(np.nan)
            ls_precise.append(np.nan)
            orbit_nums.append(np.nan)
 
    return image_time_precise, ls_precise, orbit_nums


def apply_marci_timing_precision(df, marci_lists_dir, cumindex_df, n_jobs,
                                 mdgm_col='MDGM', mars_year_col='Year', 
                                 ls_col='Ls', dt_col='dt',
                                 logger=logging.getLogger(__name__)):
    """
    Apply MARCI-based timing precision to storm dataframe.
    
    Updates mars_year (calculated from datetime), ls, dt, orbit_num.
    Stores originals as: mars_year_orig, ls_orig, sol_orig (if sol exists).
    
    Note: This function does NOT call add_mars_sols() - caller should do that after
    this function returns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Storm dataframe to update
    marci_lists_dir : Path
        Directory containing MARCI lists for each phase
    cumindex_df : pd.DataFrame
        Processed cumindex dataframe with image info
    n_jobs : int
        Number of parallel workers
    mdgm_col : str
        Name of MDGM column
    mars_year_col : str
        Name of Mars year column
    ls_col : str
        Name of Ls column
    dt_col : str
        Name of datetime column
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Updated dataframe with precise timing values
    """
    df = df.copy()
    
    logger.info('Adding precise Ls/datetime values using MARCI data')
    dt_precise, ls_precise, orbit_nums = calculate_marci_precise_timing(df, marci_lists_dir, cumindex_df, mdgm_col=mdgm_col)

    df[dt_col] = dt_precise
    df['orbit_num'] = np.array(orbit_nums).astype(int)
    
    ######################################
    # Calculate Mars Year from datetime. Need to do this as it may need updating for storms spanning the start/end of a Mars year
    my_precise = np.array([datetime_to_marstime(temp_dt).year for temp_dt in dt_precise]).astype(int)
    my_diffs = np.sum((my_precise - df[mars_year_col]).abs() > 0)
    df.rename(columns={mars_year_col: f'{mars_year_col}_orig'}, inplace=True)
    df[mars_year_col] = my_precise
    
    # Error check to make sure the Ls values are not very different
    ls_change = np.abs(df[ls_col].to_numpy() - np.array(ls_precise))
    if np.any(np.logical_and(ls_change > 1, ls_change < 359)):
        logger.warning('Large difference between original ls and ls_precise. This may indicate a problem with the MARCI-based timing improvements.')
    
    ######################################
    # Replace Ls and store original
    df.rename(columns={ls_col: f'{ls_col}_orig'}, inplace=True)
    df[ls_col] = np.array(ls_precise)

    logger.info(f'{my_diffs} storms have a different Mars year than the original data file.')
    logger.info(f'Ls values overwritten with precise values from MARCI swath metadata. Datetime (of swath start) also added.')
    
    return df


def add_mars_sols(df, n_jobs, dt_col='dt', verbose=True):
    """Helper to augment a database of storms with MarsTime information"""
    df = df.copy()
    
    def calc_sol(dt):
        return datetime_to_marstime(dt).sol

    with parallel_config(n_jobs=n_jobs, verbose=verbose):
        storm_sols = Parallel()(delayed(calc_sol)(row[dt_col]) for _, row in df.iterrows())
    
    df['sol'] = storm_sols

    # Drop decimal of sol. Should be equivalent to simple `.astype(int)` cast since values here are all positive, 
    # but keeping consistence with relative sol calcs
    df['sol_int'] = np.floor(storm_sols).astype(int)  
    
    return df


def add_onset_end_and_merger_columns(df: pd.DataFrame, storm_id_col='StormID', dt_col='dt'):
    """
    Storm IDs persist over multiple sols if storm continues.
    Merged storms have '+' in name joining merged Storm IDs.
    If row is first instance of Storm ID that isn't a merged storm, mark as onset.
    If it's a merged storm, mark as merger onset
    """
    df = df.copy()
    onset_merger_mapping = df.groupby(storm_id_col)[dt_col].min() # earliest time of storm
    end_merger_mapping = df.groupby(storm_id_col)[dt_col].max()  # latest time of storm

    df["storm_onset"] = df.apply(
        lambda row: 1 if (
            row[dt_col] == onset_merger_mapping.loc[row[storm_id_col]]
        ) and not (
            "+" in row[storm_id_col]
        ) else 0,
        axis=1
    )
    df["merger_onset"] = df.apply(
        lambda row: 1 if (
            row[dt_col] == onset_merger_mapping.loc[row[storm_id_col]]
        ) and (
            "+" in row[storm_id_col]
        ) else 0,
        axis=1
    )
    df["storm_end"] = df.apply(
        lambda row: 1 if (
            row[dt_col] == end_merger_mapping.loc[row[storm_id_col]]
        ) and not does_this_storm_merge(
            df, row[storm_id_col], storm_id_col=storm_id_col
        ) else 0,
        axis=1
    )

    return df


def does_this_storm_merge(df, storm_id, storm_id_col='StormID'):
    """Check if this storm merges."""
    try:
        storm_prefix, individual_ids = storm_id.split("_")
    except ValueError as e:
        # A few storms have IDs that I don't understand, ignore for now
        return None
    if "+" in individual_ids:
        all_members_in_this_storm = individual_ids.split("+")  # get individual storm names
    else:
        all_members_in_this_storm = [individual_ids]  # only one storm
    same_prefix = [x for x in df[storm_id_col].unique() if storm_prefix+"_" in x and x!=storm_id]  # all other storms with same prefix
    future_merge = [
        x for x in same_prefix if [
            s for s in all_members_in_this_storm if s in x.split("_")[1]
        ]
    ]  # has same prefix and number shows up in merger
    if len(future_merge) >0:
        return 1
    else:
        return 0


def add_cumulative_sols_column(df: pd.DataFrame, storm_id_col='StormID', dt_col='dt'):
    """
    For each storm member, determine how many sols have elapsed
    since the start or merger of that storm.
    """
    df = df.copy()
    first_time_of_storm = df.groupby(storm_id_col)[dt_col].min()
    df["sols_since_onset/merger"] = df.apply(
        lambda row: sols_elapsed(row[dt_col], first_time_of_storm.loc[row[storm_id_col]]),
        axis=1
    ).round()

    return df


def centroid_distance_traveled_for_single_storm(storm_df: pd.DataFrame, dt_col='dt'):
    """
    For single storm with multiple members, determine how far the centroid moved
    (per sol) from the previous member of the same storm
    """
    storm_df_sol_prior = storm_df.shift(1)
    time_between_observations = storm_df.apply(lambda row: sols_elapsed(row[dt_col], storm_df_sol_prior.loc[row.name, dt_col]), axis=1)
    distance_traveled = storm_df.apply(
        lambda row: haversine_dist(
            row["lat"],
            row["lon"], 
            storm_df_sol_prior.loc[row.name, "lat"], 
            storm_df_sol_prior.loc[row.name, "lon"], 
            radius=MARS_GEOID_A/1000  # Geoid radius in kilometers
        ), axis=1)/time_between_observations
    
    return distance_traveled


def area_change_for_single_storm(storm_df: pd.DataFrame, dt_col='dt'):
    """
    For a single storm with multiple members, determine the absolute
    change in area (per sol) since the last member
    """
    storm_df_sol_prior = storm_df.shift(1)
    time_between_observations = storm_df.apply(lambda row: sols_elapsed(row[dt_col], storm_df_sol_prior.loc[row.name, dt_col]), axis=1)
    area_increase = (storm_df["area"]-storm_df_sol_prior["area"])/time_between_observations
    
    return area_increase


# TODO: Need to figure out how to handle values where date is <1 sol
def add_sol_by_sol_metrics(df: pd.DataFrame, storm_id_col='StormID', dt_col='dt'):
    """
    Add all distance and area metrics for strom members across
    the full storm DF.
    """
    df = df.copy()
    distance_traveled_values, area_increase_values = [], []
    for n, g in df.groupby(storm_id_col):

        dist_traveled = centroid_distance_traveled_for_single_storm(g, dt_col)
        dist_traveled = dist_traveled.replace([np.inf, -np.inf], 0)
        distance_traveled_values.append(dist_traveled)

        temp_area_change = area_change_for_single_storm(g, dt_col)
        temp_area_change = temp_area_change.replace([np.inf, -np.inf], 0)
        area_increase_values.append(temp_area_change)

    distance_traveled_values = pd.concat(distance_traveled_values)
    area_increase_values = pd.concat(area_increase_values)
    df["centroid_distance_per_sol"] = distance_traveled_values
    df["area_increase_per_sol"] = area_increase_values
    df["fractional_area_increase_per_sol"] = df["area"]/(df["area"] - df["area_increase_per_sol"])
    r_eff = np.sqrt((df["area"] - df["area_increase_per_sol"])/np.pi)
    df["centroid_distance_per_sol_reff"] = df["centroid_distance_per_sol"]/r_eff
    
    return df


def find_centermost_profiles_per_sol(atmospheric_df, storm_center_lat, storm_center_lon, mcs_ddr1_latlon="Surf"):
    """
    Find the profile closest to the storm center for each relative sol.
    
    Parameters
    ----------
    atmospheric_df : pd.DataFrame
        DataFrame containing atmospheric data with Profile_lat, Profile_lon,
        rel_sol_int, and Profile_identifier columns
    storm_center_lat : float
        Latitude of the storm center
    storm_center_lon : float
        Longitude of the storm center
    
    Returns
    -------
    dict
        Dictionary mapping rel_sol_int to Profile_identifier for the centermost profile
    """
    # Get coordinates from DDR2 dataframe (helps to drop duplicates since we have per-level information here)
    profile_coords = (atmospheric_df
                      .dropna(subset=[f'{mcs_ddr1_latlon}_lat', f'{mcs_ddr1_latlon}_lon'])
                      .groupby(['rel_sol_int', 'Profile_identifier'])
                      .first()[[f'{mcs_ddr1_latlon}_lat', f'{mcs_ddr1_latlon}_lon']]
                      .reset_index())
    
    centermost_profiles = {}
    
    for rel_sol in profile_coords['rel_sol_int'].unique():
        sol_profiles = profile_coords[profile_coords['rel_sol_int'] == rel_sol]
        
        # Compute distance in radians. Find shortest angle
        rad_distances = haversine_dist(storm_center_lat, storm_center_lon,
                                       sol_profiles[f'{mcs_ddr1_latlon}_lat'].values,
                                       sol_profiles[f'{mcs_ddr1_latlon}_lon'].values)
        
        min_idx = rad_distances.argmin()
        centermost_profiles[rel_sol] = sol_profiles.iloc[min_idx]['Profile_identifier']
    
    return centermost_profiles
