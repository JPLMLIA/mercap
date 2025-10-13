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

from mercap.config import INT_SOLS_PER_MY, MARCI_DT_FORMAT


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
        # Iterate over all variable names and read their data
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
    if np.any(lons>180) and np.any(lons<-180):
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

    # TODO: Add geometry validation either here or in the database using some of the Geometry validation tools in
    #    PostGIS: https://postgis.net/docs/reference.html#Geometry_Processing
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
        Name of the column containing years, by default 'Year'
    
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


def calculate_marci_precise_timing(mdssd_df, marci_lists_dir, cumindex_df):
    """
    Calculate precise Ls values for each storm using MARCI image data.
    
    Parameters
    ----------
    mdssd_df : pd.DataFrame
        Storm dataframe containing mdgm and lon columns
    marci_lists_dir : Path
        Directory containing MARCI lists for each phase
    cumindex_df : pd.DataFrame
        Processed cumindex dataframe with image info
        
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
    for _, row in tqdm(mdssd_df.iterrows(), total=len(mdssd_df), desc='Calculating precise Ls values'):
        try:
            # Extract phase and day from MDGM column (e.g., "P12" and "day09" from "P12day09")
            mdgm = row['MDGM']
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
            logging.warning(f'Error calculating ls_precise for {row.get("mdgm", "unknown")}: {e}')
            image_time_precise.append(np.nan)
            ls_precise.append(np.nan)
            orbit_nums.append(np.nan)
 
    return image_time_precise, ls_precise, orbit_nums