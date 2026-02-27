from pathlib import Path

import re
import numpy as np
import pandas as pd
import click
import shapely
import cv2
from shapely import wkt
from shapely.geometry import Point, Polygon
import sqlalchemy
from sqlalchemy import String, Float, Integer, Boolean
from geoalchemy2 import Geometry
import netCDF4 as nc
from pyproj import Geod

from mars_time import marstime_to_datetime, datetime_to_marstime, MarsTime
from mercap import config
from mercap.utils.storm_data_proc import (write_storms_db, write_storms_csv, 
                                          get_binary_mask_polygon, get_polygon_properties,
                                          poly_fix_antimeridian_crossing, find_intersecting_storms,
                                          load_marci_cumindex, apply_marci_timing_precision,
                                          add_mars_sols, add_onset_end_and_merger_columns,
                                          add_cumulative_sols_column, add_sol_by_sol_metrics)
from mercap.utils.db_utils import sqlalchemy_engine_check
from mercap.utils.util import get_logger
from mercap.utils.viz_utils import plot_polygon_list


# Whether or not to do an erode/dilate cycle on the binary masks (recommended to keep True)
ERODE_DILATE_DURING_POLYGON_EXTRACTION = True


def parse_mem_id_to_mask_value(mem_id):
    """
    Parse a mem_ID string to extract the mask integer value from the first storm.

    Parameters
    ----------
    mem_id : str
        Member ID string from MDAD CSV. Can be simple (e.g., "D01_031"),
        have a 'b' suffix (e.g., "D16_001b"), or have multiple combined
        storms (e.g., "D01_031+032", "P08_033+P09_010+011+013").
        Only extracts the value from the first part before any '+'.

    Returns
    -------
    int or None
        The uint8 mask value from the first storm, or None if parsing fails.

    Examples
    --------
    >>> parse_mem_id_to_mask_value("D01_031")
    31
    >>> parse_mem_id_to_mask_value("D16_001b")
    134
    >>> parse_mem_id_to_mask_value("D01_031+032")
    31
    >>> parse_mem_id_to_mask_value("P08_033+P09_010+011+013")
    33
    """

    first_part = mem_id.split('+')[0]
    
    # Check if part has full format (e.g., "P09_010" or "D16_001b"): underscore + digits + optional 'b' at end
    match_full = re.search(r'_(\d+)(b)?$', first_part)
    
    if match_full:
        value = int(match_full.group(1))
        if match_full.group(2) == 'b':
            value += 133
        return value
    
    return None


@click.command()
@click.option('--mdad_head_dir', type=click.Path(exists=True), required=True, help='Path to the MDAD head directory.')
@click.option('--mdad_csv_fpath', type=click.Path(exists=True), required=True, help='Path to the MDAD master CSV file.')
@click.option('--poly_extraction_epsilon', type=float, default=0.00001, help='Polygon extraction epsilon.')
@click.option('--mdgm_arr_size', type=(int, int), default=(1801, 3600), help='MDGM array size (rows/cols).')
@click.option('--mdgm_offset', type=(int, int), default=(-1801, 900), help='MDGM offset (x,y) for converting to lon/lat.')
@click.option('--csv_output_fpath', type=click.Path(), default=None, help='File path to write the CSV output.')
@click.option('--db_url', type=str, default=None, help='Database URL path if saving to PostGRES+PostGIS.')
@click.option('--marci_lists_dir', type=click.Path(exists=True), default=None, help='Path to directory containing MARCI lists for each phase.')
@click.option('--marci_cumindex_fpath', type=click.Path(exists=True), default=None, help='Path to MARCI cumindex.tab file.')
@click.option('--log_fpath', type=click.Path(), default=None, help='Path to log file for saving processing output.')
@click.option('--n_jobs', type=int, default=1, help='Number of workers to use for parallelized portions of code.')
@click.option('--smoke_test', is_flag=True, help='Enable smoke test to debug by processing a much smaller portion of data.')
@click.option('--gen_binary_masks', is_flag=True, help='Save out masks of individual storms.')
def cli(mdad_head_dir, mdad_csv_fpath, poly_extraction_epsilon, mdgm_arr_size, mdgm_offset, 
        csv_output_fpath, db_url, marci_lists_dir, marci_cumindex_fpath, log_fpath, n_jobs, smoke_test, gen_binary_masks):
    
    logger = get_logger(log_fpath)
    
    if db_url is not None:
        sqlalchemy_engine_check(db_url)

    #######################################
    # Read MDAD files and convert to pandas
    logger.info(f'Reading MDAD data from {mdad_head_dir}')
    logger.info(f'Reading MDAD master CSV from {mdad_csv_fpath}')
    
    mdad_metadata_df = pd.read_csv(mdad_csv_fpath, engine='python')
    logger.info(f'Loaded {len(mdad_metadata_df)} rows from MDAD main CSV')
    
    # Initialize lists for error tracking
    polygon_error_cases = []
    shape_abnormalities = []
    lost_mask_error_cases = []
    antimeridian_error_cases = []
    precision_Ls_mismatch = []
    
    # Compile the subdirs (each subdir is one MY)
    temp_storm_list = []
    my_subdirs = sorted([x for x in Path(mdad_head_dir).iterdir() if 'MY' in str(x) and x.is_dir()])
    total_subdirs = 1 if smoke_test else len(my_subdirs)
 
    for data_subdir in my_subdirs[:total_subdirs]:

        fname = f'MDAD{data_subdir.stem}.nc'
        fpath = data_subdir / Path(fname)
        mars_year = int(data_subdir.stem[2:4])
        logger.info(f'\n\nProcessing {fname}')

        # Process NetCDF directly instead of loading all NetCDFs into memory at once
        with nc.Dataset(fpath, 'r') as dataset:
            n_soy = len(dataset.variables['soy'])
            
            # Loop over each sol of the year (soy)
            for soy_idx in range(n_soy):
                # Extract data for this soy
                mdad_mask = dataset.variables['MDAD'][soy_idx, :, :]
                mdgm_id = dataset.variables['dayList'][soy_idx]  # MDGM ID in format XXX_dayZZ, where XXX is the MARCI subphase and ZZ is the sol in that subphase
                subphase = mdgm_id.split('_')[0]
                subphase_sol = int(mdgm_id.split('_')[1][3:]) # Remove 'day' from the sol string

                # TODO: Use 'soy', 'mean soy', 'median soy'?
                mdad_soy = float(dataset.variables['soy'][soy_idx])
                mdad_time_vals = dataset.variables['timeList'][soy_idx, :]
                mdad_dt = marstime_to_datetime(MarsTime(mars_year, mdad_soy))
                
                if mdad_mask.mask:
                    raise ValueError(f'MDAD is masked for soy: {mdad_soy}. Need to understand if the mask is used.')
                
                # TODO: Verify that mask is in lon/lat (3600/1801) and needs transpose/flip to row/col
                # Original shape: (3600, 1801); Original lon: [-180, 179.9]; Original lat:  [-90, 90]
                # New lon: (1801, 3600); New lon: [-180, 179.9]; New lat: [90, -90]
                mdad_mask = np.flip(np.transpose(mdad_mask), axis=0)
                
                # Using median Ls. The mean Ls has issues for the few storms wrapping around the Mars Year in MDAD v1.1
                median_ls = float(mdad_time_vals[7])

                #############################
                # Debugging code: save out image so we can visualize
                if gen_binary_masks:
                    from PIL import Image

                    binary_mask = np.logical_and(mdad_mask > 0, mdad_mask < 255)
                 
                    # This requires that the 'masks' folder is created under each MY
                    uint8_mask = np.uint8(binary_mask * 255)
                    pil_img = Image.fromarray(uint8_mask, mode='L')
                    
                    if not Path(fpath.parent / Path('masks')).exists():
                        Path(fpath.parent / Path('masks')).mkdir(parents=True, exist_ok=True)
                    pil_img.save(fpath.parent / Path('masks') / Path(f'{fpath.stem}_{mdgm_id}.png'))

                    # Apply the morphological ops to deal with small, erroneous storm appendages
                    kernel = np.ones((3, 3), np.uint8)
                    mask_fixed = cv2.dilate(cv2.erode(uint8_mask, kernel), kernel)
                    pil_img = Image.fromarray(mask_fixed, mode='L')
                    pil_img.save(fpath.parent / Path('masks') / Path(f'{fpath.stem}_{mdgm_id}_morphOps.png'))

                ############################# 
                # Find all CSV rows matching this MDGM (by subphase and sol)
                csv_rows_mask = ((mdad_metadata_df['subphase'] == subphase) &
                                 (mdad_metadata_df['sol'] == subphase_sol))
                matched_rows = mdad_metadata_df[csv_rows_mask]
                object_polygons = []
                
                # Loop over each storm row in the CSV for this MDGM
                for _, row in matched_rows.iterrows():
                    mem_id = row['mem_ID']
                    mask_value = parse_mem_id_to_mask_value(mem_id)
                    if mask_value is None:
                        logger.error(f'Error parsing mem_id: {mem_id}')
                        lost_mask_error_cases.append(f'MDGM: {mdgm_id}, mem_id: {mem_id}')
                        continue

                    logger.debug(f'Processing mem_id: {mem_id} (mask value: {mask_value}) in {mdgm_id}')

                    # Generate binary mask for the mask value
                    binary_mask = (mdad_mask == mask_value)
                    if np.sum(binary_mask) == 0:
                        logger.error(f'Binary mask is all zeros for MDGM: {mdgm_id}, mem_id: {mem_id}, mask_val: {mask_value}.')
                        lost_mask_error_cases.append(f'MDGM: {mdgm_id}, mem_id: {mem_id}, mask_val: {mask_value}')
                        continue

                    # Extract row properties directly
                    seq_id = row['seq_ID'] if pd.notna(row['seq_ID']) else 'None'
                    confidence = int(row['confidence_flag'])
                    conflev = config.MDAD_CONFLEV_CONVERSION[confidence]
                    csv_area = float(row['area'])
                    csv_swath_ls = float(row['swathLs'])

                    # Ls precision check
                    if not abs(csv_swath_ls - median_ls) < 0.5 and not abs(csv_swath_ls - median_ls) > 359.5:
                        logger.warning(f'Ls precision mismatch for MDGM: {mdgm_id}, mem_id: {mem_id}. '
                                       f'CSV swathLs: {csv_swath_ls}, median_ls: {median_ls}')
                        precision_Ls_mismatch.append(f'MDGM: {mdgm_id}, mem_id: {mem_id}, '
                                                     f'swathLs: {csv_swath_ls}, median_ls: {median_ls}')


                    storm_polygon = get_binary_mask_polygon(binary_mask, (0, 0), poly_extraction_epsilon,
                                                            erode_dilate=ERODE_DILATE_DURING_POLYGON_EXTRACTION,
                                                            logger=logger)

                    if storm_polygon is None:
                        logger.error(f'Storm found with no positive pixels. MDGM: {mdgm_id}, mem_id: {mem_id}, mask_val: {mask_value}.')
                        lost_mask_error_cases.append(f'MDGM: {mdgm_id}, mem_id: {mem_id}, mask_val: {mask_value}')
                        continue

                    # Skip invalid polygons
                    if len(storm_polygon) < 4:
                        polygon_error_cases.append(f'MDGM: {mdgm_id}, mem_id: {mem_id}, mask_val: {mask_value}')
                        logger.warning(f'Received polygon with <4 points for MDGM: {mdgm_id}, mem_id: {mem_id}, mask_val: {mask_value}')
                        continue
                    
                    # TODO: Incorporate these shape properties eventually
                    # polygon_properties = get_polygon_properties(storm_polygon)
                        
                    #########
                    # Add any user-specified offsets
                    # New lon: [-180, 179.9]; New lat: [90, -90]
                    storm_polygon[:, 1] = mdgm_offset[1] + storm_polygon[:, 1] * -1
                    storm_polygon[:, 0] += mdgm_offset[0]

                    # Generate storm outline as shapely Polygon. Each pixel represents 0.1 degrees
                    storm_polygon_shapely = Polygon(0.1 * storm_polygon)

                    # Deal with antimeridian splits
                    storm_polygon_shapely = poly_fix_antimeridian_crossing(storm_polygon_shapely)

                    if storm_polygon_shapely.is_empty:
                        logger.error(f'Polygon became empty after antimeridian fix for MDGM: {mdgm_id}, mem_id: {mem_id}, mask value: {mask_value}.')
                        antimeridian_error_cases.append(f'MDGM: {mdgm_id}, mem_id: {mem_id}, mask value: {mask_value}.')
                        continue

                    # Track the shapely polygons for saving out as images
                    object_polygons.append(storm_polygon_shapely)

                    # Detailed geometry validation
                    is_valid = storm_polygon_shapely.is_valid
                    is_simple = storm_polygon_shapely.is_simple
                    validity_reason = None
                    
                    if not is_valid or not is_simple:
                        validity_reason = shapely.validation.explain_validity(storm_polygon_shapely)
                        logger.error(f'Shape abnormality on MDGM: {mdgm_id}, mem_id: {mem_id}, mask value: {mask_value}. '
                                   f'is_valid: {is_valid}, is_simple: {is_simple}, reason: {validity_reason}')
                        shape_abnormalities.append(f'MDGM: {mdgm_id}, mem_id: {mem_id}, mask value: {mask_value}, reason: {validity_reason}')
                    
                    # Additional geometry checks
                    if is_valid and storm_polygon_shapely.has_z:
                        logger.warning(f'Polygon has Z coordinates (unexpected) for MDGM: {mdgm_id}, mem_id: {mem_id}, mask value: {mask_value}.')
                    
                    # Convert to WKT with precision control
                    storm_polygon_wkt = shapely.wkt.dumps(storm_polygon_shapely, trim=True, rounding_precision=5)

                    # Check if polygon exceeds Excel's character limit
                    if len(storm_polygon_wkt) > 32768:
                        logger.warning(f'Polygon character string exceeds excel\'s loading limit for MDGM: {mdgm_id}, mem_id: {mem_id}, mask value: {mask_value}.')

                    ##########
                    temp_storm_list.append({'mdgm': mdgm_id,
                                            'storm_id': f'{mem_id}',
                                            'mars_year': mars_year,
                                            'mask_value': mask_value,
                                            'seq_id': seq_id,
                                            'confidence': confidence,
                                            'conflev': conflev,
                                            'sol': mdad_soy,
                                            'sol_int': np.floor(mdad_soy).astype(int),  # Same operation as storm_data_proc.add_mars_sols used in parse_mdssd.py
                                            'ls': median_ls,
                                            'dt': mdad_dt,
                                            'lon': row['centroid_lon'],
                                            'lat': row['centroid_lat'],
                                            'area': csv_area,
                                            'storm_center_point': Point([row['centroid_lon'], row['centroid_lat']]).wkt,
                                            'storm_polygon': storm_polygon_wkt,
                                            'geometry_valid': is_valid and is_simple})
                    logger.info(f'Finished processing mem_id: {mem_id} in {mdgm_id}')

                if not Path(fpath.parent / Path('masks')).exists():
                    Path(fpath.parent / Path('masks')).mkdir(parents=True, exist_ok=True)

                polygons_fig_save_fpath = fpath.parent / Path('masks') / Path(f'{fpath.stem}_{mdgm_id}_polygons.png')
                plot_polygon_list(object_polygons, polygons_fig_save_fpath)

    mdad_df = pd.DataFrame(temp_storm_list)
    
    # Log error summaries
    logger.info(f'Polygon errors: {len(polygon_error_cases)}:\n{polygon_error_cases}\n\n')
    logger.info(f'Shape abnormality errors: {len(shape_abnormalities)}:\n{shape_abnormalities}\n\n')
    logger.info(f'Lost mask errors (after CV morphological ops): {len(lost_mask_error_cases)}:\n{lost_mask_error_cases}\n\n')
    logger.info(f'Antimeridian fix errors (polygon became empty): {len(antimeridian_error_cases)}:\n{antimeridian_error_cases}\n\n')
    logger.info(f'Ls precision mismatch errors: {len(precision_Ls_mismatch)}:\n{precision_Ls_mismatch}\n\n')

    ######################################
    # If desired, calculate precise Ls values using MARCI data and replace the Ls column
    if config.IMPROVE_MARCI_TIMING_PRECISION:
        cumindex_df = load_marci_cumindex(marci_cumindex_fpath)
        mdad_df = apply_marci_timing_precision(mdad_df, marci_lists_dir, cumindex_df, n_jobs,
                                               mdgm_col='mdgm', mars_year_col='mars_year',
                                               ls_col='ls', dt_col='dt', logger=logger)
    else:
        # Create placeholder if not using MARCI timing precision
        mdad_df['orbit_num'] = -1
    
    ######################################
    # Add Mars time information

    # Add Mars sols from datetime
    logger.info('Adding Mars sols from datetime')
    mdad_df.rename(columns={'sol': 'sol_orig'}, inplace=True)
    mdad_df = add_mars_sols(mdad_df, n_jobs, dt_col='dt')
    
    logger.info('Adding sols, onset, end, merger, and cumulative sols timing information to storms')
    mdad_df = add_onset_end_and_merger_columns(mdad_df, storm_id_col='storm_id', dt_col='dt')
    mdad_df = add_cumulative_sols_column(mdad_df, storm_id_col='storm_id', dt_col='dt')
    logger.info('Adding distance and area metrics to storms')
    mdad_df = add_sol_by_sol_metrics(mdad_df, storm_id_col='storm_id', dt_col='dt')
    
    ######################################
    # Reorder columns to place timing columns next to each other near start of df
    target_cols = ['mars_year_orig', 'mars_year', 'ls', 'ls_orig', 'sol', 'sol_orig', 'sol_int']
    existing_target_cols = [col for col in target_cols if col in mdad_df.columns]
    
    if existing_target_cols:
        cols = list(mdad_df.columns)
        cols_reordered = [c for c in cols if c not in existing_target_cols]
        for col in reversed(existing_target_cols):
            cols_reordered.insert(2, col)
        mdad_df = mdad_df[cols_reordered]

    ######################################
    # Handle any remaining empty polygons
    mdad_df['storm_polygon'].fillna('POLYGON EMPTY', inplace=True)
    mdad_df.loc[mdad_df['storm_polygon'] == '', 'storm_polygon'] = 'POLYGON EMPTY'
    
    n_empty_polygons = len(mdad_df.loc[mdad_df['storm_polygon'] == 'POLYGON EMPTY'])
    if n_empty_polygons:
        logger.warning('%d storms (of %d total) do not have a geospatial polygon', n_empty_polygons, len(mdad_df))

    ######################################
    # Find all storm polygons that intersect with other storms from different years
    logger.info('Finding intersecting storms')
    # TODO: Verify this temporal_overlap_size calculation is appropriate for MDAD data
    intersections = find_intersecting_storms(mdad_df, temporal_overlap_size=int(np.ceil(config.controls['collage_controls_smoothing_window_size'] / 2)),
                                             id_col='storm_id', year_col='mars_year')
    mdad_df = pd.concat([mdad_df, intersections], axis=1)

    ######################################
    # Write to CSV and database if desired
    if csv_output_fpath:
        write_storms_csv(mdad_df, csv_output_fpath)
        
    if db_url:
        column_types = {'mdgm': String,
                        'storm_id': String,
                        'mars_year': Integer,
                        'sol': Float,
                        'sol_int': Integer,
                        'mask_value': Integer,
                        'seq_id': String,
                        'confidence': Integer,
                        'conflev': Integer,
                        'ls': Float,
                        'lon': Float,
                        'lat': Float,
                        'area': Float,
                        'dt': sqlalchemy.TIMESTAMP(timezone=True),  # Timezone info is important to maintain UTC designation (and override with server's local TZ)
                        'orbit_num': Integer,
                        'storm_onset': Boolean,
                        'merger_onset': Boolean,
                        'storm_end': Boolean,
                        'sols_since_onset/merger': Integer, 
                        'centroid_distance_per_sol': Float, 
                        'centroid_distance_per_sol_reff': Float, 
                        'area_increase_per_sol': Float, 
                        'fractional_area_increase_per_sol': Float,
                        'intersecting_storm_ids': String,
                        'intersecting_storm_years': String,
                        'intersecting_storm_sols': String,
                        'storm_center_point': Geometry('POINT'),
                        'storm_polygon': Geometry('MULTIPOLYGON'),
                        'geometry_valid': Boolean}
        if 'mars_year_orig' in mdad_df.columns:
            column_types['mars_year_orig'] = Integer
        if 'ls_orig' in mdad_df.columns:
            column_types['ls_orig'] = Float
        if 'sol_orig' in mdad_df.columns:
            column_types['sol_orig'] = Float

        write_storms_db(mdad_df, 'mdad', db_url, column_types, db_existing_behavior='replace')
        logger.info('MDAD database insertion complete. Added %d storm instances into DB.', len(mdad_df))

if __name__ == '__main__':
    cli()
