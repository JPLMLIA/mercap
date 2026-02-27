from pathlib import Path

import click
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import sqlalchemy
from sqlalchemy import String, Float, Integer, Boolean
from shapely.geometry import Point, Polygon
from geoalchemy2 import Geometry
import geopandas as gpd
import shapely

from mars_time import marstime_to_datetime, MarsTime
from mercap import config

from mercap.utils.storm_data_proc import (read_mdssd_idl, get_polygon_properties, 
                                          get_binary_mask_polygon,
                                          write_storms_csv, write_storms_db,
                                          poly_fix_antimeridian_crossing, find_intersecting_storms,
                                          load_marci_cumindex, apply_marci_timing_precision,
                                          add_mars_sols, add_onset_end_and_merger_columns,
                                          add_cumulative_sols_column, add_sol_by_sol_metrics)
from mercap.utils.db_utils import sqlalchemy_engine_check
from mercap.utils.util import get_logger
from mercap.utils.viz_utils import plot_polygon_list


# Rename columns to be clearer, avoid capital letters to match SQL convention
COLUMNS_RENAME = {'MDGM': 'mdgm',
                  'Year': 'mars_year',
                  'Year_orig': 'mars_year_orig',
                  'Ls': 'ls',
                  'Ls_orig': 'ls_orig',
                  'StormID': 'storm_id',
                  'SeqID': 'seq_id',
                  'Conflev': 'conflev'}

ERODE_DILATE_DURING_POLYGON_EXTRACTION = True


@click.command()
@click.option('--mdssd_head_dir', type=click.Path(exists=True), required=True, help='Path to the MDSSD head directory.')
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
def cli(mdssd_head_dir, poly_extraction_epsilon, mdgm_arr_size, mdgm_offset, 
        csv_output_fpath, db_url, marci_lists_dir, marci_cumindex_fpath, log_fpath, n_jobs, smoke_test, gen_binary_masks):

    # Set up logging with user-specified log file
    logger = get_logger(log_fpath)

    # Ensure we have a live PostGRES database POSTGIS MUST BE ENABLED. 
    #     If it isn't, connect to the database of interest and enter `CREATE EXTENSION postgis;` in the terminal
    if db_url is not None:
        sqlalchemy_engine_check(db_url)

    mdssd_df = pd.read_csv(mdssd_head_dir / Path('mdssd.csv'), sep='\s+', engine='python')

    ######################################
    # Apply smoke_test filter if enabled (limit to 10 unique MDGMs)
    if smoke_test:
        unique_mdgms = pd.unique(mdssd_df['MDGM'])[:10]
        mdssd_df = mdssd_df[mdssd_df['MDGM'].isin(unique_mdgms)]
        logger.info(f'Smoke test enabled: Limited to {len(unique_mdgms)} unique MDGMs')

    ######################################
    # If desired, calculate precise Ls values using MARCI data and replace the Ls column
    if config.IMPROVE_MARCI_TIMING_PRECISION:
        cumindex_df = load_marci_cumindex(marci_cumindex_fpath)
        mdssd_df = apply_marci_timing_precision(mdssd_df, marci_lists_dir, cumindex_df, n_jobs,
                                                mdgm_col='MDGM', mars_year_col='Year',
                                                ls_col='Ls', dt_col='dt', logger=logger)
    else:
        # Create placeholder if not using MARCI timing precision
        mdssd_df['orbit_num'] = -1
        
    ######################################
    # Add Mars time information

    logger.info('Adding Mars sols from datetime')
    # Store original sol if it exists
    if 'sol' in mdssd_df.columns:
        mdssd_df.rename(columns={'sol': 'sol_orig'}, inplace=True)
    mdssd_df = add_mars_sols(mdssd_df, n_jobs, dt_col='dt')

    logger.info('Adding onset, end, merger, and cumulative sols timing information to storms')
    mdssd_df = add_onset_end_and_merger_columns(mdssd_df, dt_col='dt')
    mdssd_df = add_cumulative_sols_column(mdssd_df, dt_col='dt')

    logger.info('Adding distance and area metrics to storms')
    mdssd_df = add_sol_by_sol_metrics(mdssd_df, dt_col='dt')
    
    ######################################
    # Reorder columns to place timing columns next to each other near start of df (before column renaming)
    target_cols = ['Year_orig', 'Year', 'Ls', 'Ls_orig', 'sol', 'sol_orig', 'sol_int']
    existing_target_cols = [col for col in target_cols if col in mdssd_df.columns]
    
    if existing_target_cols:
        cols = list(mdssd_df.columns)
        cols_reordered = [c for c in cols if c not in existing_target_cols]
        for col in reversed(existing_target_cols):
            cols_reordered.insert(2, col)
        mdssd_df = mdssd_df[cols_reordered]

    ###################################
    logger.info('Adding geometry information')
    storm_center_points = mdssd_df.apply(lambda row: Point([row['lon'], row['lat']]), axis=1)
    mdssd_df['storm_center_point'] = [pt.wkt for pt in storm_center_points]

    # Initialize the storm polygon series and geometry validation flag
    mdssd_df['storm_polygon'] = pd.Series([''] * len(mdssd_df))
    mdssd_df['geometry_valid'] = pd.Series([None] * len(mdssd_df))

    # Initialize lists for error tracking
    polygon_error_cases = []
    multi_match_error_cases = []
    no_match_error_cases = []
    shape_abnormalities = []
    lost_mask_error_cases = []

    # Set how many MDGMs to track
    mdgm_names = pd.unique(mdssd_df['MDGM'])
    total_mdgms = len(mdgm_names)
    storm_count = 0

    # Process each storm row in the dataframe to extact the mask and store as a polygon
    for mdgm in tqdm(mdgm_names, total=total_mdgms, desc='Analyzing MDSSD MDGMs'):

        # Extract MDGM name code
        phase_letter, phase, day = mdgm[0], mdgm[:3], mdgm[3:]

        # Get the filepath to the IDL storm mask file and load it
        fpath_stub = Path(f'data_{phase_letter}/{phase}/{phase}_{day}_roi.sav')
        fpath = mdssd_head_dir / fpath_stub
        idl_output, _ = read_mdssd_idl(fpath)
        
        #############################
        # Debugging code: save out image so we can visualize against MDGMs
        if gen_binary_masks:
            logger.info('Processing: %s', fpath)
            # Save as image
            from PIL import Image
            binary_mask = np.zeros(mdgm_arr_size, dtype=bool)
            for idl_obj in idl_output:
                # Initialize binary mask, and set all pixels at the (X, Y) points to True
                binary_mask[mdgm_arr_size[0] - idl_obj.roiy[0] - 1, idl_obj.roix[0] - 1] = True
             
            uint8_mask = np.uint8(binary_mask * 255)
            pil_img = Image.fromarray(uint8_mask, mode='L')
            pil_img.save(fpath.parent / Path(fpath.stem + '.png'))
             
            # Apply the morphological ops to deal with small, erroneous storm appendages
            kernel = np.ones((3, 3), np.uint8)
            mask_fixed = cv2.dilate(cv2.erode(uint8_mask, kernel), kernel)
            pil_img = Image.fromarray(mask_fixed, mode='L')
            pil_img.save(fpath.parent / Path(fpath.stem + '_morphOps.png'))

        #############################

        # Loop over all storm masks identified for this MDGM, convert to polygons, and save in the database
        object_polygons = []
        for idl_obj in idl_output:

            storm_id = idl_obj.storm_id[0].decode()

            obj_seq_id = 'None'
            if idl_obj.sequence_id[0] != '':
                obj_seq_id = idl_obj.sequence_id[0].decode()

            # Initialize binary mask, and set all pixels at the (X, Y) points to True
            binary_mask = np.zeros(mdgm_arr_size, dtype=bool)
            binary_mask[mdgm_arr_size[0] - idl_obj.roiy[0] - 1, idl_obj.roix[0] - 1] = True  # Convert x,y to row,col and move origin to TL

            storm_polygon = get_binary_mask_polygon(binary_mask, (0, 0), poly_extraction_epsilon, 
                                                    erode_dilate=ERODE_DILATE_DURING_POLYGON_EXTRACTION, logger=logger)

            if storm_polygon is None:
                logger.error(f'Storm found with no positive pixels. MGDM: {mdgm}, StormID: {storm_id}, SeqID: {obj_seq_id}.')
                lost_mask_error_cases.append(f'MGDM: {mdgm}, StormID: {storm_id}, SeqID: {obj_seq_id}')
                continue
            
            # Converting from OpenCV coordinates (0, 0) at top left, and in x, y
            storm_polygon[:, 1] = mdgm_offset[1] + storm_polygon[:, 1] * -1
            storm_polygon[:, 0] += mdgm_offset[0]

            # TODO: Incorporate these shape properties eventually
            #polygon_properties = get_polygon_properties(storm_polygon)

            # Skip invalid polygons
            if len(storm_polygon) < 4:
                polygon_error_cases.append(f'MGDM: {mdgm}, StormID: {storm_id}, SeqID: {obj_seq_id}')
                logger.warning('Recieved polygon with <4 points on file: %s', fpath)
                continue

            #######
            # Generate storm outline as shapely `Polygon` types. Each pixel represents 0.1 degrees
            storm_polygon_shapely = Polygon(0.1 * storm_polygon)
            # Deal with antimeridian splits. Long values > 180 need to have 360 deg subtracted to wrap
            storm_polygon_shapely = poly_fix_antimeridian_crossing(storm_polygon_shapely)

            # Detailed geometry validation
            is_valid = storm_polygon_shapely.is_valid
            is_simple = storm_polygon_shapely.is_simple
            
            if not is_valid or not is_simple:
                validity_reason = shapely.validation.explain_validity(storm_polygon_shapely)
                logger.error(f'Shape abnormality on MGDM: {mdgm}, StormID: {storm_id}, SeqID: {obj_seq_id}. '
                           f'is_valid: {is_valid}, is_simple: {is_simple}, reason: {validity_reason}')
            
            # Additional geometry checks
            if is_valid and storm_polygon_shapely.has_z:
                logger.warning(f'Polygon has Z coordinates (unexpected) for MGDM: {mdgm}, StormID: {storm_id}, SeqID: {obj_seq_id}.')

            object_polygons.append(storm_polygon_shapely)
            storm_polygon_wkt = shapely.wkt.dumps(storm_polygon_shapely, trim=True, rounding_precision=5)  # Reduceds artificial ballooning of decimal places
            if len(storm_polygon_wkt) > 32768:
                logger.warning('Polygon character string exceeds excel\'s loading limit.')
            ########
            # Find the right row index in our dataframe to update with a polygon
            # Match storm ID to IDL data. Some storm names were truncated, so attempt to adapt to that
            storm_id_match_mask = (mdssd_df['StormID'] == storm_id).to_numpy()
            if np.sum(storm_id_match_mask) == 0:
                shortened_storm_id = storm_id[:15]  # Some rows in the MDSSD spreadsheet shortened to 15 characters
                storm_id_match_mask = (mdssd_df['StormID'] == shortened_storm_id).to_numpy()
            
            # Correct error in IDL files where there are 0 conflev storms that should've been 1
            conflev = idl_obj.conflev[0]
            if conflev == 0:
                logger.error(f'Found storm with conflev==0 on MGDM: {mdgm}, StormID: {storm_id}, SeqID: {obj_seq_id}.')
                conflev = 1

            # Begin constructing matching information to identify the MDSSD CSV row corresponding to this storm
            # s_extent = storm_polygon_shapely.bounds
            # location_match_mask = np.array([(s_extent[0] <= pt.x <= s_extent[2]) and
            #                                 (s_extent[1] <= pt.y <= s_extent[3] )
            #                                  for pt in storm_center_points])
  
            # Find the corresponding row
            row_sel_criteria = \
                (mdssd_df['MDGM'] == mdgm) & \
                (storm_id_match_mask) & \
                (mdssd_df['SeqID'] == obj_seq_id) & \
                (mdssd_df['Conflev'] == conflev) #& \
                #(location_match_mask)
                
            # TODO: dump errors to a processing log
            if np.sum(row_sel_criteria) > 1:
                logger.error(f'Found multiple matching rows for MGDM: {mdgm}, StormID: {storm_id}, SeqID: {obj_seq_id}.')
                multi_match_error_cases.append(f'MGDM: {mdgm}, StormID: {storm_id}, SeqID: {obj_seq_id}')
            elif np.sum(row_sel_criteria) == 0:
                logger.error(f'Found no matching rows for MGDM: {mdgm}, StormID: {storm_id}, SeqID: {obj_seq_id}.')
                no_match_error_cases.append(f'MGDM: {mdgm}, StormID: {storm_id}, SeqID: {obj_seq_id}')
            if not is_valid or not is_simple:
                shape_abnormalities.append(f'MGDM: {mdgm}, StormID: {storm_id}, SeqID: {obj_seq_id}')
                
            mdssd_df.loc[row_sel_criteria, 'storm_polygon'] = storm_polygon_wkt
            mdssd_df.loc[row_sel_criteria, 'geometry_valid'] = is_valid and is_simple
            storm_count += 1
            
        polygons_fig_save_fpath = fpath.parent / Path(fpath.stem + '_polygons.png')
        plot_polygon_list(object_polygons, polygons_fig_save_fpath)

    logger.info(f'Polygon errors: {len(polygon_error_cases)}:\n{polygon_error_cases}\n\n')
    logger.info(f'Multi match errors: {len(multi_match_error_cases)}:\n{multi_match_error_cases}\n\n')
    logger.info(f'No match errors: {len(no_match_error_cases)}:\n{no_match_error_cases}\n\n')
    logger.info(f'Shape abnormality errors: {len(shape_abnormalities)}:\n{shape_abnormalities}\n\n')
    logger.info(f'Lost mask errors (after CV morphological ops): {len(lost_mask_error_cases)}:\n{lost_mask_error_cases}\n\n')

    ######################################
    # Handle any remaining empty polygons by filling them with EMPTY
    mdssd_df['storm_polygon'].fillna('POLYGON EMPTY', inplace=True)
    mdssd_df.loc[mdssd_df['storm_polygon'] == '', 'storm_polygon'] = 'POLYGON EMPTY'
    
    n_empty_polygons = len(mdssd_df.loc[mdssd_df['storm_polygon'] == 'POLYGON EMPTY'])
    if n_empty_polygons:
        logger.warning('%d storms (of %d total) do not have a geospatial polygon', n_empty_polygons, len(mdssd_df))

    ######################################
    # Find all storm polygons that intersect with other storms from different years
    # This'll help exclude matched control profiles that are actually part of a separate storm
    intersections = find_intersecting_storms(mdssd_df, temporal_overlap_size=int(np.ceil(config.controls['collage_controls_smoothing_window_size'] / 2)))
    mdssd_df = pd.concat([mdssd_df, intersections], axis=1)

    ######################################
    # Rename columns to be clearer, avoid capital letters to match SQL convention
    mdssd_df.rename(columns=COLUMNS_RENAME, inplace=True)

    ######################################
    # Write to CSV and database if desired
    if csv_output_fpath:
        write_storms_csv(mdssd_df, csv_output_fpath)
    if db_url:
        column_types = {'mdgm': String,
                        'mars_year': Integer,
                        'sol': Float,
                        'sol_int': Integer,
                        'ls': Float,
                        'storm_id': String,
                        'seq_id': String,
                        'lon': Float,
                        'lat': Float,
                        'area': Float,
                        'conflev': Integer,
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
        if 'mars_year_orig' in mdssd_df.columns:
            column_types['mars_year_orig'] = Integer
        if 'ls_orig' in mdssd_df.columns:
            column_types['ls_orig'] = Float
        if 'sol_orig' in mdssd_df.columns:
            column_types['sol_orig'] = Float

        # XXX: Occasionally, this will hang. If it does, restart. The DB insertion for 10k storms should only take ~10s.
        write_storms_db(mdssd_df, 'mdssd', db_url, column_types, db_existing_behavior='replace')

if __name__ == '__main__':
    cli()
