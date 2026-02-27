import os.path as op
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import click
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_config
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import mars_time
from mars_time.constants import seconds_per_sol

from mercap.config import N_JOBS, SQL_ENGINE_STRING_MCS13, TIME_SUBPLOT_SOL_RANGE, SOL_WINDOW_BOUNDS
from mercap import config
from mercap.utils.profile_search import (find_ddr1_storm_profile_matches, find_ddr1_control_profile_matches, load_ddr2_from_profile_matches, 
                                         get_storms_with_min_conf, get_storm_metadata_by_database_id, get_storm_metadata_by_storm_id, 
                                         generate_control_exclusions_dict, add_rel_sol_columns)
from mercap.utils.storm_data_proc import find_centermost_profiles_per_sol
from mercap.utils.viz_utils import (gen_plot_collage_1, make_time_profile_plot, make_storm_control_distributions_plot, visualize_controls, 
                                    make_storm_control_profile_comparison_plot)
from mercap.utils.db_utils import get_database_table_info


logging.basicConfig(level=logging.INFO,
                    datefmt='%H:%M:%S',
                    format='%(asctime)s | %(lineno)d | %(levelname)-5s | %(module)-15s | %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(f'storm_collages_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', mode='w')
                    ])


# TODO: ensure both dt's are in UTC
def add_delta_sols_from_storm_column(ddr2_df, storm_row, new_column_name="delta_t"):
    """Helper to compute change in time in sols (as a float)"""
    ddr2_df[new_column_name] = (ddr2_df["dt"] - storm_row["dt"]).dt.total_seconds()/seconds_per_sol
    return ddr2_df


def create_time_profile_subplot(db_url, storm_metadata, data_source, save_png_dir, verbose=False, sol_range=TIME_SUBPLOT_SOL_RANGE):
    """
    For a single storm instance, grab MCS data SOL_RANGE sols before storm through SOL_RANGE sols after
    and plot temp and dust altitude profiles colored by time.
    """
    storm_db_id = storm_metadata['storm_db_id']

    # Parse data source date range from config
    date_start_str, date_end_str = config.dust_database_time_spans[data_source]
    data_source_date_start = datetime.strptime(date_start_str, '%Y-%m-%d')
    data_source_date_end = datetime.strptime(date_end_str, '%Y-%m-%d')

    engine = create_engine(db_url, poolclass=NullPool)
    try:
        with engine.connect() as connection:
            ###############
            # Find intersecting profiles
            if verbose:
                logging.info(f'Storm DB ID {storm_db_id:05}: Extracting MCS profiles from DB') 

            storm_profile_hits = find_ddr1_storm_profile_matches(
                connection, 
                storm_db_id, 
                storm_dt=storm_metadata['dt'],
                time_window_bounds=sol_range,  # Sols before/after
                data_source=data_source,
                ltst_start=config.tolerances['LTST_DAY_START'], 
                ltst_end=config.tolerances['LTST_NIGHT_START'],
                good_obs_quals=config.good_obs_quals,
                data_source_date_start=data_source_date_start,
                data_source_date_end=data_source_date_end,
                verbose=verbose
            )
            if storm_profile_hits.empty:
                logging.warning('No profiles found for storm: %d', storm_db_id)
                return
    
            # Modify MDSSD/MDAD Storm info
            storm_metadata["dt"] = pd.to_datetime(storm_metadata["dt"], utc=True)

            # Add relative timing information
            storm_profile_hits = add_rel_sol_columns(storm_profile_hits, storm_metadata['dt'])
            
            # Pull the MCS profiles and add a column sols before/after time of storm in MDSSD/MDAD
            storm_atmospheric_df = load_ddr2_from_profile_matches(storm_profile_hits, verbose)
            storm_atmospheric_df = add_delta_sols_from_storm_column(
                storm_atmospheric_df,
                storm_metadata
            )
            
            ###############
            # Set up the fpaths for saving plot output
            save_png_fpath = None
            if save_png_dir:
                save_png_fpath = op.join(save_png_dir, f'{storm_metadata["storm_id"]}_MY{storm_metadata["mars_year"]}_sol{storm_metadata["sol_int"]:03}_dbid{storm_db_id:05}.png')
            
            # Generate plot
            make_time_profile_plot(storm_atmospheric_df, storm_metadata, sol_range, save_path=save_png_fpath)

        engine.dispose()
        return 0

    except Exception as e:
        logging.error('Problem with storm DB ID %i during time profile plotting: %s', storm_db_id, e)
        raise e
    
def create_profile_comparison_subplot(db_url, storm_metadata, data_source, save_png_dir, verbose=False):
    """
    For a single storm instance, grab MCS data in 3 time windows
    and plot temp and dust altitude profiles for storms and controls.
    """
    storm_db_id = storm_metadata['storm_db_id']

    # Parse data source date range from config
    date_start_str, date_end_str = config.dust_database_time_spans[data_source]
    data_source_date_start = datetime.strptime(date_start_str, '%Y-%m-%d')
    data_source_date_end = datetime.strptime(date_end_str, '%Y-%m-%d')

    engine = create_engine(db_url, poolclass=NullPool)
    try:
        with engine.connect() as connection:
            ###############
            # Find intersecting profiles
            if verbose:
                logging.info(f'Storm DB ID {storm_db_id:05}: Extracting MCS profiles from DB') 
            storm_time_bounds = [(-9, -3), (-3, 3), (3, 9)]  # trying to ~5-sol orbit pattern
            # Matching profiles in each time bound range

            # Grab intersecting storm information to exclude confounding storms from control profiles
            sol_exclusion_window = (config.controls['sol_exclusion_window_size'], 1 + config.controls['sol_exclusion_window_size'])
            control_exclusions_dict = generate_control_exclusions_dict(storm_metadata['intersecting_storm_years'],
                                                                       storm_metadata['intersecting_storm_sols'],
                                                                       sol_exclusion_window_size=sol_exclusion_window)
           
            control_profiles = [
                find_ddr1_control_profile_matches(
                    connection,
                    storm_db_id, 
                    storm_dt=storm_metadata['dt'],
                    data_source=data_source,
                    time_window_bounds=bound,
                    ltst_start=config.tolerances['LTST_DAY_START'], 
                    ltst_end=config.tolerances['LTST_NIGHT_START'],
                    good_obs_quals=config.good_obs_quals,
                    control_exclusions_dict=control_exclusions_dict,
                    data_source_date_start=data_source_date_start,
                    data_source_date_end=data_source_date_end,
                    verbose=verbose,
                ) for bound in storm_time_bounds
            ]
            storm_profiles = [
                find_ddr1_storm_profile_matches(
                    connection,
                    storm_db_id, 
                    storm_dt=storm_metadata['dt'],
                    time_window_bounds=bound,  # Sols before/after
                    data_source=data_source,
                    ltst_start=config.tolerances['LTST_DAY_START'], 
                    ltst_end=config.tolerances['LTST_NIGHT_START'],
                    good_obs_quals=config.good_obs_quals,
                    data_source_date_start=data_source_date_start,
                    data_source_date_end=data_source_date_end,
                    verbose=verbose,
                )
                for bound in storm_time_bounds
            ]
            
            storm_dfs = []
            for profile_set in storm_profiles:
                if profile_set.empty:
                    storm_dfs.append(None)
                else:
                    profile_set = add_rel_sol_columns(profile_set, storm_metadata['dt'])
                    storm_dfs.append(load_ddr2_from_profile_matches(profile_set))

            control_dfs = []
            for profile_set in control_profiles:
                if profile_set.empty:
                    control_dfs.append(None)
                else:
                    profile_set = add_rel_sol_columns(profile_set, storm_metadata['dt'], zero_out_year=True)
                    control_dfs.append(load_ddr2_from_profile_matches(profile_set))

            ###############
            # Generate and save plots
            save_png_fpath = op.join(save_png_dir, f'{storm_metadata["storm_id"]}_MY{storm_metadata["mars_year"]}_sol{storm_metadata["sol_int"]:03}_dbid{storm_db_id:05}.png')
            make_storm_control_profile_comparison_plot(control_dfs, storm_dfs, storm_metadata['sol'], storm_time_bounds, yaxis="Alt", save_path=save_png_fpath)

        engine.dispose()
        return 0
    
    except Exception as e:
        logging.error('Problem with storm DB ID %i during time profile plotting: %s', storm_db_id, e)
        engine.dispose()
        raise e

def create_distribution_plot(db_url, storm_metadata, data_source, save_png_dir, verbose=False):
    """
    For a single storm instance, plot the distribution of storm/control temperatures and dust distributions.
    """
    storm_db_id = storm_metadata['storm_db_id']

    # Parse data source date range from config
    date_start_str, date_end_str = config.dust_database_time_spans[data_source]
    data_source_date_start = datetime.strptime(date_start_str, '%Y-%m-%d')
    data_source_date_end = datetime.strptime(date_end_str, '%Y-%m-%d')

    engine = create_engine(db_url, poolclass=NullPool)
    try:
        with engine.connect() as connection:
            ##############################
            # Find storm/control profiles

            ############
            # Storm data
            storm_profile_hits = find_ddr1_storm_profile_matches(
                connection, 
                storm_db_id, 
                storm_dt=storm_metadata['dt'],
                time_window_bounds=(0, 1),  # Limit our range to 1 sol (the sol of the storm)
                data_source=data_source,
                ltst_start=config.tolerances['LTST_DAY_START'], 
                ltst_end=config.tolerances['LTST_NIGHT_START'],
                good_obs_quals=config.good_obs_quals,
                data_source_date_start=data_source_date_start,
                data_source_date_end=data_source_date_end,
                verbose=verbose
            )
            if storm_profile_hits.empty:
                logging.warning('No storm profiles found for storm: %d. Skipping', storm_db_id)
                return 1

            # Add relative timing information
            storm_profile_hits = add_rel_sol_columns(storm_profile_hits, storm_metadata['dt'])

            storms_merged_ddr = load_ddr2_from_profile_matches(storm_profile_hits, verbose)

            # Merge DDR1 and DDR2 information. Drop most columns
            storms_merged_ddr_filtered = storms_merged_ddr[['dt', 'Pres', 'T', 'Dust', 'Alt', 'level', 'Date', 'UTC', 'mars_year', 'sol', 'L_s', 'LTST', 'Profile_lat', 'Profile_lon', 'Dust_column', 'Surf_lat', 'Surf_lon']].copy()
            dust_arr = storms_merged_ddr_filtered['Dust'].to_numpy()
            storms_merged_ddr_filtered['LogDust'] = np.where(dust_arr > 0, np.log10(dust_arr), np.nan)

            ##############
            # Control data
            control_profile_hits = find_ddr1_control_profile_matches(
                connection, 
                storm_db_id, 
                storm_dt=storm_metadata['dt'],
                data_source=data_source,
                ls_tolerance=config.controls['Ls_window_distribution_plots'],
                ltst_start=config.tolerances['LTST_DAY_START'], 
                ltst_end=config.tolerances['LTST_NIGHT_START'],
                good_obs_quals=config.good_obs_quals,
                data_source_date_start=data_source_date_start,
                data_source_date_end=data_source_date_end,
                verbose=verbose)

            if control_profile_hits.empty:
                logging.warning('No storm profiles found for storm: %d. Skipping', storm_db_id)
                return 1

            # Add relative timing information
            control_profile_hits = add_rel_sol_columns(control_profile_hits, storm_metadata['dt'], zero_out_year=True)

            controls_merged_ddr = load_ddr2_from_profile_matches(control_profile_hits, verbose)

            # Merge DDR1 and DDR2 information. Drop most columns
            controls_merged_ddr_filtered = controls_merged_ddr[['dt', 'Pres', 'T', 'Dust', 'Alt', 'level', 'Date', 'UTC', 'mars_year', 'sol', 'L_s', 'LTST', 'Profile_lat', 'Profile_lon', 'Dust_column', 'Surf_lat', 'Surf_lon']].copy()
            controls_merged_ddr_filtered['LogDust'] = controls_merged_ddr_filtered['Dust'].apply(lambda x: np.log10(x) if x > 0 else np.nan)

            ####################
            # Combine dataframes
            storms_merged_ddr_filtered['profile_type'] = 'storm'
            controls_merged_ddr_filtered['profile_type'] = 'control'
            df_concatenated = pd.concat([storms_merged_ddr_filtered, controls_merged_ddr_filtered])
            
            ##############################
            # Create save path and plot
            save_png_fpath = None
            if save_png_dir:
                save_png_fpath = op.join(save_png_dir, f'{storm_metadata["storm_id"]}_MY{storm_metadata["mars_year"]}_sol{storm_metadata["sol_int"]:03}_dbid{storm_db_id:05}.png')

            make_storm_control_distributions_plot(df_concatenated, storm_db_id, save_png_fpath)
            
        engine.dispose()
        return 0

    except Exception as e:
        logging.error('Problem with storm DB ID %i during distribution plotting: %s', storm_db_id, e)
        raise e
            

def create_collage_plot(db_url, storm_metadata, data_source, save_png_dir, save_html_dir, verbose):
    """Helper to grab atmospheric info for one storm and produce a plot collage"""

    storm_db_id = storm_metadata['storm_db_id']

    # Parse data source permitted date range from config (need this as MDSSD/MDAD didn't cover the same dates)
    date_start_str, date_end_str = config.dust_database_time_spans[data_source]
    data_source_date_start = datetime.strptime(date_start_str, '%Y-%m-%d')
    data_source_date_end = datetime.strptime(date_end_str, '%Y-%m-%d')

    #XXX: Could consider changing the below to use a Session, but would require not using Pandas' SQL functionality
    engine = create_engine(db_url, poolclass=NullPool)
    extracted_metadata = {}
    try:
        with engine.connect() as connection:
            ###############
            # Extract Storm Profiles
            if verbose:
                logging.info(f'Storm DB ID {storm_db_id:05}: Extracting storms from DB') 
            storm_profile_hits = find_ddr1_storm_profile_matches(connection, storm_db_id, 
                                                                 storm_dt=storm_metadata['dt'],
                                                                 time_window_bounds=SOL_WINDOW_BOUNDS,  # Sols before/after
                                                                 data_source=data_source,
                                                                 ltst_start=config.tolerances['LTST_DAY_START'], 
                                                                 ltst_end=config.tolerances['LTST_NIGHT_START'],
                                                                 good_obs_quals=config.good_obs_quals,
                                                                 data_source_date_start=data_source_date_start,
                                                                 data_source_date_end=data_source_date_end,
                                                                 verbose=verbose)

            if storm_profile_hits.empty:
                logging.warning('No storm profiles found for storm: %d. Skipping', storm_db_id)

                # Deleting variables to ensure they're released from memory
                storm_profile_hits = None
                engine.dispose()
                extracted_metadata['exit_status'] = 1
                return extracted_metadata
            
            # Add relative timing information
            storm_profile_hits = add_rel_sol_columns(storm_profile_hits, storm_metadata['dt'])
            # Flag if rel_sol differences are far from zero (only for storms, not controls). Ideally, values % 1 should be close to integers
            if (storm_profile_hits['rel_sol'] % 1).between(0.25, 0.75, inclusive='neither').any():
                logging.warning(f'Searched storm profiles on DB ID {storm_db_id} (area: {storm_metadata["area"]:.3e} km^2) had relative time diffs (mod 1 sol) > 0.25 sols away from midday')

            # Add timing info (do it this way to better handle wrapping around edge of Mars year)
            init_mt = mars_time.datetime_to_marstime(storm_profile_hits['dt'].min())
            mean_mt = mars_time.datetime_to_marstime(storm_profile_hits['dt'].mean())
            end_mt = mars_time.datetime_to_marstime(storm_profile_hits['dt'].max())

            # Figure out how many sols the storm lasted
            # TODO: This is somewhat inefficient, could be refactored
            storm_metadata['storm_len_in_sols'] = len(get_storm_metadata_by_storm_id(connection, storm_metadata['storm_id'], data_source, verbose=verbose))

            # Pull the atmospheric info and add a column for log10(dust)
            storm_atmospheric_df = load_ddr2_from_profile_matches(storm_profile_hits, config.min_dust_permitted, verbose)
            storm_atmospheric_df['LogDust'] = storm_atmospheric_df['Dust'].apply(lambda x: np.log10(x) if x > 0 else np.nan)
            storm_atmospheric_df['sol_int'] = storm_atmospheric_df['sol'].astype(int)
            
            # Identify the center-most profile on each sol of the storm we have data for
            centermost_profiles_dict = find_centermost_profiles_per_sol(storm_atmospheric_df,
                                                                        storm_metadata['lat'],
                                                                        storm_metadata['lon'])
            storm_metadata['centermost_profiles_per_sol'] = str(centermost_profiles_dict)

            ###############
            # Extract control profiles
            if verbose:
                logging.info(f'Storm DB ID {storm_db_id:05}: Extracting controls from DB') 

            # Grab intersecting storm information to exclude confounding storms from control profiles
            sol_exclusion_window = (config.controls['sol_exclusion_window_size'], 1 + config.controls['sol_exclusion_window_size'])
            control_exclusions_dict = generate_control_exclusions_dict(storm_metadata['intersecting_storm_years'],
                                                                       storm_metadata['intersecting_storm_sols'],
                                                                       sol_exclusion_window_size=sol_exclusion_window)
                
            control_profile_hits = find_ddr1_control_profile_matches(connection, storm_db_id, 
                                                                     storm_dt=storm_metadata['dt'],
                                                                     data_source=data_source,
                                                                     time_window_bounds=SOL_WINDOW_BOUNDS,  # Sols before/after
                                                                     ltst_start=config.tolerances['LTST_DAY_START'], 
                                                                     ltst_end=config.tolerances['LTST_NIGHT_START'],
                                                                     good_obs_quals=config.good_obs_quals,
                                                                     control_exclusions_dict=control_exclusions_dict,
                                                                     data_source_date_start=data_source_date_start,
                                                                     data_source_date_end=data_source_date_end,
                                                                     verbose=verbose)
            control_atmospheric_df = pd.DataFrame()

            if not control_profile_hits.empty:

                # Add relative timing information to DDR1 (only capturing sol difference, but discarding difference in years)
                control_profile_hits = add_rel_sol_columns(control_profile_hits, storm_metadata['dt'], zero_out_year=True)

                # Pull the atmospheric info and add a column for log10(dust)
                control_atmospheric_df = load_ddr2_from_profile_matches(control_profile_hits, config.min_dust_permitted, verbose)
                control_atmospheric_df['LogDust'] = control_atmospheric_df['Dust'].apply(lambda x: np.log10(x) if x > 0 else np.nan)
                control_atmospheric_df['sol_int'] = control_atmospheric_df['sol'].astype(int)
 
            # TODO: Need to remove these checks and make the plotting script robust to cases with empty control profiles
            else:
                logging.warning('No control profiles found for storm: %d. Skipping.', storm_db_id)

                # Deleting variables to ensure they're released from memory
                storm_profile_hits = None
                control_profile_hits = None
                storm_atmospheric_df = None
                control_atmospheric_df = None
                engine.dispose()
                extracted_metadata['exit_status'] = 1
                return extracted_metadata
 
            if control_atmospheric_df.loc[(control_atmospheric_df['rel_sol_int'] >= storm_atmospheric_df['rel_sol_int'].min()) & 
                                          (control_atmospheric_df['rel_sol_int'] <= storm_atmospheric_df['rel_sol_int'].max())].empty:
                logging.warning('No control profiles match storm sol range for storm: %d. Skipping.', storm_db_id)

                # Deleting variables to ensure they're released from memory
                storm_profile_hits = None
                control_profile_hits = None
                storm_atmospheric_df = None
                control_atmospheric_df = None
                engine.dispose()
                extracted_metadata['exit_status'] = 1
 
                return extracted_metadata
 
            if verbose:
                logging.info(f'Found {len(storm_profile_hits)} storm profiles and {len(control_profile_hits)} control profiles')

            #########################################
            # Add original metadata to extracted metadata with some modifications
            storm_profiles_on_sol = storm_atmospheric_df.loc[storm_atmospheric_df.loc[:, 'sol_int'] == storm_metadata["sol_int"], 'Profile_identifier'].nunique()
            extracted_metadata.update({'n_storm_profiles_sol_of': storm_profiles_on_sol,
                                       'n_storm_profiles_total': len(storm_atmospheric_df.drop_duplicates(subset='Profile_identifier')),
                                       'n_control_profiles': len(control_atmospheric_df.drop_duplicates(subset='Profile_identifier')),
                                       'storm_len_in_sols': storm_metadata['storm_len_in_sols'],
                                       'mars_year': storm_metadata['mars_year'],
                                       'sol': storm_metadata['sol'],
                                       'Ls': storm_metadata['ls'],
                                       'storm_db_id': storm_db_id,
                                       'mdgm': storm_metadata['mdgm'],
                                       'storm_id': storm_metadata['storm_id'],
                                       'seq_id': storm_metadata['seq_id'],
                                       'area': storm_metadata['area'],
                                       'conflev': storm_metadata['conflev'],
                                       'lon': storm_metadata['lon'],
                                       'lat': storm_metadata['lat'],
                                       'centroid_distance_per_sol': storm_metadata['centroid_distance_per_sol'],
                                       'centroid_distance_per_sol_reff': storm_metadata['centroid_distance_per_sol_reff'],
                                       'area_increase_per_sol': storm_metadata['area_increase_per_sol'],
                                       'fractional_area_increase_per_sol': storm_metadata['fractional_area_increase_per_sol'],
                                       'storm_onset': storm_metadata['storm_onset'],
                                       'merger_onset': storm_metadata['merger_onset'],
                                       'storm_end': storm_metadata['storm_end'],
                                       'sols_since_onset/merger': storm_metadata['sols_since_onset/merger'],
                                       'centermost_profiles_per_sol': storm_metadata['centermost_profiles_per_sol'],
                                       })
            ###############
            # Prep for plotting
            
            # Set up the fpaths for saving plot output
            save_png_fpath = None
            save_html_fpath = None
            if save_png_dir:
                save_png_fpath = op.join(save_png_dir, f'{storm_metadata["storm_id"]}_MY{storm_metadata["mars_year"]}_sol{storm_metadata["sol_int"]:03}_dbid{storm_db_id:05}.png')
            if save_html_dir:
                save_html_fpath = op.join(save_html_dir, f'{storm_metadata["storm_id"]}_MY{storm_metadata["mars_year"]}_sol{storm_metadata["sol_int"]:03}_dbid{storm_db_id:05}.html')
                
            # Generate a descriptive title and list for plots
            ls_str = f'<br>Profiles Ls mean: {mean_mt.solar_longitude:0.2f}; (Ls range: {init_mt.solar_longitude:0.2f} to {end_mt.solar_longitude:0.2f})'
            mars_time_str = f'<br>MY {mean_mt.year} Sol: {mean_mt.sol}'
            storm_meta_str = f'<br>Area {int(storm_metadata["area"]):.3e} km^2 Conf: {storm_metadata["conflev"]}, Lon: {storm_metadata["lon"]}, Lat: {storm_metadata["lat"]}'
            plot_title = f'DB ID: {storm_db_id}, MDGM: {storm_metadata["mdgm"]}, Storm ID:{storm_metadata["storm_id"]}, Seq: {storm_metadata["seq_id"]}<br>{extracted_metadata["n_storm_profiles_total"]} storm profiles ({extracted_metadata["n_storm_profiles_sol_of"]} sol of). {extracted_metadata["n_control_profiles"]} total controls.{ls_str}{mars_time_str}{storm_meta_str}'
            
            # Generate collage plots, capture metadata
            extracted_metadata.update(gen_plot_collage_1(storm_metadata, storm_atmospheric_df, control_atmospheric_df, 
                                                         save_png_fpath, save_html_fpath, plot_title))

            # Deleting variables to ensure they're released from memory
            storm_profile_hits = None
            control_profile_hits = None
            storm_atmospheric_df = None
            control_atmospheric_df = None
            if verbose:
                logging.info(f'Storm DB ID {storm_db_id:05}: Plot collage saved. Processing complete.\n\n.') 

        engine.dispose()
        extracted_metadata['exit_status'] = 0

    except Exception as e:
        logging.error('Problem with storm DB ID %i: %s', storm_db_id, e)
        engine.dispose()
        extracted_metadata['exit_status'] = 1
        raise e

    engine.dispose()

    return extracted_metadata


def create_control_visualization(db_url, storm_metadata, data_source, save_png_dir, verbose):
    """Helper to create control visualization plots for a single storm's controls"""

    storm_db_id = storm_metadata['storm_db_id']

    # Parse data source date range from config
    date_start_str, date_end_str = config.dust_database_time_spans[data_source]
    data_source_date_start = datetime.strptime(date_start_str, '%Y-%m-%d')
    data_source_date_end = datetime.strptime(date_end_str, '%Y-%m-%d')

    engine = create_engine(db_url, poolclass=NullPool)
    try:
        with engine.connect() as connection:

            # Extract control profiles
            if verbose:
                logging.info(f'Extracting controls from DB for storm DB ID {storm_db_id:05}') 

            # Extract control profiles using existing helper
            control_profile_hits = find_ddr1_control_profile_matches(
                connection, storm_db_id,
                storm_dt=storm_metadata['dt'],
                data_source=data_source,
                time_window_bounds=SOL_WINDOW_BOUNDS,
                ltst_start=config.tolerances['LTST_DAY_START'], 
                ltst_end=config.tolerances['LTST_NIGHT_START'],
                good_obs_quals=config.good_obs_quals,
                data_source_date_start=data_source_date_start,
                data_source_date_end=data_source_date_end,
                verbose=verbose)

            if control_profile_hits.empty:
                logging.warning('No control profiles found for storm: %d. Skipping', storm_db_id)
                return 1
            
            # Add relative timing information
            control_profile_hits = add_rel_sol_columns(control_profile_hits, storm_metadata['dt'], zero_out_year=True)

            # Load the atmospheric data
            control_atmospheric_df = load_ddr2_from_profile_matches(control_profile_hits, config.min_dust_permitted, verbose)
            control_atmospheric_df['sol_int'] = control_atmospheric_df['sol'].astype(int)
  
            # Create save path
            save_png_fpath = None
            if save_png_dir:
                save_png_fpath = op.join(save_png_dir, f'{storm_metadata["storm_id"]}_MY{storm_metadata["mars_year"]}_sol{storm_metadata["sol_int"]:03}_dbid{storm_db_id:05}.png')

            # Generate visualization
            visualize_controls(control_atmospheric_df, 
                               pressure_levels=(40, 30, 20, 10), 
                               png_save_fpath=save_png_fpath)

        engine.dispose()
        return 0

    except Exception as e:
        logging.error('Problem with storm DB ID %i during control visualization: %s', storm_db_id, e)
        raise e


@click.command()
@click.option('--save_metadata_fpath', type=click.Path(), help='Path to the CSV to save out extracted storm/control metadata.')
@click.option('--collage_save_png_dir', type=click.Path(), default=None, help='Path to the PNG save directory.')
@click.option('--collage_save_html_dir', type=click.Path(), default=None, help='Path to the HTML save directory.')
@click.option('--profile_save_png_dir', type=click.Path(), default=None, help='Path to the PNG save directory for profile plots.')
@click.option('--profile_comparison_save_png_dir', type=click.Path(), default=None, help='Path to the PNG save directory for profile comparison plots.')
@click.option('--distributions_save_png_dir', type=click.Path(), default=None, help='Path to the PNG save directory for temp/dust storm/control distribution plots.')
@click.option('--control_visualization_png_dir', type=click.Path(), default=None, help='Path to the PNG save directory for control visualization plots.')
@click.option('--db_url', default=SQL_ENGINE_STRING_MCS13, help='PostGIS database URL.')
@click.option('--min_conf', type=int, default=1, help='Minimum confidence to use when selecting storms for analysis.')
@click.option('--data_source', type=click.Choice(['mdssd', 'mdad'], case_sensitive=False), default='mdssd', help='Which dataset to work from.')
@click.option('--verbose', is_flag=True, help='Enable verbose mode to print out debug info.')
@click.option('--smoke_test', is_flag=True, help='Enable smoke test to debug by processing a much smaller portion of data.')
def plot_storm_collages(save_metadata_fpath, collage_save_png_dir, collage_save_html_dir, profile_save_png_dir, profile_comparison_save_png_dir, 
                        distributions_save_png_dir, control_visualization_png_dir, db_url, min_conf, data_source, 
                        verbose, smoke_test):
    
    engine = create_engine(db_url, poolclass=NullPool)
    
    with engine.connect() as connection:
        # Print out some diagnostic info
        _ = get_database_table_info(connection, verbose=True)

        # Get all storm "members" above a minimum confidence
        storm_pd = get_storms_with_min_conf(connection, min_conf=min_conf, data_source=data_source)
        storm_db_ids = sorted(storm_pd.to_numpy().squeeze())

        # For each storm, get all metadata to assist in downstream processing
        storm_metadatas = [get_storm_metadata_by_database_id(connection, storm_db_id, data_source=data_source).squeeze() 
                           for storm_db_id in tqdm(storm_db_ids, desc='Getting storm metadatas', miniters=10)]
        storm_metadatas = pd.DataFrame(storm_metadatas)
        storm_metadatas.rename(columns={'id': 'storm_db_id'}, inplace=True)
        storm_metadatas_list = storm_metadatas.to_dict(orient='records')

        n_members = len(storm_db_ids)
    engine.dispose()

    # Set some number of parallel jobs and number of storms to process
    n_jobs = N_JOBS

    if smoke_test:
        n_jobs = 1

        # Pick first 3 storms
        # storm_metadatas_list = storm_metadatas_list[:3]
        # n_members = len(storm_metadatas_list)

        # Select individual storms to debug individual DB ID problems
        storm_db_ids_select = [storm_db_ids.index(ind) for ind in [11537]]
        storm_metadatas_list = [storm_metadatas_list[ind] for ind in storm_db_ids_select]
        n_members = len(storm_metadatas_list)

    logging.info('Starting processing on %d %s storm members', n_members, data_source)
    logging.info('Beginning execution with %d job(s)', n_jobs)

    with parallel_config(n_jobs=n_jobs, verbose=verbose):

        # Run storm collage generation
        if collage_save_png_dir or collage_save_html_dir:
            extracted_metadatas = Parallel()(delayed(create_collage_plot)(db_url, storm_metadata, data_source, collage_save_png_dir, collage_save_html_dir, verbose) 
                                             for storm_metadata in tqdm(storm_metadatas_list, desc='Processing collages', miniters=n_jobs))
            logging.info('Storm collages plots complete')

            df_extracted_metadatas = pd.DataFrame([d for d in extracted_metadatas if isinstance(d, dict)]).set_index('storm_db_id')
            n_exit_errors = (df_extracted_metadatas['exit_status'] > 0).sum() + df_extracted_metadatas['exit_status'].isna().sum() 

            if save_metadata_fpath:
                df_extracted_metadatas.to_csv(save_metadata_fpath)
                logging.info('Storm metadata saved to %s', save_metadata_fpath)

            n_exit_errors = (df_extracted_metadatas['exit_status'] > 0).sum() + df_extracted_metadatas['exit_status'].isna().sum() 
            logging.info('%d total storm collage plotting jobs had non-zero exit status', n_exit_errors)

        # Generate plot with storm profiles over time
        if profile_save_png_dir:
            profile_plot_exit_statuses = Parallel()(
                delayed(create_time_profile_subplot)(
                        db_url, storm_metadata, data_source, profile_save_png_dir, verbose)
                    for storm_metadata in tqdm(storm_metadatas_list, desc='Processing profile plots', miniters=n_jobs))
            logging.info('Profiles through time plots complete')

            n_exit_errors = np.sum([val for val in profile_plot_exit_statuses if val is not None])
            logging.info('%d total profile plotting jobs had non-zero exit status', n_exit_errors)

        # Generate storm and control profile plots before/during/after
        if profile_comparison_save_png_dir:
            profile_comparison_plot_exit_statuses = Parallel()(
                delayed(create_profile_comparison_subplot)(
                        db_url, storm_metadata, data_source, profile_comparison_save_png_dir, verbose)
                        for storm_metadata in tqdm(storm_metadatas_list, desc='Processing prof. comparison plots', miniters=n_jobs))

            logging.info('Storm-control profile comparison plots complete')

            n_exit_errors = np.sum([val for val in profile_comparison_plot_exit_statuses if val is not None])
            logging.info('%d total profile plotting jobs had non-zero exit status', n_exit_errors)

        # Generate plot with storm/control distribution plots
        if distributions_save_png_dir:
            dist_plot_exit_statuses = Parallel()(delayed(create_distribution_plot)(db_url, storm_metadata, data_source, distributions_save_png_dir, verbose) 
                                                 for storm_metadata in tqdm(storm_metadatas_list, desc='Processing dist. plots', miniters=n_jobs))
            logging.info('Storm/control distribution plots complete')

            n_exit_errors = np.sum(dist_plot_exit_statuses)
            logging.info('%d total profile distribution plotting jobs had non-zero exit status', n_exit_errors)

        # Generate plot that shows matched controls
        if control_visualization_png_dir:
            control_viz_exit_statuses = Parallel()(
                delayed(create_control_visualization)(
                        db_url, storm_metadata, data_source, control_visualization_png_dir, verbose) 
                for storm_metadata in tqdm(storm_metadatas_list, desc='Processing control plots', miniters=n_jobs))
            logging.info('Control visualization plots complete')

            n_exit_errors = np.sum([val for val in control_viz_exit_statuses if val is not None])
            logging.info('%d total control visualization jobs had non-zero exit status', n_exit_errors)


if __name__ == '__main__':
    plot_storm_collages()
