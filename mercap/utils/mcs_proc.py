import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mercap.utils.mcs_l2_2d_reader import MCSL22DFile


def add_mcs_filenames(df, data_path_handler, time_tol):  #, return_none_if_missing=True):
    """Helper to add a MCS data filenames given a dataframe of storm targets"""
    
    def get_all_L2_fnames(dph, utc, time_tol):
        start_dt = utc - time_tol
        end_dt = utc + time_tol

        # Get a bank of times to search over
        l2_paths, missing = dph.find_files_from_daterange(start_dt, end_dt)
        
        # Reduce to unique values
        return l2_paths
        

    df['L2_fname'] = df.apply(lambda row: get_all_L2_fnames(
        data_path_handler, row.UTC, time_tol), axis=1)
    
    
def filter_missing_cols(orig_targets, column_name='L2_fname'):
    """Helper to filter pandas DFs containing storm info based on whether L2 file exists"""
    bad_inds = np.where(orig_targets[column_name].isna().to_numpy())[0]
    filt_targets = orig_targets.dropna(subset=column_name)
    good_inds = filt_targets.index.to_numpy()
    
    return filt_targets, good_inds, bad_inds


# TODO: convert properties from tuple here to configurable list
def pull_MCS_target_data(mcs_fpath, prof_num, 
                         mcs_ddr1_keys=('Date', 'UTC', 'Profile_lat', 'Profile_lon', 'L_s', 'Dust_column', 'H2Oice_column', 
                                        'Obs_qual', 'Rqual', 'P_qual', 'T_qual', 'Dust_qual', 'H2Oice_qual', 'surf_qual',
                                        'Surf_lat', 'Surf_lon'),
                         mcs_ddr2_keys=('Alt', 'T', 'Pres', 'Dust', 'H2Oice', 'level'), 
                         level_group_size=None, is_seq=None, storm_area=None, target_ind=None, great_circ_dist=None,
                         lat_lon_diff=None, target_type=None, mdad_ind=None,
                         level_lims=None, reader=None):
    """Extract a subset of MCS data given a known profile number"""
    if not isinstance(mcs_ddr1_keys, (tuple, list)):
        mcs_ddr1_keys = (mcs_ddr1_keys)
    if not isinstance(mcs_ddr2_keys, (tuple, list)):
        mcs_ddr2_keys = (mcs_ddr2_keys)
    
    if level_lims is None:
        level_lims = (0, np.inf)
        
    if reader is None:
        reader = MCSL22DFile()
    
    ddr1 = reader.load(mcs_fpath, ddr="DDR1")
    ddr1_vals = ddr1.loc[ddr1['Prof#'] == prof_num,  mcs_ddr1_keys].values[0]

    ddr2 = reader.load(mcs_fpath, ddr="DDR2")
    targ_data = ddr2.loc[(ddr2['Prof#'] == prof_num) & \
                         (ddr2['level'] >= level_lims[0]) & \
                         (ddr2['level'] < level_lims[1]), 
                         mcs_ddr2_keys]
    
    if great_circ_dist is not None:
        targ_data['great_circ_dist'] = great_circ_dist
    if lat_lon_diff is not None:
        targ_data['lat_diff'] = lat_lon_diff[0]
        targ_data['lon_diff'] = lat_lon_diff[1]
    if target_type is not None:
        targ_data['target_type'] = target_type
    if is_seq is not None:
        targ_data['is_seq'] = is_seq
    if storm_area is not None:
        targ_data['storm_area'] = storm_area
    if mdad_ind is not None:
        targ_data['mdad_ind'] = mdad_ind
    if target_ind is not None:
        targ_data['target_ind'] = int(target_ind)
    if level_group_size is not None:
        targ_data['level_group'] = targ_data['level'] // level_group_size
        targ_data = targ_data.groupby('level_group').mean()

    # Tack on DDR1 info. Not the most data-efficient approach since this copies the info many times
    for key, ddr1_val in zip(mcs_ddr1_keys, ddr1_vals):
        targ_data[key] = ddr1_val
    
    # Downcast many of the data types to save on memory

    # TODO: consider moving to config
    targ_data = targ_data.astype({'Profile_lat':'float32',
                                  'Profile_lon':'float32',
                                  'Surf_lat':'float32',
                                  'Surf_lon':'float32',
                                  'L_s': 'float32',
                                  'great_circ_dist': 'float32',
                                  'Dust_column':'float32',
                                  'H2Oice_column':'float32',
                                  'H2Oice_qual':'uint8',
                                  'Rqual':'uint8',
                                  'P_qual':'uint8',
                                  'T_qual':'uint8',
                                  'Dust_qual':'uint8',
                                  'Obs_qual': 'uint8',
                                  'surf_qual': 'int32',
                                  'Alt':'float32',
                                  'T':'float32',
                                  'Pres':'float32',
                                  'Dust':'float32',
                                  'H2Oice':'float32',
                                  'level':'uint8',
                                  'is_seq': 'uint8',
                                  'storm_area': 'float32',
                                  'target_ind':'int64',
                                  'target_type':'str',
                                  'lat_diff': 'float32',
                                  'lon_diff': 'float32',
                                  'mdad_ind': 'int64'}) 
    
    return targ_data


def extract_MCS_profile_data(profiles, level_group_size=None, n_jobs=-1, verbose=1):
    """Extract profile data for a large set of target locations/times"""

    # Prep for parallel processing
    mcs_fp_list = [profile['mcs_l2_fname'] for profile in profiles]
    prof_nums = [profile['prof_num'] for profile in profiles]
    is_seqs = [profile['is_seq'] for profile in profiles]
    storm_areas = [profile['storm_area'] for profile in profiles]
    great_circ_dists = [profile['great_circ_dist'] for profile in profiles]
    great_circ_dists = [profile['great_circ_dist'] for profile in profiles]
    lat_lon_diffs = [(profile['lat_diff'], profile['lon_diff']) for profile in profiles]
    mdad_inds = [profile['mdad_ind'] for profile in profiles]
    target_types = [profile['target_type'] for profile in profiles]
    ti_nums = range(len(profiles))

    # Extract MCS data from all storms in parallel
    targ_data = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(pull_MCS_target_data)(mcs_fp, prof_num, 
                                      is_seq=seq, 
                                      storm_area=sa,
                                      target_ind=ti,  
                                      great_circ_dist=cd,
                                      lat_lon_diff=ll_diff,
                                      target_type=tt,
                                      mdad_ind=mdi,
                                      level_group_size=level_group_size)
        for mcs_fp, prof_num, seq, sa, ti, cd, ll_diff, tt, mdi,
        in zip(mcs_fp_list, prof_nums, is_seqs, storm_areas, ti_nums, great_circ_dists, lat_lon_diffs, target_types, mdad_inds))
    
    # Convert to pandas DF
    targ_data = pd.concat(targ_data, ignore_index=True)

    return targ_data


def select_closest_target(profile_matches, prof_key, ddr1_props=None):
    """Select closest target profile if more than one were found"""

    no_match_dict = {prof_key: None, 
                     'mdad_ind': profile_matches['mdad_ind'],
                     'mcs_l2_fname': profile_matches['mcs_l2_fname'],
                     'is_seq': None,
                     'storm_area': None,
                     'target_type': None,
                     'great_circ_dist': None,
                     'Profile_lat': None,
                     'Profile_lon': None, 
                     'Surf_lat': None,
                     'Surf_lon': None, 
                     'lat_diff': None, 
                     'lon_diff': None}

    # Select based on minimum Haversine distance
    n_profs = len(profile_matches[prof_key])
    
    # Make sure we have at least one profile to pick from, otherwise return orig data
    if n_profs == 0:
        return no_match_dict
    elif n_profs == 1:
        good_ind = 0
    else:
        good_ind = np.argmin(profile_matches['great_circ_dist'])
        
    
    # Filter if there are multiple profiles
    targ_processed = {prof_key: profile_matches[prof_key][good_ind],
                      'mdad_ind': profile_matches['mdad_ind'],
                      'mcs_l2_fname': profile_matches['mcs_l2_fname'][good_ind],
                      'is_seq': profile_matches['is_seq'],
                      'storm_area': profile_matches['storm_area'],
                      'target_type': profile_matches['target_type'],
                      'great_circ_dist': profile_matches['great_circ_dist'][good_ind],
                      'Profile_lat': profile_matches['Profile_lat'][good_ind],
                      'Profile_lon': profile_matches['Profile_lon'][good_ind], 
                      'Surf_lat': profile_matches['Surf_lat'][good_ind],
                      'Surf_lon': profile_matches['Surf_lon'][good_ind], 
                      'lat_diff': profile_matches['lat_diff'][good_ind], 
                      'lon_diff': profile_matches['lon_diff'][good_ind]}
    
    if ddr1_props:
        for ddr1_prop in ddr1_props:
            targ_processed[ddr1_prop] = profile_matches[ddr1_prop][good_ind]
            
    return targ_processed


def load_filt_target_batch(profile_json_fpath, prof_key='prof_num', ddr1_props=None):
    """Load DDR1 profiles, remove those without matching profile, subselect profile if >1"""
    
    with open(profile_json_fpath) as fp:
        raw_profiles_list = json.load(fp)
    
    filt_profiles = []
    for profile_matches in raw_profiles_list:
        
        # Remove profiles that don't have an MCS data match
        # TODO: could move this to the initial DDR1 processing
        if not profile_matches[prof_key]:
            continue
            
        selected_targ = select_closest_target(profile_matches, prof_key, ddr1_props)
        filt_profiles.append(selected_targ)
            
    return filt_profiles
