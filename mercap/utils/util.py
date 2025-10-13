from datetime import timedelta

import numpy as np
import pandas as pd
import logging
import colorlog
#from mcstools.util.mars_time import MY_Ls_to_UTC


def get_logger(log_file_path=None, log_level=logging.INFO):
    """
    Get a colorlog logger with optional file logging support.
    
    Parameters
    ----------
    log_file_path : str or Path, optional
        Path to log file. If None, no file logging is performed.
    log_level : int, optional
        Logging level (default: logging.INFO)
        
    Returns
    -------
    logger
        Configured colorlog logger
    """
    # Create logger
    logger = colorlog.getLogger('mercap_logger')
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set level
    logger.setLevel(log_level)
    
    # Prevent propagation to parent loggers (this prevents duplicate output)
    logger.propagate = False
    
    # Console handler with colors
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(asctime)s | %(lineno)d | %(module)-15s |%(reset)s %(message)s",
        datefmt='%H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG':    'black',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # File handler (if requested)
    if log_file_path:
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(lineno)d | %(levelname)-8s | %(module)-15s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        file_handler = logging.FileHandler(log_file_path, mode='w')  # 'w' creates fresh file
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
        
        logger.info(f'Logging initialized - output to terminal and {log_file_path}')
    else:
        logger.info('Logging initialized - output to terminal only')
    
    return logger


def haversine_dist(lat1, lon1, lat2, lon2, radius=None):
    """Calculate the great circle distance between lat/lon pairs. Specify params in degrees.
    
    If `radius` is None (default), returned value will be great circle distance
    in radians. If it's specified, value will be multiplied by result (e.g., to
    get great circle distance in kilometers)
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Get diffs in lat/lons
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + \
        (1 - np.sin(dlat / 2) ** 2 - np.sin((lat1 + lat2) / 2) ** 2) * \
        np.sin(dlon / 2) ** 2

    c = 2 * np.arcsin(np.sqrt(a))

    if radius is None:
        return c 
    else:
        return radius * c


###
# Older utility functions below
'''
def utc_with_offset_from_MY_Ls(row, time_offset=None):
    """Helper to get UTC time given Mars year and Ls"""

    if not time_offset:
        time_offset = timedelta(0)
        
    return MY_Ls_to_UTC(row['Mars Year'], row['Ls']) + time_offset


def get_storm_instance_prof(filepaths, lat_lims, lon_lims, time_lims, ltst_lims,
                            add_l1_props, reader=None):
    """Helper to get profile numbers + metadata from a MCS L2 file given one storm's spatiotemporal limits"""

    # Error checking input params
    if not filepaths:
        return {}
    if None in [lat_lims[0], lat_lims[1], lon_lims[0], lon_lims[1], 
                time_lims[0], time_lims[1], ltst_lims[0], ltst_lims[1]]:
        return {}
    if not reader:
        reader = MCSL22DFile()

    ddr1s = {}
    for fp in filepaths:
        ddr1_temp = reader.load(fp, ddr="DDR1")
        ddr1s[fp] = ddr1_temp

    # Create a single multi-indexed dataframe storing
    ddr1 = pd.concat(ddr1s)
    
    # Make this handle multiple FPs properly
    # Need to generate datetime object as date and time fields are stored as strings
    ddr1['datetime'] = pd.to_datetime(ddr1['Date'] + ddr1['UTC'], format='%d-%b-%Y%H:%M:%S.%f')
    
    # Find storms within spatial and time bounds
    prof_matches =  ddr1.loc[(ddr1['Profile_lat'].between(lat_lims[0], lat_lims[1], inclusive='both')) &
                             (ddr1['Profile_lon'].between(lon_lims[0], lon_lims[1], inclusive='both')) &
                             (ddr1['datetime'].between(time_lims[0], time_lims[1], inclusive='both')) &
                             (ddr1['LTST'].between(ltst_lims[0], ltst_lims[1], inclusive='both'))]
            
    # Also compute lat/lon diffs and great circle distances using Haversine formula
    query_lat_cent, query_lon_cent = np.mean(lat_lims, axis=0), np.mean(lon_lims, axis=0)
    
    lat_diff = prof_matches['Profile_lat'].to_numpy() - query_lat_cent
    lon_diff = prof_matches['Profile_lon'].to_numpy() - query_lon_cent
    prof_l2_fnames = [ind[0] for ind in prof_matches.index.to_list()]  # Extract the L2 filenames from the dataframe multindex
    great_circ_dist = haversine_dist(query_lat_cent, query_lon_cent,
                                     prof_matches['Profile_lat'].to_numpy(), 
                                     prof_matches['Profile_lon'].to_numpy())
    
    # Add some extra L1 properties if they exist
    add_l1_props_to_get = list(set(prof_matches.columns).intersection(set(add_l1_props)))
    return_dict = {key: prof_matches[key].to_list() for key in add_l1_props_to_get}
    assert len(prof_l2_fnames) == len(great_circ_dist.tolist())

    return_dict.update({'prof_num': prof_matches['Prof#'].to_list(), 
                        'L2_fname': prof_l2_fnames,
                        'great_circ_dist': great_circ_dist.tolist(),
                        'lat_diff': lat_diff.tolist(),
                        'lon_diff': lon_diff.tolist()})

    return return_dict


def get_storm_batch_profs(filepaths, lat_lims, lon_lims, time_lims, ltst_lims,
                          add_l1_props, n_jobs=16, verbose=1):
    """Helper to get profile numbers for a batch of storms"""
    
    logging.info(f'Processing {len(filepaths)} storm instances.')
    
    # Run parallel processing job for each storm
    all_profiles = Parallel(n_jobs, verbose=verbose)(
        delayed(get_storm_instance_prof)(fp, lat_lims, lon_lims, time_lims, 
                                         ltst_lims, add_l1_props)  # delayed catches outputs
        for fp, lat_lims, lon_lims, time_lims 
        in zip(filepaths, lat_lims, lon_lims, time_lims))
    
    return all_profiles
'''