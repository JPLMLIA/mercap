"""Command line calls for MDAD processing"""
from pathlib import Path
import json
import logging

import click

from mercap.utils.mcs_proc import extract_MCS_profile_data
from mercap import config

logging.getLogger().setLevel(logging.INFO)


@click.command()
@click.argument('fpath_profile_json', type=click.Path(exists=True, readable=True))
@click.argument('save_fpath', type=click.Path(writable=True, path_type=Path))
@click.option('--level_group_size', default=None, help='Size of block in block-wise level averaging.', type=int)
@click.option('--n_jobs', default=config.N_JOBS, help='Number of cores to use in multiprocessing.', type=int)
def extract_climate_measurements(fpath_profile_json, save_fpath, level_group_size, n_jobs):

    ###########################
    # Error checks
    if save_fpath.suffix != '.parquet':
        raise ValueError(f'Path defined in `savepath` must use parquet extension. Got {save_fpath}.')

    ###########################
    # Load targets and extract DDR2 data for profiles of interest
    with open(fpath_profile_json) as fp:
        filt_profiles = json.load(fp)

    # Remove any targets that did not have a profile match 
    filt_profiles = [p for p in filt_profiles if p['prof_num'] is not None]
    logging.info(f'\nLoaded {len(filt_profiles)} profile matches\n')

    profile_data = extract_MCS_profile_data(filt_profiles, level_group_size, n_jobs=n_jobs)
        
    ###########################
    # Save to disk
    profile_data.to_parquet(save_fpath)
