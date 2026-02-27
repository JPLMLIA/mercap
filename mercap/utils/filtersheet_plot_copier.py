"""
Copy storm plots to a new directory for fast viewing using filtersheet metadata.
"""
import logging
import shutil
from pathlib import Path
from glob import glob

import click
import pandas as pd

from mercap.utils.filtering import filter_storms
from mercap.utils.util import get_logger


logger = get_logger()


def copy_storm_files(filter_sheet_path, src_dir, dest_dir, ls_min=0, ls_max=120,
                     lat_min=-90, lat_max=90, pval_max=None, conflev_min=2,
                     storm_len_max=2, n_profiles_min=0, area_max=1.6e6,
                     pval_window_start=-1, pval_window_end=2, 
                     storm_lifecycle_filter=False, dry_run=False):
    """
    Copy storm files based on filtered CSV data.
    
    Parameters
    ----------
    filter_sheet_path : Path
        Path to the filtersheet CSV
    src_dir : Path
        Source directory containing storm files (can include wildcards)
    dest_dir : Path
        Destination directory for copied files
    ls_min : float
        Minimum solar longitude
    ls_max : float
        Maximum solar longitude
    lat_min : float
        Minimum latitude
    lat_max : float
        Maximum latitude
    pval_max : float or None
        Maximum p-value for dust opacity, if None no p-value filtering is applied
    conflev_min : int
        Minimum confidence level
    storm_len_max : float
        Maximum storm length in sols
    n_profiles_min : int
        Minimum number of profiles
    area_max : float
        Maximum area
    pval_window_start : int
        Start sol for time series filtering
    pval_window_end : int
        End sol for time series filtering
    storm_lifecycle_filter : bool
        If True, filter storms to those that have an onset and end, and are not a sequence
    dry_run : bool
        If True, only print actions without copying files
    """
    # Load and filter the data
    logger.info(f"Loading data from {filter_sheet_path}")
    df = pd.read_csv(filter_sheet_path)
    
    filtered_df = filter_storms(df, ls_min=ls_min, ls_max=ls_max, 
                                lat_min=lat_min, lat_max=lat_max,
                                conflev_min=conflev_min, storm_len_max=storm_len_max, 
                                n_profiles_min=n_profiles_min,
                                area_max=area_max, 
                                pval_max=pval_max, pval_window_start=pval_window_start, pval_window_end=pval_window_end,
                                storm_lifecycle_filter=storm_lifecycle_filter)
                              
    logger.info(f"Filtered from {len(df)} to {len(filtered_df)} storms")

    # Create destination directory if it doesn't exist and not in dry run mode
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files for each storm ID
    for row in filtered_df.itertuples(index=False):
        filename = f"{row.storm_id}_MY{int(row.mars_year)}_sol{int(row.sol):03d}_dbid{int(row.storm_db_id):05d}.png"
        
        # Find all matching source files
        matching_files = []
        for path in glob(str(src_dir)):
            src_file = Path(path) / filename
            if src_file.exists():
                matching_files.append(src_file)

        # Copy the files
        if matching_files:
            for src_file in matching_files:
                dest_file = dest_dir / src_file.parent.name / filename
                
                if dry_run:
                    logger.info(f"Would copy: {src_file} -> {dest_file}")
                else:
                    if not dest_file.parent.exists():
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dest_file)
                    logger.debug(f"Copied: {src_file} -> {dest_file}")
        else:
            logger.warning(f"No files found for storm {row.storm_id}")

    msg = "Would copy" if dry_run else "Copied"
    logger.info(f"{msg} {len(filtered_df)} storms to {dest_dir}")


@click.command()
@click.option('--filter_sheet_path', type=click.Path(exists=True, path_type=Path), required=True,
              help='Path to the filtersheet CSV file')
@click.option('--src_dir', type=click.Path(path_type=Path), required=True,
              help='Source directory containing storm files (can include wildcards)')
@click.option('--dest_dir', type=click.Path(path_type=Path), required=True,
              help='Destination directory for copied files')
@click.option('--ls_min', type=click.FloatRange(min=0, max=360), default=0,
              help='Minimum solar longitude')
@click.option('--ls_max', type=click.FloatRange(min=0, max=360), default=120,
              help='Maximum solar longitude')
@click.option('--lat_min', type=click.FloatRange(min=-90, max=90), default=-90,
              help='Minimum latitude')
@click.option('--lat_max', type=click.FloatRange(min=-90, max=90), default=90,
              help='Maximum latitude')
@click.option('--conflev_min', type=click.IntRange(min=1, max=4), default=2,
              help='Minimum confidence level')
@click.option('--storm_len_max', type=click.IntRange(min=1, max=4), default=2,
              help='Maximum storm length in sols')
@click.option('--n_profiles_min', type=int, default=0,
              help='Minimum number of profiles')
@click.option('--area_max', type=float, default=1.6e6,
              help='Maximum area')
@click.option('--pval_max', type=click.FloatRange(min=0, max=1), default=1,
              help='Maximum p-value for dust opacity filtering. Use 1 to disable filtering. Note that storms w/out a pval (not enough data) will be excluded')
@click.option('--pval_window_start', type=int, default=-1,
              help='Start sol for time series filtering')
@click.option('--pval_window_end', type=int, default=2,
              help='End sol for time series filtering')
@click.option('--storm_lifecycle_filter', is_flag=True, default=False,
              help='Filter storms to those that have an onset and end, and are not a sequence')
@click.option('--dry_run', is_flag=True, default=False,
              help='Only print actions without copying files')
def main(filter_sheet_path, src_dir, dest_dir, ls_min, ls_max, lat_min, lat_max,
         conflev_min, storm_len_max, n_profiles_min, area_max, pval_max,
         pval_window_start, pval_window_end, storm_lifecycle_filter, dry_run):
    """Copy storm plots to a new directory based on filtersheet metadata."""
    # Handle the None case for max_pval (1 means disable)
    pval_max_value = None if pval_max == 1 else pval_max
    
    # Call the copy function with all parameters
    copy_storm_files(
        filter_sheet_path=filter_sheet_path,
        src_dir=src_dir,
        dest_dir=dest_dir,
        ls_min=ls_min,
        ls_max=ls_max,
        lat_min=lat_min,
        lat_max=lat_max,
        pval_max=pval_max_value,
        conflev_min=conflev_min,
        storm_len_max=storm_len_max,
        n_profiles_min=n_profiles_min,
        area_max=area_max,
        pval_window_start=pval_window_start,
        pval_window_end=pval_window_end,
        storm_lifecycle_filter=storm_lifecycle_filter,
        dry_run=dry_run
    )


if __name__ == "__main__":
    main()
