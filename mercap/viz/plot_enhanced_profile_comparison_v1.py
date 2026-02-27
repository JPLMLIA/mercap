import logging
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
            

from mercap.config import SQL_ENGINE_STRING_MCS13
from mercap import config
from mercap.utils.profile_search import (
    find_ddr1_storm_profile_matches,
    find_ddr1_control_profile_matches,
    load_ddr2_from_profile_matches,
    get_storm_metadata_by_database_id,
    generate_control_exclusions_dict,
    add_rel_sol_columns
)


logging.basicConfig(level=logging.INFO,
                    datefmt='%H:%M:%S',
                    format='%(asctime)s | %(lineno)d | %(levelname)-5s | %(module)-15s | %(message)s')


########################################################
# Configuration Constants

# Raw profile line styling
RAW_CONTROL_ALPHA = 0.1
RAW_CONTROL_LW = 0.1
RAW_CONTROL_COLOR = 'lightblue'
RAW_STORM_ALPHA = 1.0
RAW_STORM_LW = 0.33
RAW_STORM_COLOR = 'red'

PERCENTILE_BANDS = [0.1, 0.5, 0.9]
PERCENTILE_STYLES = [':', '-', ':']
CONTROL_PERCENTILE_LW = 0.5
CONTROL_PERCENTILE_COLOR = 'blue'
STORM_PERCENTILE_LW = 0.0 
STORM_PERCENTILE_COLOR = 'red'

SHADED_FILL_ALPHA = 0.15
SHADED_FILL_COLOR = 'blue'
SHADED_FILL_MEDIAN_LW = 2.0
SHADED_FILL_MEDIAN_ALPHA = 0.0

DENSITY_CMAP = 'Blues'
DENSITY_BINS = 50

FIG_WIDTH = 6
FIG_ROW_HEIGHT = 3.5
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
DPI = 300

GRID_ALPHA = 0.3
GRID_LW = 0.75
GRID_COLOR = 'gray'

YAXIS_LABELS = {
    'Alt': 'Altitude (km)',
    'Pres': 'Pressure (Pa)',
    'level': 'Level'
}

N_ANNOTATION_FONTSIZE = 10
N_ANNOTATION_POSITION_X = 0.95
N_ANNOTATION_POSITION_Y = 0.96
N_ANNOTATION_STORM_COLOR = 'firebrick'
N_ANNOTATION_CONTROL_COLOR = 'blue'
N_ANNOTATION_LINE_SPACING = 0.06

LEGEND_FONTSIZE = 7


def plot_control_profiles_shaded_fill(ax, control_df, yaxis):
    """
    Plot control profiles as shaded region between percentiles with median line.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    control_df : pd.DataFrame
        Control profile data
    yaxis : str
        Y-axis variable name
    """
    if control_df is None or control_df.empty:
        return
    
    # Calculate 10th, 50th (median), and 90th percentiles at each level
    quantiles_10 = control_df.groupby("level")[["T", "Dust", yaxis]].quantile(0.1)
    quantiles_50 = control_df.groupby("level")[["T", "Dust", yaxis]].quantile(0.5)
    quantiles_90 = control_df.groupby("level")[["T", "Dust", yaxis]].quantile(0.9)
    
    # Plot temperature profile with shaded region between 10th and 90th percentiles
    ax[0].fill_betweenx(quantiles_50[yaxis], quantiles_10["T"], quantiles_90["T"],
                        alpha=SHADED_FILL_ALPHA, color=SHADED_FILL_COLOR)
    ax[0].plot(quantiles_50["T"], quantiles_50[yaxis],
               c=SHADED_FILL_COLOR, lw=SHADED_FILL_MEDIAN_LW, ls='-', alpha=SHADED_FILL_MEDIAN_ALPHA)
    
    # Plot dust profile with shaded region between 10th and 90th percentiles
    ax[1].fill_betweenx(quantiles_50[yaxis], quantiles_10["Dust"], quantiles_90["Dust"],
                        alpha=SHADED_FILL_ALPHA, color=SHADED_FILL_COLOR)
    ax[1].plot(quantiles_50["Dust"], quantiles_50[yaxis],
               c=SHADED_FILL_COLOR, lw=SHADED_FILL_MEDIAN_LW, ls='-', alpha=SHADED_FILL_MEDIAN_ALPHA)


def plot_control_profiles_density(ax, control_df, yaxis):
    """
    Plot control profiles as 2D density visualization.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    control_df : pd.DataFrame
        Control profile data
    yaxis : str
        Y-axis variable name
    """
    if control_df is None or control_df.empty:
        return
    
    # Plot temperature density heatmap
    temp_data = control_df[["T", yaxis]].dropna()
    if not temp_data.empty:
        ax[0].hexbin(temp_data["T"], temp_data[yaxis],
                     gridsize=DENSITY_BINS, cmap=DENSITY_CMAP, mincnt=1)
    
    # Plot dust density heatmap
    dust_data = control_df[["Dust", yaxis]].dropna()
    if not dust_data.empty:
        ax[1].hexbin(dust_data["Dust"], dust_data[yaxis],
                     gridsize=DENSITY_BINS, cmap=DENSITY_CMAP, mincnt=1)


def plot_control_profiles_lines(ax, control_df, yaxis):
    """
    Plot control profiles as individual lines with percentile overlays.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    control_df : pd.DataFrame
        Control profile data
    yaxis : str
        Y-axis variable name
    """
    if control_df is None or control_df.empty:
        return
    
    # Plot individual control profiles as thin, semi-transparent lines
    for cp, cp_group in control_df.groupby("Profile_identifier"):
        cp_data = cp_group.sort_values(yaxis)
        ax[0].plot(cp_data["T"], cp_data[yaxis],
                   c=RAW_CONTROL_COLOR, lw=RAW_CONTROL_LW, alpha=RAW_CONTROL_ALPHA)
        ax[1].plot(cp_data["Dust"], cp_data[yaxis],
                   c=RAW_CONTROL_COLOR, lw=RAW_CONTROL_LW, alpha=RAW_CONTROL_ALPHA)
    
    # Overlay percentile bands (10th, 50th, 90th) as thicker lines
    for quantile, style in zip(PERCENTILE_BANDS, PERCENTILE_STYLES):
        quant_group = control_df.groupby("level")[["T", "Dust", yaxis]].quantile(quantile)
        ax[0].plot(quant_group["T"], quant_group[yaxis],
                   c=CONTROL_PERCENTILE_COLOR, lw=CONTROL_PERCENTILE_LW, ls=style)
        ax[1].plot(quant_group["Dust"], quant_group[yaxis],
                   c=CONTROL_PERCENTILE_COLOR, lw=CONTROL_PERCENTILE_LW, ls=style)


########################################################
# Main Plotting Function

def make_enhanced_profile_comparison_plot(control_dfs, storm_dfs, storm_time_bounds,
                                          storm_sols_highlight, storm_sol_color, storm_sol_lw,
                                          control_mode, sharex, sharey, yaxis, include_histograms, save_path):
    """
    Plot control and storm profiles (and percentiles) before/during/after storm.
    
    Parameters
    ----------
    control_dfs : List[pd.DataFrame]
        List of control profile dataframes for each time step
    storm_dfs : List[pd.DataFrame]
        List of storm profile dataframes for each time step
    storm_time_bounds : List[tuple]
        List of (start, end) sol bounds for each time step
    storm_sols_highlight : List[int]
        List of sols to highlight as "storm day"
    storm_sol_color : str
        Color for storm-day profiles
    storm_sol_lw : float
        Line width for storm-day profiles
    control_mode : str
        Control visualization mode: shaded_fill, density, or lines
    sharex : bool
        Whether to share x-axis within each column
    sharey : bool
        Whether to share y-axis within each row
    yaxis : str
        Y-axis variable: Alt, Pres, or level
    include_histograms : bool
        Whether to include histogram columns
    save_path : str
        Path to save the figure
    """
    n_time_steps = len(storm_dfs)
    n_cols = 4 if include_histograms else 2
    fig, ax = plt.subplots(n_time_steps, n_cols, figsize=(FIG_WIDTH, FIG_ROW_HEIGHT*n_time_steps),
                           sharex='col' if sharex else False,
                           sharey='row' if sharey else False,
                           constrained_layout=True)
    
    if n_time_steps == 1:
        ax = ax.reshape(1, -1)
    
    # Process each time step (before/during/after)
    for i in range(n_time_steps):
        if control_dfs[i] is None or control_dfs[i].empty:
            continue
        
        # Plot control profiles using selected visualization mode
        if control_mode == 'shaded_fill':
            plot_control_profiles_shaded_fill(ax[i, :], control_dfs[i], yaxis)
        elif control_mode == 'density':
            plot_control_profiles_density(ax[i, :], control_dfs[i], yaxis)
        elif control_mode == 'lines':
            plot_control_profiles_lines(ax[i, :], control_dfs[i], yaxis)
        
        # Plot storm profiles
        if storm_dfs[i] is not None and not storm_dfs[i].empty:
            # Plot individual storm profiles, highlighting specified sols
            for sp, sp_group in storm_dfs[i].groupby("Profile_identifier"):
                profile_rel_sol_int = int(sp_group["rel_sol_int"].unique().squeeze())
                
                # Use highlight color/width for storm sols, default for others
                if profile_rel_sol_int in storm_sols_highlight:
                    lw = storm_sol_lw
                    c = storm_sol_color
                else:
                    lw = RAW_STORM_LW
                    c = RAW_STORM_COLOR
                
                sp_sorted = sp_group.sort_values(yaxis)
                ax[i, 0].plot(sp_sorted["T"], sp_sorted[yaxis], c=c, lw=lw, alpha=RAW_STORM_ALPHA, zorder=10000)
                ax[i, 1].plot(sp_sorted["Dust"], sp_sorted[yaxis], c=c, lw=lw, alpha=RAW_STORM_ALPHA, zorder=10000)
            
            # Overlay storm percentile bands
            for quantile, style in zip(PERCENTILE_BANDS, PERCENTILE_STYLES):
                if STORM_PERCENTILE_LW:
                    storm_quant = storm_dfs[i].groupby("level")[["T", "Dust", yaxis]].quantile(quantile)
                    ax[i, 0].plot(storm_quant["T"], storm_quant[yaxis],
                                 c=STORM_PERCENTILE_COLOR, lw=STORM_PERCENTILE_LW, ls=style, zorder=1000)
                    ax[i, 1].plot(storm_quant["Dust"], storm_quant[yaxis],
                                 c=STORM_PERCENTILE_COLOR, lw=STORM_PERCENTILE_LW, ls=style, zorder=1000)
        
        # Add profile count annotations to dust profile subplot
        n_storm = len(storm_dfs[i].groupby("Profile_identifier")) if storm_dfs[i] is not None and not storm_dfs[i].empty else 0
        n_control = len(control_dfs[i].groupby("Profile_identifier")) if control_dfs[i] is not None and not control_dfs[i].empty else 0
        
        # Compute text width so integers are right-aligned
        max_width = max(len(str(n_storm)), len(str(n_control)))
        storm_text = f"N Storm   = {n_storm:>{max_width}}"
        control_text = f"N Control = {n_control:>{max_width}}"
        
        ax[i, 1].text(N_ANNOTATION_POSITION_X, N_ANNOTATION_POSITION_Y, storm_text,
                      transform=ax[i, 1].transAxes, fontsize=N_ANNOTATION_FONTSIZE,
                      color=N_ANNOTATION_STORM_COLOR, ha='right', va='top',
                      family='monospace')
        ax[i, 1].text(N_ANNOTATION_POSITION_X, N_ANNOTATION_POSITION_Y - N_ANNOTATION_LINE_SPACING,
                      control_text, transform=ax[i, 1].transAxes,
                      fontsize=N_ANNOTATION_FONTSIZE, color=N_ANNOTATION_CONTROL_COLOR,
                      ha='right', va='top', family='monospace')
        
        # Plot control and storm histograms (if enabled)
        if include_histograms:
            # Plot control histograms for dust opacity and surface temperature
            dust_data = control_dfs[i].groupby("Profile_identifier")["Dust_column"].first().apply(
                lambda x: np.log10(x) if x > 0 else np.nan)
            temp_data = control_dfs[i].groupby("Profile_identifier")["T_surf"].first().dropna()
            
            if not dust_data.empty and not dust_data.isna().all():
                dust_weights = np.ones(len(dust_data)) / dust_data.count()
                ax[i, 2].hist(dust_data, weights=dust_weights, color=RAW_CONTROL_COLOR,
                             density=False, bins=np.arange(-2.5, 1, 0.1))
            if not temp_data.empty and not temp_data.isna().all():
                temp_weights = np.ones(len(temp_data)) / temp_data.count()
                ax[i, 3].hist(temp_data, weights=temp_weights, color=RAW_CONTROL_COLOR, density=False)
            
            # Plot storm histograms for dust opacity and surface temperature
            if storm_dfs[i] is not None and not storm_dfs[i].empty:
                dust_data = storm_dfs[i].groupby("Profile_identifier")["Dust_column"].first().apply(
                    lambda x: np.log10(x) if x > 0 else np.nan)
                dust_nan_count = dust_data.isna().sum()
                if dust_nan_count:
                    ax[i, 2].annotate(f'{dust_nan_count} NaN(s)\n{dust_nan_count/len(dust_data):0.1%}',
                                     xy=(0.05, 0.85), xycoords='axes fraction', fontsize=TICK_FONTSIZE)
                
                if not dust_data.empty and not dust_data.isna().all():
                    dust_weights = np.ones(len(dust_data)) / dust_data.count()
                    ax[i, 2].hist(dust_data, weights=dust_weights, color=RAW_STORM_COLOR,
                                 alpha=0.5, density=False, bins=np.arange(-2.5, 1, 0.1))
                
                temp_data = storm_dfs[i].groupby("Profile_identifier")["T_surf"].first().dropna()
                temp_nan_count = temp_data.isna().sum()
                if temp_nan_count:
                    ax[i, 3].annotate(f'{temp_nan_count} NaN(s)\n{temp_nan_count/len(temp_data):0.1%}',
                                     xy=(0.05, 0.85), xycoords='axes fraction', fontsize=TICK_FONTSIZE)
                
                if not temp_data.empty and not temp_data.isna().all():
                    temp_weights = np.ones(len(temp_data)) / temp_data.count()
                    ax[i, 3].hist(temp_data, weights=temp_weights, color=RAW_STORM_COLOR, alpha=0.5, density=False)
        
        # Set axis limits and scales
        ax[i, 0].set_xlim(130, 230)
        ax[i, 1].set_xlim(1e-6, 1e-2)
        ax[i, 1].set_xscale("log")
        
        if include_histograms:
            ax[i, 2].set_xlim(-2.5, -0.5)
        
        # Configure y-axis limits and add grid to profile plots
        for a in ax[i, 0:2]:
            if yaxis == "Alt":
                a.set_ylim(0, 60)
            else:
                a.set_ylim(2.e+3, 1.e-2)
                a.set_yscale("log")
            
            a.grid(True, alpha=GRID_ALPHA, lw=GRID_LW, color=GRID_COLOR)
        
        # Set axis labels
        ylabel = f"{YAXIS_LABELS[yaxis]}\n(Rel. Sols {storm_time_bounds[i][0]} thru {storm_time_bounds[i][1]-1})"
        ax[i, 0].set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
        
        if include_histograms:
            ax[i, 2].set_ylabel("Proportion", fontsize=LABEL_FONTSIZE)
            ax[i, 3].set_ylabel("Proportion", fontsize=LABEL_FONTSIZE)
        
        # Add legend to top-left subplot only
        if i == 0:
            legend_elements = [
                Line2D([0], [0], color=RAW_STORM_COLOR, lw=RAW_STORM_LW,
                       label='Storm profiles'),
                Line2D([0], [0], color=storm_sol_color, lw=storm_sol_lw,
                       label='Storm profiles\n(Sol 0)'),
                Patch(facecolor=SHADED_FILL_COLOR, alpha=SHADED_FILL_ALPHA,
                      label='Controls\n(10-90th %)')
            ]
            ax[i, 0].legend(handles=legend_elements, loc='upper right',
                           fontsize=LEGEND_FONTSIZE, framealpha=0.9)
    
    # Set column titles and bottom row x-axis labels
    ax[0, 0].set_title("Temperature Profiles", fontsize=TITLE_FONTSIZE)
    ax[0, 1].set_title("Dust Profiles", fontsize=TITLE_FONTSIZE)
    
    if include_histograms:
        ax[0, 2].set_title("Dust Opacity", fontsize=TITLE_FONTSIZE)
        ax[0, 3].set_title("Surface Temp", fontsize=TITLE_FONTSIZE)
    
    ax[-1, 0].set_xlabel("Temperature (K)", fontsize=LABEL_FONTSIZE)
    ax[-1, 1].set_xlabel(r"Dust (km$^{-1}$)", fontsize=LABEL_FONTSIZE)
    
    if include_histograms:
        ax[-1, 2].set_xlabel(r"Dust Opacity (log scale)", fontsize=LABEL_FONTSIZE)
        ax[-1, 3].set_xlabel(r"Surface Temp (K)", fontsize=LABEL_FONTSIZE)
    
    # Set tick label sizes for all subplots
    for a in ax.flat:
        a.tick_params(labelsize=TICK_FONTSIZE)
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    save_path_obj = Path(save_path)
    if save_path_obj.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
        fig.savefig(save_path, dpi=DPI)
    else:
        fig.savefig(save_path)
    plt.close(fig)
    
    return fig, ax


@click.command()
@click.option('--storm_db_id', type=int, required=True, help='Database ID of storm to plot.')
@click.option('--data_source', type=click.Choice(['mdssd', 'mdad'], case_sensitive=False), default='mdssd', help='Which dataset to work from.')
@click.option('--sol_range_before', type=str, default='-9,-3', help='Relative sol range for "before" row (e.g., "-9,-3"). Ranges are inclusive on start, exclusive on end.')
@click.option('--sol_range_during', type=str, default='-3,3', help='Relative sol range for "during" row (e.g., "-2,2"). Ranges are inclusive on start, exclusive on end.')
@click.option('--sol_range_after', type=str, default='3,9', help='Relative sol range for "after" row (e.g., "3,9"). Ranges are inclusive on start, exclusive on end.')
@click.option('--storm_sols', type=str, default='0', help='Comma-separated sols to highlight as "storm day" (e.g., "0" or "0,1,2").')
@click.option('--storm_sol_color', type=str, default='red', help='Color for storm-day profiles.')
@click.option('--storm_sol_lw', type=float, default=1, help='Line width for storm-day profiles.')
@click.option('--control_mode', type=click.Choice(['shaded_fill', 'density', 'lines'], case_sensitive=False), default='shaded_fill', help='Control visualization mode.')
@click.option('--yaxis', type=click.Choice(['Alt', 'Pres', 'level'], case_sensitive=True), default='Alt', help='Y-axis variable.')
@click.option('--sharex/--no_sharex', default=True, help='Share x-axis within each column.')
@click.option('--sharey/--no_sharey', default=False, help='Share y-axis within each row.')
@click.option('--include_histograms', is_flag=True, help='Include histogram columns for dust opacity and surface temperature.')
@click.option('--save_path_plot', type=click.Path(), required=True, help='Output file path (PNG, PDF, or other matplotlib-supported format).')
@click.option('--save_path_csv', type=click.Path(), default=None, help='Optional output CSV path for concatenated control and storm profile data.')
@click.option('--db_url', type=str, default=SQL_ENGINE_STRING_MCS13, help='PostGIS database URL.')
@click.option('--verbose', is_flag=True, help='Enable verbose mode to print out debug info.')
def main(storm_db_id, data_source, sol_range_before, sol_range_during, sol_range_after,
         storm_sols, storm_sol_color, storm_sol_lw, control_mode, yaxis, sharex, sharey,
         include_histograms, save_path_plot, save_path_csv, db_url, verbose):
    """
    Generate enhanced profile comparison plots for a single storm.
    
    Creates before/during/after profile comparison plots with fine-grained
    control over sol ranges, plotting styles, and control visualization modes.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse sol range strings into tuples
    def parse_sol_range(range_str):
        parts = range_str.split(',')
        return (int(parts[0]), int(parts[1]))
    
    storm_time_bounds = [
        parse_sol_range(sol_range_before),
        parse_sol_range(sol_range_during),
        parse_sol_range(sol_range_after)
    ]
    
    # Parse storm sols to highlight
    storm_sols_highlight = [int(s.strip()) for s in storm_sols.split(',')]
    
    logging.info(f'Processing storm DB ID {storm_db_id} from {data_source} dataset')
    logging.info(f'Time bounds: {storm_time_bounds}')
    logging.info(f'Storm sols to highlight: {storm_sols_highlight}')
    
    # Parse data source date range from config
    date_start_str, date_end_str = config.dust_database_time_spans[data_source]
    data_source_date_start = datetime.strptime(date_start_str, '%Y-%m-%d')
    data_source_date_end = datetime.strptime(date_end_str, '%Y-%m-%d')
    
    # Connect to database and fetch data
    engine = create_engine(db_url, poolclass=NullPool)
    try:
        with engine.connect() as connection:
            logging.info(f'Storm DB ID {storm_db_id:05}: Fetching storm metadata')
            storm_metadata = get_storm_metadata_by_database_id(connection, storm_db_id).squeeze()
            
            if verbose:
                logging.info(f'Storm metadata: MY{storm_metadata["mars_year"]}, '
                           f'Sol {storm_metadata["sol"]}, Ls {storm_metadata["ls"]:.2f}')
            
            # Generate control exclusions to avoid confounders from other storms
            sol_exclusion_window = (config.controls['sol_exclusion_window_size'],
                                   1 + config.controls['sol_exclusion_window_size'])
            control_exclusions_dict = generate_control_exclusions_dict(
                storm_metadata['intersecting_storm_years'],
                storm_metadata['intersecting_storm_sols'],
                sol_exclusion_window_size=sol_exclusion_window
            )
            
            # Fetch profiles for each time window (before/during/after)
            control_profiles = []
            storm_profiles = []
            
            for bound in storm_time_bounds:
                if verbose:
                    logging.info(f'Fetching profiles for time bound {bound}')
                
                # Fetch storm profiles for this time window
                storm_profile_hits = find_ddr1_storm_profile_matches(
                    connection,
                    storm_db_id,
                    storm_dt=storm_metadata['dt'],
                    time_window_bounds=bound,
                    data_source=data_source,
                    ltst_start=config.tolerances['LTST_DAY_START'],
                    ltst_end=config.tolerances['LTST_NIGHT_START'],
                    good_obs_quals=config.good_obs_quals,
                    data_source_date_start=data_source_date_start,
                    data_source_date_end=data_source_date_end,
                    verbose=verbose
                )
                
                if storm_profile_hits.empty:
                    storm_profiles.append(storm_profile_hits)
                else:
                    # Add relative sol columns for storm profiles
                    storm_profile_hits = add_rel_sol_columns(storm_profile_hits, storm_metadata['dt'])
                    
                    # Filter profiles to ensure rel_sol_int falls within expected range [bound[0], bound[1])
                    valid_mask = (storm_profile_hits['rel_sol_int'] >= bound[0]) & (storm_profile_hits['rel_sol_int'] < bound[1])
                    dropped_profiles = storm_profile_hits[~valid_mask]
                    if not dropped_profiles.empty:
                        dropped_rel_sols = dropped_profiles['rel_sol'].to_numpy()
                        logging.warning(f"Dropping {len(dropped_rel_sols)} storm profile(s) outside range [{bound[0]}, {bound[1]}): "
                                      f"rel_sol values = {dropped_rel_sols}")

                    storm_profile_hits = storm_profile_hits[valid_mask]
                    storm_profiles.append(storm_profile_hits)

                    # Warn if profiles are far from midday
                    if not storm_profile_hits.empty and (storm_profile_hits['rel_sol'] % 1).between(0.25, 0.75, inclusive='neither').any():
                        logging.warning(f'Storm profiles for bound {bound} had rel_sol values (mod 1) > 0.25 sols away from midday')
                    
                    logging.info(f"Storm profiles relative sols for bound {bound}:\n{storm_profile_hits['rel_sol']}")
                
                # Fetch control profiles for this time window
                control_profile_hits = find_ddr1_control_profile_matches(
                    connection,
                    storm_db_id,
                    storm_dt=storm_metadata['dt'],
                    time_window_bounds=bound,
                    data_source=data_source,
                    ltst_start=config.tolerances['LTST_DAY_START'],
                    ltst_end=config.tolerances['LTST_NIGHT_START'],
                    good_obs_quals=config.good_obs_quals,
                    control_exclusions_dict=control_exclusions_dict,
                    data_source_date_start=data_source_date_start,
                    data_source_date_end=data_source_date_end,
                    verbose=verbose
                )
                
                if control_profile_hits.empty:
                    control_profiles.append(control_profile_hits)
                else:
                    # Add relative sol columns for control profiles
                    control_profile_hits = add_rel_sol_columns(control_profile_hits, storm_metadata['dt'], zero_out_year=True)
                    
                    # Filter profiles to ensure rel_sol_int falls within expected range [bound[0], bound[1])
                    valid_mask = (control_profile_hits['rel_sol_int'] >= bound[0]) & (control_profile_hits['rel_sol_int'] < bound[1])
                    dropped_profiles = control_profile_hits[~valid_mask]
                    if not dropped_profiles.empty:
                        dropped_rel_sols = dropped_profiles['rel_sol'].to_numpy()
                        logging.warning(f"Dropping {len(dropped_rel_sols)} control profile(s) outside range [{bound[0]}, {bound[1]}): "
                                        f"rel_sol values = {dropped_rel_sols}")

                    control_profile_hits = control_profile_hits[valid_mask]
                    control_profiles.append(control_profile_hits)

                    # Warn if profiles are far from midday
                    if not control_profile_hits.empty and (control_profile_hits['rel_sol'] % 1).between(0.25, 0.75, inclusive='neither').any():
                        logging.warning(f'Control profiles for bound {bound} had rel_sol values (mod 1) > 0.25 sols away from midday')

                    logging.info(f"Control profiles relative sols for bound {bound}:\n{control_profile_hits['rel_sol']}")
                    
            
            # Load DDR2 data for storm profiles
            storm_dfs = []
            for profile_set in storm_profiles:
                if profile_set.empty:
                    storm_dfs.append(None)
                else:
                    # Ensure rel_sol columns exist before loading DDR2
                    storm_dfs.append(load_ddr2_from_profile_matches(profile_set))
            
            # Load DDR2 data for control profiles
            control_dfs = []
            for profile_set in control_profiles:
                if profile_set.empty:
                    control_dfs.append(None)
                else:
                    # Zero out year difference for controls (seasonal matching only)
                    #profile_set = add_rel_sol_columns(profile_set, storm_metadata['dt'], zero_out_year=True)
                    control_dfs.append(load_ddr2_from_profile_matches(profile_set))
            
            # Generate and save the plot
            logging.info('Generating plot')
            make_enhanced_profile_comparison_plot(
                control_dfs, storm_dfs, storm_time_bounds,
                storm_sols_highlight, storm_sol_color, storm_sol_lw,
                control_mode, sharex, sharey, yaxis, include_histograms, save_path_plot
            )

            logging.info(f'Plot saved to {save_path_plot}')

            if save_path_csv is not None:
                control_frames = [df.assign(profile_type='control') for df in control_dfs if df is not None and not df.empty]
                storm_frames = [df.assign(profile_type='storm') for df in storm_dfs if df is not None and not df.empty]
                combined_df = pd.concat(control_frames + storm_frames, ignore_index=True)

                Path(save_path_csv).parent.mkdir(parents=True, exist_ok=True)
                combined_df.to_csv(save_path_csv, index=False)
                logging.info(f'CSV saved to {save_path_csv}')
        
        engine.dispose()
    
    except Exception as e:
        logging.error(f'Problem with storm DB ID {storm_db_id}: {e}')
        engine.dispose()
        raise e


if __name__ == '__main__':
    main()
