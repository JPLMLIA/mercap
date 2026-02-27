"""
CLI script to create side-by-side time series plots by latitude bands for two filter sheet inputs.

This script plots a single aggregated central-tendency (mean/median) trace with a shaded
quantile band across latitude bands over a sol range. The layout shows two columns: MDAD
(left) and MDSSD (right), with one row per latitude band. Y-axes for the quantity are
shared within each row across the two columns; histogram twin axes are independent.
"""
import logging
from pathlib import Path

import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mercap.utils.util import get_logger
from mercap.utils.filtering import filter_storms

# Configure logging
logger = get_logger(log_level=logging.INFO)

# Constants
LAT_BANDS = [(60, 90, "60° to 90°"),
             (30, 60, "30° to 60°"),
             (0, 30, "0° to 30°"),
             (-30, 0, "-30° to 0°"),
             (-60, -30, "-60° to -30°"),
             (-90, -60, "-90° to -60°")]

LAT_BANDS = [(0, 90, "0° to 90°"),
             (-90, 0, "-90° to 0°")]

FIG_HEIGHT = 4
FIG_PER_COL_WIDTH = 3.5

LOWER_QUANTILE = 0.1
HIGHER_QUANTILE = 0.9

# Plot styling constants
HIST_ALPHA = 0.25
TRACE_AGGREGATION_FUNC = 'mean'  # mean or median
N_POINTS_COLOR_THRESH = 50
LOW_N_TRACE_COLOR = 'black'

# Font size defaults
TITLE_FONTSIZE = 9
SUPTITLE_FONTSIZE = 12
LABEL_FONTSIZE = 8
TICK_FONTSIZE = 7
LEGEND_FONTSIZE = 7

# Shared y-axis override: set to (ymin, ymax) to force fixed limits, or None for auto
SHARED_YLIMS = None


def _derive_x(cols, suffix_to_strip=''):
    """Extract sol numbers from column names for x-axis values.

    Parameters
    ----------
    cols : list of str
        Column names containing sol numbers (e.g., 'dust_opacity_mean_sol_020').
    suffix_to_strip : str, optional
        Suffix to remove before parsing the integer. Default is ''.

    Returns
    -------
    list of int or float
        Parsed sol values (np.nan on parse failure).
    """
    xs = []
    for c in cols:
        try:
            sol_str = c.split('_sol_')[-1]
            if suffix_to_strip and sol_str.endswith(suffix_to_strip):
                sol_str = sol_str[:-len(suffix_to_strip)]
            xs.append(int(sol_str))
        except Exception:
            xs.append(np.nan)
    return xs


def _plot_single_column(ax_main, ax_hist, lat_df, mean_pattern, sol_range,
                        trace_color, label_fontsize, tick_fontsize,
                        show_main_ylabel, show_hist_ylabel, main_ylabel,
                        show_hist_yticklabels):
    """Plot one column's time series and histogram for a single latitude band row.

    Parameters
    ----------
    ax_main : matplotlib.axes.Axes
        Primary axis for the quantity trace.
    ax_hist : matplotlib.axes.Axes
        Twin axis for the storm-count histogram.
    lat_df : pandas.DataFrame
        DataFrame filtered to the current latitude band.
    mean_pattern : str
        Column name pattern (e.g., 'dust_opacity_mean').
    sol_range : list of int
        Sol values defining the x-axis range.
    trace_color : str
        Color for the trace line and quantile fill.
    label_fontsize : int
        Font size for axis labels.
    tick_fontsize : int
        Font size for tick labels.
    show_main_ylabel : bool
        Whether to draw the quantity y-label on ax_main.
    show_hist_ylabel : bool
        Whether to draw the 'N. Instances' label on ax_hist.
    main_ylabel : str
        Text for the quantity y-label.
    show_hist_yticklabels : bool
        Whether to show tick labels on ax_hist's y-axis.

    Returns
    -------
    tuple
        (n_good_rows, low_quantile_values, high_quantile_values) where quantile arrays
        may be None if no valid columns were found.
    """
    # Build expected column names and keep only those present in the DataFrame
    orig_cols = [f"{mean_pattern}_sol_{sol:03d}" for sol in sol_range]
    valid_orig_cols = [col for col in orig_cols if col in lat_df.columns]

    expected_count = len(sol_range)
    have_count = len(valid_orig_cols)

    n_good_rows = 0
    low_q = None
    high_q = None

    if valid_orig_cols:
        # Count storms that have at least one non-NaN value across the sol range
        n_good_rows = (~lat_df[valid_orig_cols].isna().all(axis=1)).sum()

        if TRACE_AGGREGATION_FUNC == 'median':
            trace_values = lat_df[valid_orig_cols].median().values
        elif TRACE_AGGREGATION_FUNC == 'mean':
            trace_values = lat_df[valid_orig_cols].mean().values
        else:
            raise RuntimeError(
                f"Aggregation function isn't mean or median. Got {TRACE_AGGREGATION_FUNC}")

        low_q = lat_df[valid_orig_cols].quantile(LOWER_QUANTILE).values
        high_q = lat_df[valid_orig_cols].quantile(HIGHER_QUANTILE).values
        # Per-sol storm counts drive the background histogram bars
        valid_counts = lat_df[valid_orig_cols].notna().sum().values

        x_vals = _derive_x(valid_orig_cols)

        if have_count != expected_count:
            logger.warning(
                f"Column count mismatch for {mean_pattern}: "
                f"expected {expected_count}, found {have_count}.")

        # Switch to a neutral color when the sample size is too small to trust the trace
        plot_color = LOW_N_TRACE_COLOR if n_good_rows < N_POINTS_COLOR_THRESH else trace_color

        if not np.all(np.isnan(trace_values)):
            ax_main.plot(x_vals, trace_values, color=plot_color,
                         linewidth=1, marker='.', markersize=0.5)
            ax_main.fill_between(x_vals, low_q, high_q, color=plot_color, alpha=0.2)

        ax_hist.bar(x_vals, valid_counts, alpha=HIST_ALPHA, color='gray', width=0.8)

    # Only draw quantity y-label on the leftmost column to avoid repetition
    if show_main_ylabel:
        ax_main.set_ylabel(main_ylabel, fontsize=label_fontsize)
    # Only draw histogram y-label on the rightmost column
    if show_hist_ylabel:
        ax_hist.set_ylabel('N. Instances', fontsize=label_fontsize, rotation=270, va='bottom')

    ax_main.tick_params(axis='both', labelsize=tick_fontsize)

    if not show_hist_yticklabels:
        ax_hist.tick_params(axis='y', labelright=False)
    else:
        ax_hist.tick_params(axis='y', labelsize=tick_fontsize)

    return n_good_rows, low_q, high_q


def plot_timeseries_two_datasets(
    mdad_df, mdssd_df, base_pattern, sol_range, filepath,
    trace_color='red', height=10.5, col_width=5, title=None,
    title_fontsize=TITLE_FONTSIZE, suptitle_fontsize=SUPTITLE_FONTSIZE,
    label_fontsize=LABEL_FONTSIZE, tick_fontsize=TICK_FONTSIZE,
    legend_fontsize=LEGEND_FONTSIZE):
    """Create a 2-column figure with MDAD and MDSSD time series by latitude band.

    Parameters
    ----------
    mdad_df : pandas.DataFrame
        Filtered DataFrame for the MDAD dataset.
    mdssd_df : pandas.DataFrame
        Filtered DataFrame for the MDSSD dataset.
    base_pattern : str
        Base pattern for columns (e.g., 'dust_opacity').
    sol_range : list of int
        Sol values defining the x-axis range.
    filepath : str or Path
        Output path for the saved figure.
    trace_color : str, default='red'
        Color for trace lines and quantile fill.
    height : float, default=10.5
        Figure height in inches.
    col_width : float, default=5
        Width per column in inches.
    title : str or None, default=None
        Suptitle text; derived from base_pattern if None.
    title_fontsize : int, default=TITLE_FONTSIZE
        Font size for subplot titles.
    suptitle_fontsize : int, default=SUPTITLE_FONTSIZE
        Font size for figure suptitle.
    label_fontsize : int, default=LABEL_FONTSIZE
        Font size for axis labels.
    tick_fontsize : int, default=TICK_FONTSIZE
        Font size for tick labels.
    legend_fontsize : int, default=LEGEND_FONTSIZE
        Font size for legend text.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    mean_pattern = f"{base_pattern}_mean"

    if title is None:
        title = base_pattern.replace('_', ' ').title()
        title = title.replace('Perturbation', 'Pert.').replace('T Surf', 'T. Surf.')

    n_rows = len(LAT_BANDS)
    n_cols = 2
    figsize = (col_width * n_cols, height)

    # sharex=True links x-axes so tick labels only appear on the bottom row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, squeeze=False)

    col_labels = ['MDAD', 'MDSSD']
    datasets = [mdad_df, mdssd_df]

    for i, (lat_min, lat_max, lat_label) in enumerate(LAT_BANDS):
        ax_mdad = axes[i, 0]
        ax_mdssd = axes[i, 1]
        # Link primary y-axes within the row so ylim changes propagate automatically
        ax_mdssd.sharey(ax_mdad)
        # Twin axes overlay the storm-count histogram without disturbing the quantity y-scale
        ax_hist_mdad = ax_mdad.twinx()
        ax_hist_mdssd = ax_mdssd.twinx()

        row_n_counts = []
        row_low_qs = []
        row_high_qs = []

        for col_idx, (df, ax_main, ax_hist) in enumerate(
                zip(datasets, [ax_mdad, ax_mdssd], [ax_hist_mdad, ax_hist_mdssd])):

            lat_df = df[(df['lat'] >= lat_min) & (df['lat'] < lat_max)]

            if len(lat_df) == 0:
                ax_main.text(0.5, 0.5, f"No data for {lat_label}",
                             horizontalalignment='center', verticalalignment='center',
                             transform=ax_main.transAxes, fontsize=label_fontsize)
                row_n_counts.append(0)
                row_low_qs.append(None)
                row_high_qs.append(None)
                continue

            n_good, low_q, high_q = _plot_single_column(
                ax_main=ax_main,
                ax_hist=ax_hist,
                lat_df=lat_df,
                mean_pattern=mean_pattern,
                sol_range=sol_range,
                trace_color=trace_color,
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
                show_main_ylabel=(col_idx == 0),
                show_hist_ylabel=(col_idx == 1),
                main_ylabel=title,
                show_hist_yticklabels=True)

            row_n_counts.append(n_good)
            row_low_qs.append(low_q)
            row_high_qs.append(high_q)

            ax_main.grid(True, linestyle='--', alpha=0.3)
            # Mark storm onset at sol 0
            ax_main.axvline(0, color='gray', linestyle='--', linewidth=0.5)
            if 'perturbation' in base_pattern:
                ax_main.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            ax_main.tick_params(axis='x', labelsize=tick_fontsize)

            ax_main.set_title(
                f"{col_labels[col_idx]}: {lat_label} (N={row_n_counts[col_idx]})",
                fontsize=title_fontsize)

        # Apply shared y-limits for the quantity axes within this row; setting on
        # ax_mdad propagates to ax_mdssd automatically via sharey
        if SHARED_YLIMS is not None:
            ax_mdad.set_ylim(SHARED_YLIMS)
        else:
            # Derive limits from the union of both columns' quantile bands so the
            # two datasets remain directly comparable within each latitude row
            all_low = [q for q in row_low_qs if q is not None]
            all_high = [q for q in row_high_qs if q is not None]
            if all_low and all_high:
                row_ymin = np.nanmin([np.nanmin(q) for q in all_low])
                row_ymax = np.nanmax([np.nanmax(q) for q in all_high])
                if not (np.isnan(row_ymin) or np.isnan(row_ymax)):
                    y_range = row_ymax - row_ymin
                    buffer = 0.05 * y_range if y_range > 0 else 0.1
                    ax_mdad.set_ylim(row_ymin - buffer, row_ymax + buffer)

        ax_hist_mdad.tick_params(axis='y', labelsize=tick_fontsize)
        ax_mdssd.tick_params(axis='y', labelleft=False)


    # X-axis label on bottom row only (sharex handles hiding upper rows)
    for j in range(n_cols):
        axes[-1, j].set_xlabel('Relative Sol', fontsize=label_fontsize)

    plt.tight_layout()

    if suptitle_fontsize:
        fig.suptitle(title, fontsize=suptitle_fontsize, y=0.99)
        plt.subplots_adjust(top=0.925)

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {filepath}")
    return fig


def create_timeseries_plot(
    mdad_filter_sheet_path, mdssd_filter_sheet_path,
    base_pattern, sol_range_str, trace_color, output_path,
    ls_min, ls_max, lat_min, lat_max, conflev_min, storm_len_max,
    n_profiles_min, area_max, pval_max, pval_window_start, pval_window_end,
    storm_lifecycle_filter, height=10.5, col_width=5,
    title_fontsize=TITLE_FONTSIZE, suptitle_fontsize=SUPTITLE_FONTSIZE,
    label_fontsize=LABEL_FONTSIZE, tick_fontsize=TICK_FONTSIZE,
    legend_fontsize=LEGEND_FONTSIZE):
    """Load, filter, and plot time series for MDAD and MDSSD datasets side by side.

    Parameters
    ----------
    mdad_filter_sheet_path : Path
        Path to the MDAD filtersheet CSV.
    mdssd_filter_sheet_path : Path
        Path to the MDSSD filtersheet CSV.
    base_pattern : str
        Base pattern for columns (e.g., 'dust_opacity').
    sol_range_str : str
        Sol range as 'start,end' (e.g., '-20,21'). Rightmost sol is exclusive.
    trace_color : str
        Color for trace lines.
    output_path : Path
        Output path for the PNG file.
    ls_min, ls_max, lat_min, lat_max, conflev_min, storm_len_max, n_profiles_min,
    area_max, pval_max, pval_window_start, pval_window_end, storm_lifecycle_filter
        Filtering parameters applied identically to both datasets.
    height : float, default=10.5
        Figure height in inches.
    col_width : float, default=5
        Width per column in inches.
    title_fontsize : int, default=TITLE_FONTSIZE
        Font size for subplot titles.
    suptitle_fontsize : int, default=SUPTITLE_FONTSIZE
        Font size for figure suptitle.
    label_fontsize : int, default=LABEL_FONTSIZE
        Font size for axis labels.
    tick_fontsize : int, default=TICK_FONTSIZE
        Font size for tick labels.
    legend_fontsize : int, default=LEGEND_FONTSIZE
        Font size for legend text.
    """
    logger.info(f"Starting plotting for: {base_pattern}\n")

    try:
        sol_start, sol_end = map(int, sol_range_str.split(','))
        sol_range = list(range(sol_start, sol_end))
    except ValueError:
        raise ValueError(
            f"Invalid sol_range format: {sol_range_str}. Expected 'start,end' (e.g., '-20,20')")

    # Bundle shared filter params so they apply identically to both datasets
    filter_kwargs = dict(
        ls_min=ls_min, ls_max=ls_max, lat_min=lat_min, lat_max=lat_max,
        conflev_min=conflev_min, storm_len_max=storm_len_max,
        n_profiles_min=n_profiles_min, area_max=area_max, pval_max=pval_max,
        pval_window_start=pval_window_start, pval_window_end=pval_window_end,
        storm_lifecycle_filter=storm_lifecycle_filter)

    logger.info(f"Loading MDAD filtersheet from {mdad_filter_sheet_path}")
    mdad_raw = pd.read_csv(mdad_filter_sheet_path, low_memory=False)
    mdad_df = filter_storms(mdad_raw, **filter_kwargs)
    logger.info(f"MDAD: filtered from {len(mdad_raw)} to {len(mdad_df)} storm instances")

    logger.info(f"Loading MDSSD filtersheet from {mdssd_filter_sheet_path}")
    mdssd_raw = pd.read_csv(mdssd_filter_sheet_path, low_memory=False)
    mdssd_df = filter_storms(mdssd_raw, **filter_kwargs)
    logger.info(f"MDSSD: filtered from {len(mdssd_raw)} to {len(mdssd_df)} storm instances")

    logger.info("Creating plot...")
    plot_timeseries_two_datasets(
        mdad_df=mdad_df,
        mdssd_df=mdssd_df,
        base_pattern=base_pattern,
        sol_range=sol_range,
        filepath=output_path,
        trace_color=trace_color,
        height=height,
        col_width=col_width,
        title_fontsize=title_fontsize,
        suptitle_fontsize=suptitle_fontsize,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        legend_fontsize=legend_fontsize)

    logger.info("Plotting complete.")


@click.command()
@click.option('--mdad_filter_sheet_path', type=click.Path(exists=True, path_type=Path),
              required=True, help='Path to the MDAD filtersheet CSV file')
@click.option('--mdssd_filter_sheet_path', type=click.Path(exists=True, path_type=Path),
              required=True, help='Path to the MDSSD filtersheet CSV file')
@click.option('--base_pattern', type=str, required=True,
              help='Base pattern for columns (e.g., "dust_opacity")')
@click.option('--sol_range', type=str, required=True,
              help='Sol range as start,end (e.g., "-20,21"). Rightmost sol is exclusive.')
@click.option('--trace_color', type=str, default='red',
              help='Color for trace lines (named color or hex value)')
@click.option('--output_path', type=click.Path(path_type=Path), required=True,
              help='Output path for the PNG file')
@click.option('--ls_min', type=click.FloatRange(min=0, max=360), default=0,
              help='Minimum solar longitude')
@click.option('--ls_max', type=click.FloatRange(min=0, max=360), default=120,
              help='Maximum solar longitude')
@click.option('--lat_min', type=click.FloatRange(min=-90, max=90), default=-90,
              help='Minimum latitude')
@click.option('--lat_max', type=click.FloatRange(min=-90, max=90), default=90,
              help='Maximum latitude')
@click.option('--conflev_min', type=click.IntRange(min=1, max=4), default=3,
              help='Minimum confidence level')
@click.option('--storm_len_max', type=click.IntRange(min=1, max=1000), default=2,
              help='Maximum storm length in sols')
@click.option('--n_profiles_min', type=int, default=0,
              help='Minimum number of profiles')
@click.option('--area_max', type=float, default=1.6e6,
              help='Maximum area (km^2)')
@click.option('--pval_max', type=click.FloatRange(min=0, max=1), default=1.0,
              help='Maximum p-value for dust opacity filtering. Use 1 to disable filtering.')
@click.option('--pval_window_start', type=int, default=-1,
              help='Start sol for time series filtering')
@click.option('--pval_window_end', type=int, default=2,
              help='End sol for time series filtering')
@click.option('--storm_lifecycle_filter', is_flag=True, type=bool,
              help='Whether or not to filter out storms that are mergers or sequences')
@click.option('--height', type=float, default=FIG_HEIGHT,
              help='Height of the figure in inches')
@click.option('--col_width', type=float, default=FIG_PER_COL_WIDTH,
              help='Width per column in inches')
@click.option('--title_fontsize', type=int, default=TITLE_FONTSIZE,
              help='Font size for subplot titles')
@click.option('--suptitle_fontsize', type=int, default=SUPTITLE_FONTSIZE,
              help='Font size for figure suptitle')
@click.option('--label_fontsize', type=int, default=LABEL_FONTSIZE,
              help='Font size for axis labels')
@click.option('--tick_fontsize', type=int, default=TICK_FONTSIZE,
              help='Font size for tick labels')
@click.option('--legend_fontsize', type=int, default=LEGEND_FONTSIZE,
              help='Font size for legend text')
def main(mdad_filter_sheet_path, mdssd_filter_sheet_path, base_pattern, sol_range,
         trace_color, output_path, ls_min, ls_max, lat_min, lat_max, conflev_min,
         storm_len_max, n_profiles_min, area_max, pval_max, pval_window_start,
         pval_window_end, storm_lifecycle_filter, height, col_width,
         title_fontsize, suptitle_fontsize, label_fontsize, tick_fontsize, legend_fontsize):
    """Create side-by-side MDAD/MDSSD time series plots by latitude bands."""
    # pval_max==1.0 is the sentinel meaning "no p-value filtering"
    pval_max_value = None if pval_max == 1.0 else pval_max

    create_timeseries_plot(
        mdad_filter_sheet_path=mdad_filter_sheet_path,
        mdssd_filter_sheet_path=mdssd_filter_sheet_path,
        base_pattern=base_pattern,
        sol_range_str=sol_range,
        trace_color=trace_color,
        output_path=output_path,
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
        height=height,
        col_width=col_width,
        title_fontsize=title_fontsize,
        suptitle_fontsize=suptitle_fontsize,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        legend_fontsize=legend_fontsize)


if __name__ == "__main__":
    main()
