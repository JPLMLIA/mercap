"""
CLI script to create side-by-side time series plots by height levels for two filter sheet inputs.

This script plots a single aggregated mean/median trace with a shaded
quantile band for specified height levels over a sol range. The layout shows two columns:
MDAD (left) and MDSSD (right), with one row per pressure level. Y-axes are
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

FIG_PER_ROW_HEIGHT = 1.5
FIG_PER_COL_WIDTH = 2

LOWER_QUANTILE = 0.1
HIGHER_QUANTILE = 0.9

# Plot styling constants
HIST_ALPHA = 0.25
TRACE_AGGREGATION_FUNC = 'mean'  # mean or median
LINE_WIDTH = 0.75
N_POINTS_COLOR_THRESH = 50
LOW_N_TRACE_COLOR = 'black'

# Font size defaults
TITLE_FONTSIZE = 8
SUPTITLE_FONTSIZE = 8
LABEL_FONTSIZE = 7
TICK_FONTSIZE = 6
LEGEND_FONTSIZE = 7
GRID_LINE_LW = 0.5

# Shared y-axis override: set to (ymin, ymax) to force fixed limits, or None for auto
SHARED_YLIMS = None

# matplotlib at draw time, so it is immune to layout-engine overrides. Set to None
# to let the layout engine determine the aspect ratio automatically.
SUBPLOT_BOX_ASPECT = 0.65

# Horizontal padding in sols added to both ends of the x-axis beyond the sol range.
SOL_PAD = 1

Y_LABEL_MAPPING = {'Dust': 'Dust',
                   'T': 'Temperature',
                   'perturbation_Dust': 'Pert. Dust',
                   'perturbation_T': 'Pert. Temp.'}

logger.warning('Applying hard coded pressure levels. Be sure to check these per latitude bin. This are only for -90 to 0 and 0 to 90 latitude bins.')
LEVEL_TO_PRESSURE = {12: '420 Pa (~5 km)',
                     16: '250 Pa (~9 km )',
                     18: '200 Pa (~12 km)',
                     22: '120 Pa (~16 km)',
                     24: '93 Pa (~18 km)',
                     30: '44 Pa (~25 km)'}


def _derive_x(cols, suffix_to_strip=''):
    """Extract sol numbers from column names for x-axis values.

    Parameters
    ----------
    cols : list of str
        Column names containing sol numbers (e.g., 'T_mean_L30_sol_020').
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


def _plot_single_column(ax_main, ax_hist, df, quantity, level, sol_range,
                        trace_color, label_fontsize, tick_fontsize,
                        show_main_ylabel, show_hist_ylabel, main_ylabel,
                        show_hist_yticklabels):
    """Plot one column's time series and histogram for a single height level row.

    Parameters
    ----------
    ax_main : matplotlib.axes.Axes
        Primary axis for the quantity trace.
    ax_hist : matplotlib.axes.Axes
        Twin axis for the storm-count histogram.
    df : pandas.DataFrame
        DataFrame for this dataset (already filtered).
    quantity : str
        Quantity of interest (e.g., 'T', 'Dust', 'perturbation_T').
    level : int
        Height level index (e.g., 10, 30).
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
    ##########################################################
    # Build column lists; keep only columns present in the DataFrame
    orig_cols = [f"{quantity}_mean_L{level}_sol_{sol:03d}" for sol in sol_range]
    valid_orig_cols = [col for col in orig_cols if col in df.columns]

    expected_count = len(sol_range)
    have_count = len(valid_orig_cols)

    n_good_rows = 0
    low_q = None
    high_q = None

    if valid_orig_cols:
        ##########################################################
        # Aggregate across storms to get the central-tendency trace and quantile band
        # Count storms that have at least one non-NaN value across the sol range
        n_good_rows = (~df[valid_orig_cols].isna().all(axis=1)).sum()

        if TRACE_AGGREGATION_FUNC == 'median':
            trace_values = df[valid_orig_cols].median().values
        elif TRACE_AGGREGATION_FUNC == 'mean':
            trace_values = df[valid_orig_cols].mean().values
        else:
            raise RuntimeError(
                f"Aggregation function isn't mean or median. Got {TRACE_AGGREGATION_FUNC}")

        low_q = df[valid_orig_cols].quantile(LOWER_QUANTILE).values
        high_q = df[valid_orig_cols].quantile(HIGHER_QUANTILE).values
        # Per-sol storm counts drive the background histogram bars
        valid_counts = df[valid_orig_cols].notna().sum().values

        x_vals = _derive_x(valid_orig_cols)

        if have_count != expected_count:
            logger.warning(
                f"Column count mismatch for {quantity}_mean_L{level}: "
                f"expected {expected_count}, found {have_count}.")

        ##########################################################
        # Plot trace and quantile band; switch to neutral color when N is too small
        plot_color = LOW_N_TRACE_COLOR if n_good_rows < N_POINTS_COLOR_THRESH else trace_color

        if not np.all(np.isnan(trace_values)):
            ax_main.plot(x_vals, trace_values, color=plot_color,
                         linewidth=LINE_WIDTH, marker='.', markersize=0.5)
            ax_main.fill_between(x_vals, low_q, high_q, color=plot_color, alpha=0.2)

        # Overlay per-sol storm count as a background bar histogram
        ax_hist.bar(x_vals, valid_counts, alpha=HIST_ALPHA, color='gray', width=0.8)

    ##########################################################
    # Axis labels and tick formatting
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
    mdad_df, mdssd_df, quantity, levels, sol_range, filepath,
    trace_color='red', row_height=FIG_PER_ROW_HEIGHT, col_width=FIG_PER_COL_WIDTH,
    title_fontsize=TITLE_FONTSIZE, suptitle_fontsize=SUPTITLE_FONTSIZE,
    label_fontsize=LABEL_FONTSIZE, tick_fontsize=TICK_FONTSIZE,
    legend_fontsize=LEGEND_FONTSIZE):
    """Create a 2-column figure with MDAD and MDSSD time series by height level.

    Parameters
    ----------
    mdad_df : pandas.DataFrame
        Filtered DataFrame for the MDAD dataset.
    mdssd_df : pandas.DataFrame
        Filtered DataFrame for the MDSSD dataset.
    quantity : str
        Quantity of interest (e.g., 'T', 'Dust', 'perturbation_T').
    levels : list of int
        Height levels to plot, in desired order (top to bottom).
    sol_range : list of int
        Sol values defining the x-axis range.
    filepath : str or Path
        Output path for the saved figure.
    trace_color : str, default='red'
        Color for trace lines and quantile fill.
    row_height : float, default=FIG_PER_ROW_HEIGHT
        Height per row in inches.
    col_width : float, default=FIG_PER_COL_WIDTH
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

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    main_ylabel = Y_LABEL_MAPPING[quantity]

    ##########################################################
    # Figure setup: 2 columns (MDAD, MDSSD) x N rows (one per level)
    n_rows = len(levels)
    n_cols = 2
    figsize = (col_width * n_cols, row_height * n_rows)

    # When LEFT_PAD_INCHES is set we manage layout manually via tight_layout +
    # subplots_adjust, so constrained_layout must be off (they conflict)
    use_constrained = LEFT_PAD_INCHES is None

    # sharex=True links x-axes so tick labels only appear on the bottom row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, squeeze=False,
                             constrained_layout=use_constrained)

    col_labels = ['MDAD', 'MDSSD']
    datasets = [mdad_df, mdssd_df]

    ##########################################################
    # Iterate over levels (rows)
    for i, level in enumerate(levels):
        ax_mdad = axes[i, 0]
        ax_mdssd = axes[i, 1]
        # Link primary y-axes within the row so ylim changes propagate automatically
        ax_mdssd.sharey(ax_mdad)
        # Twin axes overlay the storm-count histogram without disturbing the quantity y-scale
        ax_hist_mdad = ax_mdad.twinx()
        ax_hist_mdssd = ax_mdssd.twinx()

        level_label = LEVEL_TO_PRESSURE[level]

        # Collect per-column quantile arrays for shared y-limit computation after both are plotted
        row_n_counts = []
        row_low_qs = []
        row_high_qs = []

        ##########################################################
        # Iterate over datasets (columns: MDAD, MDSSD)
        for col_idx, (df, ax_main, ax_hist) in enumerate(
                zip(datasets, [ax_mdad, ax_mdssd], [ax_hist_mdad, ax_hist_mdssd])):

            if len(df) == 0:
                ax_main.text(0.5, 0.5, f"No data for {level_label}",
                             horizontalalignment='center', verticalalignment='center',
                             transform=ax_main.transAxes, fontsize=label_fontsize)
                row_n_counts.append(0)
                row_low_qs.append(None)
                row_high_qs.append(None)
                continue

            n_good, low_q, high_q = _plot_single_column(
                ax_main=ax_main,
                ax_hist=ax_hist,
                df=df,
                quantity=quantity,
                level=level,
                sol_range=sol_range,
                trace_color=trace_color,
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
                show_main_ylabel=(col_idx == 0),  # y-label only on left column
                show_hist_ylabel=(col_idx == 1),  # histogram label only on right column
                main_ylabel=main_ylabel,
                show_hist_yticklabels=True)

            row_n_counts.append(n_good)
            row_low_qs.append(low_q)
            row_high_qs.append(high_q)

            ax_main.grid(True, linestyle='--', alpha=0.3)
            ax_main.axvline(0, color='gray', linestyle='--', linewidth=GRID_LINE_LW)  # storm onset
            if 'perturbation' in quantity:
                ax_main.axhline(0, color='gray', linestyle='--', linewidth=GRID_LINE_LW)
            if quantity in ('Dust', 'perturbation_Dust'):
                ax_main.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # scilimits of (0, 0) means always use scientific notation
                ax_main.yaxis.get_offset_text().set_fontsize(tick_fontsize)
            ax_main.tick_params(axis='x', labelsize=tick_fontsize)

            # Top row includes the dataset name prefix; subsequent rows omit it to reduce clutter
            if i == 0:
                ax_main.set_title(
                    f"{col_labels[col_idx]}\n{level_label}; N={n_good}",
                    fontsize=title_fontsize)
            else:
                ax_main.set_title(f"{level_label}; N={n_good}",
                                  fontsize=title_fontsize)

        ##########################################################
        # Shared y-limits: derive from the union of both columns' quantile bands so the
        # two datasets remain directly comparable within each level row
        if SHARED_YLIMS is not None:
            ax_mdad.set_ylim(SHARED_YLIMS)
        else:
            all_low = [q for q in row_low_qs if q is not None]
            all_high = [q for q in row_high_qs if q is not None]
            if all_low and all_high:
                row_ymin = np.nanmin([np.nanmin(q) for q in all_low])
                row_ymax = np.nanmax([np.nanmax(q) for q in all_high])
                if not (np.isnan(row_ymin) or np.isnan(row_ymax)):
                    y_range = row_ymax - row_ymin
                    buffer = 0.05 * y_range if y_range > 0 else 0.1
                    # Setting on ax_mdad propagates to ax_mdssd automatically via sharey
                    ax_mdad.set_ylim(row_ymin - buffer, row_ymax + buffer)

        ax_hist_mdad.tick_params(axis='y', labelsize=tick_fontsize)
        ax_mdssd.tick_params(axis='y', labelleft=False)

    ##########################################################
    # Finalize and save
    # X-axis label on bottom row only (sharex handles hiding upper rows)
    for j in range(n_cols):
        axes[-1, j].set_xlabel('Relative Sol', fontsize=label_fontsize)

    # sharex propagates this to all axes
    axes[0, 0].set_xlim(sol_range[0] - SOL_PAD, sol_range[-1] + SOL_PAD)

    if LEFT_PAD_INCHES is not None:
        # tight_layout handles vertical spacing; subplots_adjust then pins the left
        # margin to a fixed physical size so axes widths are consistent across figures
        # with different label lengths (e.g. temperature vs. dust comparisons)
        plt.tight_layout()
        fig.subplots_adjust(left=LEFT_PAD_INCHES / figsize[0])

    if SUBPLOT_BOX_ASPECT is not None:
        for ax_row in axes:
            for ax in ax_row:
                ax.set_box_aspect(SUBPLOT_BOX_ASPECT)

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {filepath}")
    return fig


def create_timeseries_plots_by_height(
    mdad_filter_sheet_path, mdssd_filter_sheet_path,
    quantity, levels_str, sol_range_str, output_dir,
    ls_min, ls_max, lat_min, lat_max, conflev_min, storm_len_max,
    n_profiles_min, area_max, pval_max, pval_window_start, pval_window_end,
    storm_lifecycle_filter, trace_color='red',
    row_height=FIG_PER_ROW_HEIGHT, col_width=FIG_PER_COL_WIDTH,
    title_fontsize=TITLE_FONTSIZE, suptitle_fontsize=SUPTITLE_FONTSIZE,
    label_fontsize=LABEL_FONTSIZE, tick_fontsize=TICK_FONTSIZE,
    legend_fontsize=LEGEND_FONTSIZE):
    """Load, filter, and plot time series by height for MDAD and MDSSD datasets side by side.

    Parameters
    ----------
    mdad_filter_sheet_path : Path
        Path to the MDAD filtersheet CSV.
    mdssd_filter_sheet_path : Path
        Path to the MDSSD filtersheet CSV.
    quantity : str
        Quantity of interest (e.g., 'T', 'Dust').
    levels_str : str
        Comma-separated list of height levels (e.g., '10,20,30').
    sol_range_str : str
        Sol range as 'start,end' (e.g., '-20,21'). Rightmost sol is exclusive.
    output_dir : Path
        Output directory for the PNG files.
    ls_min, ls_max, lat_min, lat_max, conflev_min, storm_len_max, n_profiles_min,
    area_max, pval_max, pval_window_start, pval_window_end, storm_lifecycle_filter
        Filtering parameters applied identically to both datasets.
    trace_color : str, default='red'
        Color for trace lines.
    row_height : float, default=FIG_PER_ROW_HEIGHT
        Height per row in inches.
    col_width : float, default=FIG_PER_COL_WIDTH
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
    logger.info(f"Starting plotting for: {quantity} at levels {levels_str}\n")

    ##########################################################
    # Parse CLI string arguments into usable types
    try:
        levels = [int(level.strip()) for level in levels_str.split(',')]
    except ValueError:
        raise click.BadParameter(
            f"Invalid levels format: {levels_str}. Expected format: 'level1,level2,level3' (e.g., '10,20,30')")

    try:
        sol_start, sol_end = map(int, sol_range_str.split(','))
        sol_range = list(range(sol_start, sol_end))
    except ValueError:
        raise click.BadParameter(
            f"Invalid sol_range format: {sol_range_str}. Expected format: 'start,end' (e.g., '-20,20')")

    ##########################################################
    # Load and filter both datasets with identical filter parameters
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

    if len(mdad_df) == 0 and len(mdssd_df) == 0:
        logger.error("No data after filtering for either dataset, skipping...")
        return

    ##########################################################
    # Build output path and dispatch to plotting function
    logger.info("Creating plots...")

    lat_label = f'{lat_min}_to_{lat_max}'
    output_filename = f"{quantity}_levels_{levels_str.replace(',', '_')}_lat_{lat_label}.png"
    output_path = output_dir / output_filename

    plot_timeseries_two_datasets(
        mdad_df=mdad_df,
        mdssd_df=mdssd_df,
        quantity=quantity,
        levels=levels,
        sol_range=sol_range,
        filepath=output_path,
        trace_color=trace_color,
        row_height=row_height,
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
@click.option('--quantity', type=str, required=True,
              help='Quantity of interest (e.g., "T", "Dust")')
@click.option('--levels', type=str, required=True,
              help='Comma-separated list of height levels (e.g., "30,24,18")')
@click.option('--sol_range', type=str, required=True,
              help='Sol range as start,end (e.g., "-20,20")')
@click.option('--output_dir', type=click.Path(path_type=Path), required=True,
              help='Output directory for the PNG files')
@click.option('--trace_color', type=str, default='red',
              help='Color for trace lines (named color or hex value)')
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
@click.option('--row_height', type=float, default=FIG_PER_ROW_HEIGHT,
              help='Height of each figure row in inches')
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
def main(mdad_filter_sheet_path, mdssd_filter_sheet_path, quantity, levels, sol_range,
         output_dir, trace_color, ls_min, ls_max, lat_min, lat_max, conflev_min,
         storm_len_max, n_profiles_min, area_max, pval_max, pval_window_start,
         pval_window_end, storm_lifecycle_filter, row_height, col_width,
         title_fontsize, suptitle_fontsize, label_fontsize, tick_fontsize, legend_fontsize):
    """Create side-by-side MDAD/MDSSD time series plots by height levels."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pval_max_value = None if pval_max == 1.0 else pval_max

    create_timeseries_plots_by_height(
        mdad_filter_sheet_path=mdad_filter_sheet_path,
        mdssd_filter_sheet_path=mdssd_filter_sheet_path,
        quantity=quantity,
        levels_str=levels,
        sol_range_str=sol_range,
        output_dir=output_dir,
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
        trace_color=trace_color,
        row_height=row_height,
        col_width=col_width,
        title_fontsize=title_fontsize,
        suptitle_fontsize=suptitle_fontsize,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        legend_fontsize=legend_fontsize)


if __name__ == "__main__":
    main()
