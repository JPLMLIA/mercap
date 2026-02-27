import io
import logging
import time
from pathlib import Path

from PIL import Image
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, cramervonmises_2samp
from scipy.signal import lombscargle
from pyproj import CRS
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from sqlalchemy import text
import geopandas as gpd
from shapely import MultiPolygon
import matplotlib
import matplotlib.pyplot as plt
import cmcrameri.cm as cmcm
from typing import List

import mercap.config as config


logging.basicConfig(level=logging.INFO,
                    datefmt='%H:%M:%S',
                    format='%(asctime)s | %(lineno)d | %(levelname)-5s | %(module)-15s | %(message)s')


def plot_polygon_list(polygons, save_fpath):

    # Create a figure with subplots
    fig, axs = plt.subplots(1, len(polygons), figsize=(len(polygons) * 5, 5))

    # Convert the list of polygons to a GeoSeries
    gseries = gpd.GeoSeries(polygons)

    # If there is only one polygon, axs might not be an array, so we ensure it is
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    
    # Plot each polygon in its own subplot
    for ai, ax in enumerate(axs):
        gseries.loc[[ai]].plot(ax=ax, edgecolor='black', facecolor='red', alpha=0.5, aspect='equal')

    # Adjust the subplots and save
    plt.tight_layout()
    fig.savefig(save_fpath)
    plt.close(fig)



def safe_plotly_to_image(fig, db_id, scale=1.5, format='png', max_retries=10, base_delay=0.1):
    """
    Safely convert a plotly figure to image with retry logic for parallel processing.
    
    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The plotly figure to convert
    scale : float, optional
        Scale factor for image resolution
    format : str, optional
        Output image format
    max_retries : int, optional
        Maximum number of retry attempts
    base_delay : float, optional
        Base delay in seconds for exponential backoff
        
    Returns
    -------
    bytes
        Image bytes
        
    Raises
    ------
    ValueError
        If all retry attempts fail
    """
    for attempt in range(max_retries):
        try:
            return pio.to_image(fig, scale=scale, format=format)
        except ValueError as e:
            if "Transform failed with error code 525" in str(e) and attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = base_delay * (1.5 ** attempt)
                logging.warning(f"Kaleido error on attempt {attempt + 1} on DB ID {db_id}, retrying in {delay:.2f}s...")
                time.sleep(delay)
            else:
                raise e
    raise ValueError(f"Failed to convert plotly figure to image after {max_retries} attempts")


def aggregate_heatmap_data(profiles_df, cols, agg_func='mean', uniform_MY_required=True, group_by_col_name='sol_int'):

    # TODO: Eventually fix this to allow data to span edge of Mars years
    # Extra error check to confirm that we don't have have data spanning multiple years
    if uniform_MY_required:
        if len(profiles_df['mars_year'].unique()) != 1:
            raise RuntimeError('Data spans multiple Mars years')

    cols_to_keep = [group_by_col_name, 'level'] + cols
    aggregated_data = profiles_df[cols_to_keep].copy()
    
    # NOTE: Could keep float for intra-sol analysis
    
    if agg_func == 'mean':
        aggregated_data = aggregated_data.groupby([group_by_col_name, 'level'], as_index=False).mean(numeric_only=True)
    elif agg_func == 'max':
        aggregated_data = aggregated_data.groupby([group_by_col_name, 'level'], as_index=False).max(numeric_only=True)
    elif agg_func == 'min':
        aggregated_data = aggregated_data.groupby([group_by_col_name, 'level'], as_index=False).min(numeric_only=True)
    else:
        logging.error('Aggregation method not supported')

    return aggregated_data


def generate_heatmap_graph_objs_sol(aggregated_data, col_colorbar_ys, cols, center_colorbar=False, 
                                    colorscale='Viridis', colorbar_xpos=0.33, sol_col_name='sol'):
        
    # Loop over each desired column, produce a graph object
    graph_objects = []
    for ci, (col, col_colorbar_y) in enumerate(zip(cols, col_colorbar_ys)):
        pivot_table = aggregated_data.pivot(index="level", columns=sol_col_name, values=col)

        # Fill in missing sols with NaNs to prevent cross-sol data interpolation in the heatmap figure
        try:
            sols_range = range(pivot_table.columns.min(), pivot_table.columns.max() + 1)
        except Exception as e:
            logging.error(f'Error generating heatmap: {e}, range params: {pivot_table.columns.min()}, {pivot_table.columns.max() + 1}')
            raise e

        pivot_table_filled = pivot_table.reindex(sols_range, axis=1)

        # If desired, set the center of the colorbar to zero
        if center_colorbar:
            max_abs_value = np.max(np.abs(pivot_table_filled.to_numpy()))
            graph_objects.append(go.Heatmap(z=pivot_table_filled.values, x=pivot_table_filled.columns, y=pivot_table_filled.index, 
                                            colorscale=colorscale, colorbar_title=col, 
                                            zmid=0, zmin=-max_abs_value, zmax=max_abs_value,
                                            colorbar=dict(len=0.15, x=colorbar_xpos, y=col_colorbar_y)))
        else: 
            graph_objects.append(go.Heatmap(z=pivot_table_filled.values, x=pivot_table_filled.columns, y=pivot_table_filled.index, 
                                            colorscale=colorscale, colorbar_title=col, 
                                            colorbar=dict(len=0.15, x=colorbar_xpos, y=col_colorbar_y)))
    return graph_objects


def compute_level_statistics(df1, df2, compute_column, test, min_points):
    """
    Compute statistical tests for each level in the dataframes.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first dataframe containing the data (e.g., control data).
    df2 : pd.DataFrame
        The second dataframe containing the data (e.g., storm data).
    compute_column : str
        The name of the column to compute the statistics on.
    test : str
        The name of the statistical test to use ('ks' or 'cvm').
    min_points : int
        Minimum number of data points required to compute statistics.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns 'level' and 'pvalue'.
    """
    results = []
    
    # Make sure we only compute this when the level data exists in both
    if df1.empty or df2.empty:
        return pd.DataFrame(columns=['level', 'pvalue'])

    data1 = df1[['level', compute_column]].dropna()
    data2 = df2[['level', compute_column]].dropna()

    common_levels = sorted(set(data1['level']).intersection(data2['level']))
    
    for level in common_levels:
        temp_data1 = data1.loc[data1['level'] == level, compute_column].to_numpy()
        temp_data2 = data2.loc[data2['level'] == level, compute_column].to_numpy()

        # Must have at least min_points data points
        if len(temp_data1) < min_points or len(temp_data2) < min_points:
            continue  # skip this level
        
        if test == 'cvm':
            result = cramervonmises_2samp(temp_data1, temp_data2, nan_policy='raise')
            pvalue = result.pvalue
        elif test == 'ks':
            result = ks_2samp(temp_data1, temp_data2, nan_policy='raise')
            pvalue = result.pvalue
        else:
            raise ValueError("Test must be either 'ks' or 'cvm'")

        results.append({'level': level, 'pvalue': pvalue})
    
    return pd.DataFrame(results)


def compute_level_statistics_windowed(storm_df, control_df, compute_column, test, 
                                      min_points, rolling_window_storm, 
                                      rolling_window_control, timing_col='rel_sol_int'):
    """
    Compute statistical tests for each level in the dataframes over rolling windows.

    Parameters
    ----------
    storm_df : pd.DataFrame
        The storm dataframe containing the atmospheric data.
    control_df : pd.DataFrame
        The control dataframe containing background atmospheric data.
    compute_column : str
        The name of the column to compute the statistics on.
    test : str
        The name of the statistical test to use ('ks' or 'cvm').
    min_points : int
        Minimum number of data points required to compute statistics.
    rolling_window_storm : int
        The size of the rolling window over 'timing_col' for the storm profiles.
    rolling_window_control : int
        The size of the rolling window over 'timing_col' for the control profiles.
    timing_col : str
        The name of the column to use for indexing into the timing windows.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns 'storm_window_start', 'storm_window_end', 'control_window_start', 'control_window_end', 'center_sol', 'level', 'pvalue'.
    """
    if rolling_window_storm % 2 == 0 or rolling_window_control % 2 == 0:
        raise ValueError('`rolling_window` must be odd')

    results = []
    window_edge_buffer_storm = int((rolling_window_storm - 1) / 2)
    window_edge_buffer_control = int((rolling_window_control - 1) / 2)
    
    if storm_df.empty or control_df.empty:
        logging.warning('No data to compute statistics in `compute_level_statistics_windowed`')
        return pd.DataFrame(columns=['level', 'pvalue', 'storm_window_start', 'storm_window_end', 'control_window_start', 'control_window_end', 'center_sol'])
    
    # Generate statistics for a series of sol windows
    for window_center in range(storm_df[timing_col].min(), storm_df[timing_col].max() + 1):  # include the last sol in the iteration
        storm_window_sols = list(range(window_center - window_edge_buffer_storm, window_center + window_edge_buffer_storm + 1))
        control_window_sols = list(range(window_center - window_edge_buffer_control, window_center + window_edge_buffer_control + 1))

        # Subset storm dataframes for the current sol window. Controls aren't windowed since we want the aggregate background conditions
        storm_df_windowed = storm_df[storm_df[timing_col].isin(storm_window_sols)]
        control_df_windowed = control_df[control_df[timing_col].isin(control_window_sols)]

        # Compute statistics for this window
        stats_df = compute_level_statistics(storm_df_windowed, control_df_windowed, compute_column, test, min_points)
        
        # TODO: If nothing is returned, fill one level with NaNs so sol isn't skipped while plotting
        #if stats_df.empty:
        #    stats_df = pd.DataFrame({'level': [1], 'pvalue': [np.nan]})

        stats_df['storm_window_start'] = storm_window_sols[0]
        stats_df['storm_window_end'] = storm_window_sols[-1]
        stats_df['control_window_start'] = control_window_sols[0]
        stats_df['control_window_end'] = control_window_sols[-1]
        stats_df['center_sol'] = window_center
        
        results.append(stats_df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['level', 'pvalue', 'storm_window_start', 'storm_window_end', 'control_window_start', 'control_window_end', 'center_sol'])


def generate_geospatial_overview(storm_metadata, profiles_df=None, mcs_ddr1_latlon="Surf"):
    """Helper function to generate a figure showing storm outline, center, and profile 'hits' """
    
    # TODO: Assume origin here, but could do it per storm to improve projection
    target_lat = 0
    target_lon = 0

    #if np.abs(target_lat) > 65:
    #    # Stereographic polar projection
    #    lat_nat_ori = int(np.sign(target_lat) * 90)  # Longitude of natural origin
    #    target_proj = (f'+proj=stere +lat_0={lat_nat_ori} +lon_0=0 +x_0=0 ',
    #                   f'+y_0=0 +a=3396190 +b=3376200 +units=m +no_defs')
    #else:
    # Equilateral cylinder projection
    target_proj = (f'+proj=eqc +lat_ts={target_lat} +lat_0={target_lat} '
                   f'+lon_0={target_lon} +a={config.MARS_GEOID_A} +b={config.MARS_GEOID_B} +units=m +no_defs')

    # Define the custom Mars CRS using the PROJ string
    mars_crs = CRS.from_proj4(target_proj)

    ###################################

    # Extract the storm geometry for the geopandas DF
    storm_df = pd.DataFrame(storm_metadata, index=[0])  # Dummy index to avoid pandas error
    gdf = gpd.GeoDataFrame(storm_df, geometry=gpd.GeoSeries.from_wkb(storm_df['storm_polygon']))
    
    ###################################
    # TODO: Move out this buffered polygon code. Then change this to accept a list of storm polygons rather than a storm ID
    #gdf_buffered = gdf.copy()
    #gdf_buffered.crs = "EPSG:4327" # Dummy EPSG code b/c we need something to stand in
    #os.environ['PROJ_IGNORE_CELESTIAL_BODY'] = 'YES'
    #gdf_buffered['geometry'] = gdf_buffered['geometry'].to_crs(mars_crs).buffer(500000, resolution=16).to_crs('EPSG:4236')

    # Setup the figure layout
    fig = px.scatter_geo()
    fig.update_geos(projection_type="orthographic", visible=False, lataxis_showgrid=True, lonaxis_showgrid=True,
                    lataxis_gridcolor='gray',
                    lataxis_gridwidth=0.5,
                    lonaxis_gridcolor='gray',
                    lonaxis_gridwidth=0.5)
    fig.update_layout(height=500, margin={"r":0,"t":0,"l":0,"b":0})

    ###################################
    # Create storm polygon outline traces and add them to the list
    polygon_traces = []
    for poly in gdf['geometry']:

        def add_polygon_traces(polygon, color):
            x, y = polygon.exterior.xy
            polygon_traces.append(go.Scattergeo(lon=np.array(x), lat=np.array(y), mode='lines',
                                                line=dict(width=0.5, color=color), name='Storm Outline'))

	    # Handle MultiPolygon
        if isinstance(poly, MultiPolygon): 
            for subpoly in poly.geoms: 
                add_polygon_traces(subpoly, 'red')
        else:
            add_polygon_traces(poly, 'red')

    # Batch add the polygon traces to the figure
    fig.add_traces(polygon_traces)

    ###################################
    # Add storm center points
    points_trace = go.Scattergeo(lon=storm_df['lon'], lat=storm_df['lat'], mode='markers', 
                                 marker=dict(size=8, color='blue'), name='Storm Center')
    fig.add_trace(points_trace)

    ###################################
    # Optionally add MCS profile locations
    if profiles_df is not None:
        mcs_lon_lat = profiles_df.drop_duplicates(subset=[f'{mcs_ddr1_latlon}_lon', f'{mcs_ddr1_latlon}_lat'])
        points_trace = go.Scattergeo(lon=mcs_lon_lat[f'{mcs_ddr1_latlon}_lon'], lat=mcs_lon_lat[f'{mcs_ddr1_latlon}_lat'], 
                                     mode='markers', marker=dict(size=3, color='black'), name='MCS Profiles')
        fig.add_trace(points_trace)

    ###################################
    fig.update_geos(projection_scale=1.25, 
                    projection_rotation_lat=storm_df.loc[0, 'lat'], 
                    projection_rotation_lon=storm_df.loc[0, 'lon'])

    return fig


def make_time_profile_plot(
    ddr2_df: pd.DataFrame,
    storm_row: dict,
    sol_range: tuple,
    delta_sol_column: str="delta_t",
    save_path: str=None,
):
    """
    Create figure of temperature and dust altitude profiles that intersect
    storm boundaries within some number of sols around time of storm.

    ddr2_df: profiles in DDR2 data format
    storm_row: single storm metadata (MDSSD/MDAD data) row converted from pandas to a dict
    sol_range: +/-SOL_RANGE from storm metadata time to show in colorbar
    delta_sol_column: name of column with calculated sols from storm time
    save_path: path to save figure to
    """
    # Set colorbar range to +/- sol range
    sm = plt.cm.ScalarMappable(cmap=cmcm.berlin, norm=plt.Normalize(vmin=sol_range[0], vmax=sol_range[1]))
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # setup T and Dust plots
    # Plot each profile individually
    for p, pdata in ddr2_df.groupby("Profile_identifier"):
        pdata = pdata.sort_values("Alt")
        # Temp
        line = matplotlib.lines.Line2D(pdata["T"], pdata["Alt"], c=sm.to_rgba(pdata[delta_sol_column].unique().squeeze()), alpha=0.5, lw=3)
        ax[0].add_line(line)
        # Dust
        line = matplotlib.lines.Line2D(pdata["Dust"], pdata["Alt"], c=sm.to_rgba(pdata[delta_sol_column].unique().squeeze()), alpha=0.5, lw=3)
        ax[1].add_line(line)

    # Temp axes
    ax[0].set_title("Temperature")
    ax[0].set_xlabel("T [K]")
    ax[0].set_xlim(130, 240)

    # Dust axes
    ax[1].set_title("Dust")
    ax[1].set_xlabel(r"Dust [km$^{-1}$]")
    ax[1].set_xlim(1.e-6, 1.e-2)
    ax[1].set_xscale("log")

    # Both axes
    for a in ax:
        a.set_ylim(0, 60)
        a.set_ylabel("Altitude [km]")

    fig.colorbar(sm, ax=ax[1], label="Time since storm [sol]")

    fig.suptitle(
        f"Storm ID: {storm_row['storm_id']}, ID: {storm_row['storm_db_id']}, "
        f"MY{storm_row['mars_year']} Ls {storm_row['ls']}°, "
        f"Lat {storm_row['lat']}°, Lon {storm_row['lon']}°, "
        f"Area {storm_row['area']:.1E} km^2"
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    plt.close(fig)

    return fig, ax


def make_storm_control_distributions_plot(df_concatenated, storm_db_id, save_path, level_lims=(10, 62), temp_lims=(100, 300), dust_lims=(-7, -1)):
    """Helper to create a per-altitude distribution plot of storm/control profiles for temp/dust"""
    
    # Validate input ranges
    if not (isinstance(level_lims, tuple) and len(level_lims) == 2):
        raise ValueError("level_lims must be a tuple of length 2.")
    if not (isinstance(temp_lims, tuple) and len(temp_lims) == 2):
        raise ValueError("temp_lims must be a tuple of length 2.")
    if not (isinstance(dust_lims, tuple) and len(dust_lims) == 2):
        raise ValueError("dust_lims must be a tuple of length 2.")

    # Set altitude levels
    levels = sorted(list(range(level_lims[0], level_lims[1], 2)))
    levels.reverse()

    # Set colors for different profile sources
    colors = {'storm': 'red', 'control': 'blue'}
    height = 100 * len(levels)
    width = 750

    # Create subplots with 2 columns, one for temperature and one for dust
    fig = make_subplots(rows=len(levels), cols=2, shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.01,
                        subplot_titles=["Temperature Distributions", "Dust Distributions"])

    # Loop through each altitude level
    for i, level in enumerate(levels):
        # Loop through both storms and controls
        for source in ['storm', 'control']:
            df_filtered = df_concatenated[(df_concatenated['level'] == level) & (df_concatenated['profile_type'] == source)]
            
            # Add temperature histogram trace
            fig.add_trace(
                go.Histogram(
                    x=df_filtered['T'], 
                    name=f"{source} T, Level {level}",
                    marker_color=colors[source],
                    xbins=dict(start=temp_lims[0], end=temp_lims[1], size=5),
                    histnorm='probability',
                    opacity=0.5,
                    bingroup='temperature'),
                row=i+1, col=1)
            
            # Add dust histogram trace
            fig.add_trace(
                go.Histogram(
                    x=df_filtered['LogDust'], 
                    name=f"{source} LogDust, Level {level}",
                    marker_color=colors[source],
                    xbins=dict(start=dust_lims[0], end=dust_lims[1], size=0.5), 
                    histnorm='probability',
                    opacity=0.5,
                    bingroup='dust'),
                row=i+1, col=2)
            
        # Update y-axis label
        fig.update_yaxes(title_text=f"Lvl {level}<br>Prob.", row=i+1, col=1)

    fig.update_xaxes(title_text="Temp. (K)", row=len(levels), col=1)
    fig.update_xaxes(title_text="Dust per km (Log scale)", row=len(levels), col=2)

    fig.update_layout(barmode='overlay',
                      height=height,
                      width=width,
                      showlegend=False,
                      font={'size': 16}, 
                      title_text=f"Storm DB ID: {storm_db_id:05}. Temp. and Dust Distributions")
                
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(save_path)

    return fig


def smooth_control_profiles_one_step_median(control_profiles_df, storm_profiles_df,
                                            smoothing_window_size):
    """Smooth control profiles using the original aggregate-then-smooth approach.
    
    This function applies the original smoothing method:
    1. Aggregate all profiles by taking mean across all Mars years (per sol, per level)
    2. Apply rolling median filter in time
    
    Note: This method can be dominated by Mars years with many profiles.
    
    Parameters
    ----------
    control_profiles_df : pd.DataFrame
        DataFrame containing control profile data with columns:
        'mars_year', 'rel_sol_int', 'level', 'T', 'Dust', 'Dust_column', 'T_surf'
    storm_profiles_df : pd.DataFrame
        DataFrame containing storm profile data with 'rel_sol_int' column.
        Used to determine the sol range for filtering the final result.
    smoothing_window_size : int
        Window size for rolling median filter.
        
    Returns
    -------
    pd.DataFrame
        Smoothed control profiles with columns: 'rel_sol_int', 'level', 
        'T', 'Dust', 'Dust_column', 'T_surf'
    """
    
    control_profiles_df_smoothed = aggregate_heatmap_data(
        control_profiles_df, 
        cols=['T', 'Dust', 'Dust_column', 'T_surf'], 
        uniform_MY_required=False, 
        group_by_col_name='rel_sol_int').set_index(['rel_sol_int', 'level'])

    # Unstack the level index to create columns for each atmospheric level (facilitates smoothing step)
    control_profiles_df_smoothed = control_profiles_df_smoothed.unstack('level')
    
    # Fill in missing sols with NaNs to ensure proper smoothing
    # Get the widest sol bounds from both storm and control profiles
    sol_range_min = min(storm_profiles_df['rel_sol_int'].min(), control_profiles_df['rel_sol_int'].min())
    sol_range_max = max(storm_profiles_df['rel_sol_int'].max(), control_profiles_df['rel_sol_int'].max())
    # Reindex to fill missing sols with NaNs
    control_profiles_df_smoothed = control_profiles_df_smoothed.reindex(range(sol_range_min, sol_range_max + 1))

    # Apply rolling median filter in time using a sol window of specified length
    control_profiles_df_smoothed = control_profiles_df_smoothed.rolling(
        window=smoothing_window_size,
        center=True,
        min_periods=1).median()

    # Restack to get back to original MultiIndex format and then reset to a single-index format
    control_profiles_df_smoothed = control_profiles_df_smoothed.stack('level', dropna=False)
    control_profiles_df_smoothed = control_profiles_df_smoothed.reset_index()

    # Post smoothing, now limit the control information to span the same sol range as the storm data
    control_profiles_df_smoothed = control_profiles_df_smoothed.loc[(control_profiles_df_smoothed['rel_sol_int'] >= storm_profiles_df['rel_sol_int'].min()) &
                                                                    (control_profiles_df_smoothed['rel_sol_int'] <= storm_profiles_df['rel_sol_int'].max())]
    
    return control_profiles_df_smoothed


def smooth_control_profiles_two_step_median(control_profiles_df, storm_profiles_df, 
                                            smoothing_window_size):
    """Smooth control profiles using a two-step median approach.
    
    This function applies a two-step median smoothing process:
    1. Median within each Mars year (per sol, per level)
    2. Median across Mars years (per sol, per level)
    
    This ensures each Mars year contributes equally regardless of profile count,
    preventing years with many profiles from dominating the result.
    
    Parameters
    ----------
    control_profiles_df : pd.DataFrame
        DataFrame containing control profile data with columns:
        'mars_year', 'rel_sol_int', 'level', 'T', 'Dust', 'Dust_column', 'T_surf'
    storm_profiles_df : pd.DataFrame
        DataFrame containing storm profile data with 'rel_sol_int' column.
        Used to determine the sol range for filtering the final result.
    smoothing_window_size : int
        Window size for rolling median filter applied within each Mars year.
        
    Returns
    -------
    pd.DataFrame
        Smoothed control profiles with columns: 'rel_sol_int', 'level', 
        'T', 'Dust', 'Dust_column', 'T_surf'
    """
    
    cols_to_smooth = ['T', 'Dust', 'Dust_column', 'T_surf']
    cols_to_keep = ['mars_year', 'rel_sol_int', 'level'] + cols_to_smooth
    control_data = control_profiles_df[cols_to_keep].copy()
    
    # Step 1: Median within each Mars year (per sol, per level)
    control_by_my = control_data.groupby(['mars_year', 'rel_sol_int', 'level'], as_index=False).median(numeric_only=True)

    # Get sol range for filling missing values
    sol_range_min = min(storm_profiles_df['rel_sol_int'].min(), control_profiles_df['rel_sol_int'].min())
    sol_range_max = max(storm_profiles_df['rel_sol_int'].max(), control_profiles_df['rel_sol_int'].max())

    # For each Mars year, fill missing sols and apply rolling median smoothing
    smoothed_by_mars_year_list = []
    for my in control_by_my['mars_year'].unique():
        my_data = control_by_my[control_by_my['mars_year'] == my].copy()
        my_data = my_data.set_index(['rel_sol_int', 'level'])
        my_data = my_data.unstack('level')
        my_data = my_data.reindex(range(sol_range_min, sol_range_max + 1))
        
        # Apply rolling median filter within this Mars year
        my_data = my_data.rolling(
            window=smoothing_window_size,
            center=True,
            min_periods=1).median()
        
        my_data = my_data.stack('level', dropna=False)
        my_data = my_data.reset_index()
        my_data['mars_year'] = my
        smoothed_by_mars_year_list.append(my_data)

    # Combine all Mars years
    control_by_my_smoothed = pd.concat(smoothed_by_mars_year_list, ignore_index=True)

    # Step 2: Median across Mars years (per sol, per level)
    control_profiles_df_smoothed = control_by_my_smoothed.groupby(['rel_sol_int', 'level'], as_index=False).median(numeric_only=True)

    # Post smoothing, now limit the control information to span the same sol range as the storm data
    control_profiles_df_smoothed = control_profiles_df_smoothed.loc[(control_profiles_df_smoothed['rel_sol_int'] >= storm_profiles_df['rel_sol_int'].min()) &
                                                                    (control_profiles_df_smoothed['rel_sol_int'] <= storm_profiles_df['rel_sol_int'].max())]
    
    return control_profiles_df_smoothed


def smooth_control_profiles(control_profiles_df, storm_profiles_df, smoothing_window_size=None):
    """Smooth control profiles using the method specified in config.
    
    This is a wrapper function that selects the appropriate smoothing method
    based on config.controls['control_smoothing_method'].
    
    Parameters
    ----------
    control_profiles_df : pd.DataFrame
        DataFrame containing control profile data with columns:
        'mars_year', 'rel_sol_int', 'level', 'T', 'Dust', 'Dust_column', 'T_surf'
    storm_profiles_df : pd.DataFrame
        DataFrame containing storm profile data with 'rel_sol_int' column.
        Used to determine the sol range for filtering the final result.
    smoothing_window_size : int, optional
        Window size for rolling median filter.
        If None, uses config.controls['collage_controls_smoothing_window_size']
        
    Returns
    -------
    pd.DataFrame
        Smoothed control profiles with columns: 'rel_sol_int', 'level', 
        'T', 'Dust', 'Dust_column', 'T_surf'
    """
    method = config.controls.get('control_smoothing_method')
    
    if method == 'two_step_median':
        return smooth_control_profiles_two_step_median(
            control_profiles_df, storm_profiles_df, smoothing_window_size)
    elif method == 'one_step_aggregate_then_smooth':
        return smooth_control_profiles_one_step_median(
            control_profiles_df, storm_profiles_df, smoothing_window_size)
    else:
        raise ValueError(f"Unknown control_smoothing_method: {method}. "
                         f"Must be 'two_step_median' or 'aggregate_then_smooth'")


def gen_plot_collage_1(storm_metadata, storm_profiles_df, control_profiles_df,
                       png_save_fpath=None, html_save_fpath=None, plot_title=None):        
    """Helper to show various plots with dust and temp"""
    
    if not plot_title:
        plot_title = f"Storm ID: {storm_metadata['storm_db_id']}, {len(storm_profiles_df)} profiles"

    extracted_metadata = {}

    # Create subplot layout with a secondary y-axis for the frequency analysis plot
    specs = [[{} for _ in range(5)] for _ in range(5)]
    # Enable secondary_y for the subplot that will contain frequency analysis amplitude plot
    specs[2][3] = {"secondary_y": True}  # This is for rc_amp (3, 4)
    
    fig = make_subplots(rows=5, cols=5, vertical_spacing=0.09, horizontal_spacing=0.09, 
                        specs=specs,
                        subplot_titles=("Temperature", "Temp. Control", "Temp. Diff. (Storm-Control)", "Distribution Distance p-Val, Temp.", "Dust Opacity<br>Storms (dashed) and Perturbations (solid)",
                                        "Dust", "Dust Control", "Dust Diff. (Storm-Control)", "Distribution Distance p-Val, Dust", "Dust Opacity p-value",
                                        "Log10(Dust)", "Surf. Temp.<br>Red=Failure Gray=Valid", "Temp. Retrieval<br>Level of Failure", "Dust Opacity (Orange) & Surf. Temp. (Purple)<br>Amplitudes", "Surface Temp.<br>Perturbations (solid)",
                                        "Sol-to-Sol Temp. Diff.", "Dust Column<br>Red=Failure Gray=Valid",  "Dust Retrieval<br>Level of Failure", "Dust Opacity (Orange) & Surf. Temp. (Purple)<br>Phases", "Surface Temp. p-value",
                                        "Temp Gradient","Lower Atmosphere Quartiles (Levels 0-40)", "Obs. Qual.<br>Red=Storm, Blue=Control", "Spatial Overview", ""))
    
    ###################################
    # Generate smoothed (in time) control profiles
    control_profiles_df_smoothed = smooth_control_profiles(control_profiles_df, storm_profiles_df,
                                                           smoothing_window_size=config.controls['collage_controls_smoothing_window_size'])
    
    ###################################
    # Plot heatmaps with mean temp, dust, log(dust)
    aggregated_data = aggregate_heatmap_data(storm_profiles_df, cols=['T', 'Dust'], uniform_MY_required=False, group_by_col_name='rel_sol_int')
    dust_arr = aggregated_data['Dust'].to_numpy()
    aggregated_data['LogDust'] = np.where(dust_arr > 0, np.log10(dust_arr), np.nan)
    
    # TODO: Use these saved colorbar ranges and apply to controls to unify colorbars
    # Extract vmin/vmax values for consistent colorbars
    colorbar_ranges = {}
    for col in ['T', 'Dust', 'LogDust']:
        if col in aggregated_data.columns:
            data_values = aggregated_data[col].dropna()
            if not data_values.empty:
                colorbar_ranges[col] = {'vmin': data_values.min(), 'vmax': data_values.max()}
                
    heatmap_subplots = generate_heatmap_graph_objs_sol(aggregated_data, col_colorbar_ys=[0.94, 0.725, 0.5],
                                                       cols=['T', 'Dust', 'LogDust'], colorbar_xpos=0.125,
                                                       sol_col_name='rel_sol_int') 
    fig.add_trace(heatmap_subplots[0], row=1, col=1)
    fig.add_trace(heatmap_subplots[1], row=2, col=1) 
    fig.add_trace(heatmap_subplots[2], row=3, col=1)

    # Update xaxis and yaxis titles for each subplot
    fig.update_xaxes(title_text="Sol", row=3, col=1)
    fig.update_yaxes(title_text="Level", row=1, col=1)
    fig.update_yaxes(title_text="Level", row=2, col=1)
    fig.update_yaxes(title_text="Level", row=3, col=1)

    # Add vertical line at storm sol
    fig.add_vline(x=0, line=dict(color='red', dash='dash', width=1.5), layer="below", row=1, col=1)
    fig.add_vline(x=0, line=dict(color='red', dash='dash', width=1.5), layer="below", row=2, col=1)
    fig.add_vline(x=0, line=dict(color='red', dash='dash', width=1.5), layer="below", row=3, col=1)

    # Get unique levels per sol (as a proxy to number of profiles per sol)
    sol_profile_counts = storm_profiles_df[['rel_sol_int', 'Profile_identifier']].drop_duplicates(subset='Profile_identifier').groupby(['rel_sol_int']).count()['Profile_identifier']

    # Add count annotations
    annotation_y = -1
    for ind, val in sol_profile_counts.items():
        fig.add_annotation(x=ind, y=annotation_y,
			   text=f"<i>{str(val)}</i>",
			   showarrow=False,
			   xanchor='center',
			   yanchor='top',
			   textangle=45,
			   font={'family': 'Arial', 'size': 10, 'color': 'gray'}, 
			   xref=f'x1',
			   yref=f'y1')
    
    ###################################
    # Plot per-sol temp diffs
    # TODO: Could better break out temperature aggregation code as this repeats some of what's used above
    rc = (4, 1)

    # Aggregate data by temperature, get mean, take diff through time, and fill missing sols with NaNs
    mean_aggregated_data = aggregated_data.groupby(['rel_sol_int', 'level'], as_index=False).mean()
    pivot_table = mean_aggregated_data.pivot(index="level", columns="rel_sol_int", values="T")
    sols_range = range(pivot_table.columns.min(), pivot_table.columns.max() + 1)
    pivot_table_filled_diff = pivot_table.reindex(sols_range, axis='columns').diff(axis='columns')
    
    # Plot temperature diffs
    max_abs_value = np.max(np.abs(pivot_table_filled_diff.to_numpy()))
    fig.add_trace(go.Heatmap(z=pivot_table_filled_diff.values, x=pivot_table_filled_diff.columns, y=pivot_table_filled_diff.index, 
                             colorscale='RdBu_r', colorbar_title='Temp. Diff.', 
                             zmid=0, zmin=-max_abs_value, zmax=max_abs_value,
                             colorbar=dict(len=0.15, x=0.125, y=0.28)),
                             row=rc[0], col=rc[1])

    fig.update_xaxes(title_text="Sol", row=rc[0], col=rc[1])
    fig.update_yaxes(title_text="Level", row=rc[0], col=rc[1])

    # Add vertical line at storm sol
    fig.add_vline(x=0, line=dict(color='red', dash='dash', width=1.5), layer="below", row=rc[0], col=rc[1])
    
    ###################################
    # Add dust opacity plot
    rc = (1, 5)

    # Extract the storm profiles and control profiles
    storm_subset = storm_profiles_df[['rel_sol_int', 'Dust_column', 'level']]
    control_subset = control_profiles_df_smoothed[['rel_sol_int', 'Dust_column', 'level']]
    min_sol = min(storm_subset['rel_sol_int'].min(), control_subset['rel_sol_int'].min())
    max_sol = max(storm_subset['rel_sol_int'].max(), control_subset['rel_sol_int'].max())
    rel_sol_range = range(min_sol, max_sol + 1)
    
    # Plot the mean, median, min, max values
    storm_grouped = storm_subset.groupby(['rel_sol_int'])['Dust_column']
    storm_min = storm_grouped.min().reindex(rel_sol_range)
    storm_max= storm_grouped.max().reindex(rel_sol_range)
    storm_mean= storm_grouped.mean().reindex(rel_sol_range)
    #storm_median= storm_grouped.median().reindex(rel_sol_range)
    
    #TODO: Do we want to use the smoothed or unsmoothed control data here?
    controls_grouped = control_subset.groupby(['rel_sol_int'])['Dust_column']
    control_min = controls_grouped.min().reindex(rel_sol_range)
    control_max = controls_grouped.max().reindex(rel_sol_range)
    control_mean = controls_grouped.mean().reindex(rel_sol_range)
    #control_median = controls_grouped.median().reindex(rel_sol_range)
    
    # Save this for frequency analysis
    dust_opacity_perturbation_mean_trace = storm_mean - control_mean
    
    # Get sol inds where we can subtract control mean from storm mean
    # Note that for min/max calcs, we are finding the difference between the min/max relative to the control min/max (rather than a uniform value for both)
    for rel_sol in rel_sol_range:
        extracted_metadata[f'dust_opacity_min_sol_{rel_sol:03}'] = storm_min[rel_sol]
        extracted_metadata[f'dust_opacity_max_sol_{rel_sol:03}'] = storm_max[rel_sol]
        extracted_metadata[f'dust_opacity_mean_sol_{rel_sol:03}'] = storm_mean[rel_sol]
        extracted_metadata[f'perturbation_dust_opacity_min_sol_{rel_sol:03}'] = storm_min[rel_sol] - control_min[rel_sol]
        extracted_metadata[f'perturbation_dust_opacity_max_sol_{rel_sol:03}'] = storm_max[rel_sol] - control_max[rel_sol]
        extracted_metadata[f'perturbation_dust_opacity_mean_sol_{rel_sol:03}'] = storm_mean[rel_sol] - control_mean[rel_sol]
    
    # Adding storm opacity data
    fig.add_trace(go.Scatter(x=storm_min.index, y=storm_min, mode='lines+markers', showlegend=False, line=dict(color='blue', dash='dot', width=1), marker=dict(size=4, color='blue')), row=rc[0], col=rc[1])
    fig.add_trace(go.Scatter(x=storm_max.index, y=storm_max, mode='lines+markers', showlegend=False, line=dict(color='red', dash='dot', width=1), marker=dict(size=4, color='red')), row=rc[0], col=rc[1])
    fig.add_trace(go.Scatter(x=storm_mean.index, y=storm_mean, mode='lines+markers', showlegend=False, line=dict(color='black', dash='dot', width=1), marker=dict(size=4, color='black')), row=rc[0], col=rc[1])
    #fig.add_trace(go.Scatter(x=storm_median.index, y=storm_median, mode='lines', showlegend=False, line=dict(color='gray', dash='dot', width=1)), row=rc[0], col=rc[1])

    # Add perturbation data
    fig.add_trace(go.Scatter(x=storm_min.index, y=storm_min - control_min, mode='lines+markers', name='Min Perturbation', line=dict(color='blue'), marker=dict(size=4, color='blue')), row=rc[0], col=rc[1])
    fig.add_trace(go.Scatter(x=storm_max.index, y=storm_max - control_max, mode='lines+markers', name='Max Perturbation', line=dict(color='red'), marker=dict(size=4, color='red')), row=rc[0], col=rc[1])
    fig.add_trace(go.Scatter(x=storm_mean.index, y=storm_mean - control_mean, mode='lines+markers', name='Mean Perturbation', line=dict(color='black'), marker=dict(size=4, color='black')), row=rc[0], col=rc[1])
    #fig.add_trace(go.Scatter(x=storm_median.index, y=storm_median - control_median, mode='lines', name='Median Perturbation', line=dict(color='gray')), row=rc[0], col=rc[1])

    # Add horizontal line at zero and vertical line at storm sol
    fig.add_hline(y=0, line=dict(color='red', dash='dash'), showlegend=False, row=rc[0], col=rc[1])
    fig.add_vline(x=0, line=dict(color='red', dash='dash', width=1.5), layer="below", row=rc[0], col=rc[1])

    fig.update_xaxes(title_text="Rel. Sol", row=rc[0], col=rc[1], range=[storm_min.index.min(), storm_min.index.max()])
    fig.update_yaxes(title_text="Dust Opacity", row=rc[0], col=rc[1])

    ###################################
    # Add dust opacity stats
    rc = (2, 5)

    # Extract the storm profiles and control profiles, dropping duplicate rows and NaNs
    # Note that we use the *unsmoothed* controls here as we need a distribution of control profiles to compute the statistics
    storm_subset = (storm_profiles_df[['sol_int', 'rel_sol_int', 'Dust_column', 'Profile_identifier']]
                    .drop_duplicates().dropna(subset=['Dust_column']).copy())
    control_subset = (control_profiles_df[['sol_int', 'rel_sol_int', 'Dust_column', 'Profile_identifier']] 
                      .drop_duplicates().dropna(subset=['Dust_column']).copy())

    storm_subset = storm_profiles_df[['sol_int', 'rel_sol_int', 'Dust_column', 'Profile_identifier']].drop_duplicates().dropna(subset=['Dust_column'])
    control_subset = control_profiles_df[['sol_int', 'rel_sol_int', 'Dust_column', 'Profile_identifier']].drop_duplicates().dropna(subset=['Dust_column'])

    storm_subset['level'] = -1  # Add back dummy level for compute_level_statistics_windowed function
    control_subset['level'] = -1
    
    # Compute dust opacity statistics for plotting in next section of code
    dust_opacity_stats = compute_level_statistics_windowed(
        storm_subset, control_subset, 'Dust_column', 
        test=config.stats['test_statistic'], 
        min_points=config.stats['min_points'],
        rolling_window_storm=config.stats['storm_window_size'], 
        rolling_window_control=config.stats['controls_window_size'],
        timing_col='rel_sol_int')
 
    # Convert p-values to -log scale with clipping
    if 'pvalue' not in dust_opacity_stats.columns:
        dust_opacity_stats['pvalue'] = np.nan
        dust_opacity_stats['-log_pvalue'] = np.nan
    else:
        dust_opacity_stats['-log_pvalue'] = -np.log10(dust_opacity_stats['pvalue'].clip(config.stats['min_pvalue'], config.stats['max_pvalue']))
    
    for rel_sol in dust_opacity_stats['center_sol']:
        extracted_metadata[f'dust_opacity_pval_{rel_sol:03}'] = dust_opacity_stats[dust_opacity_stats['center_sol'] == rel_sol]['pvalue'].values[0]

    if not dust_opacity_stats.empty:
        # Reset the index to the sol for plotting reasons and reindex to fill in missing sols with NaNs (to break up line plots properly)
        dust_opacity_stats.set_index('center_sol', inplace=True)
        dust_opacity_stats = dust_opacity_stats.reindex(np.arange(dust_opacity_stats.index.min(), dust_opacity_stats.index.max() + 1))

        # Plot the p-values
        fig.add_trace(go.Scatter(x=dust_opacity_stats.index, y=dust_opacity_stats['-log_pvalue'], 
                                mode='lines+markers', 
                                name='Dust Opacity p-value', 
                                line=dict(width=1, color='purple'),
                                marker=dict(size=4, color='purple')),
                    row=rc[0], col=rc[1])

    # Add horizontal line at zero and vertical line at storm sol
    fig.add_vline(x=0, line=dict(color='red', dash='dash', width=1.5), layer="below", row=rc[0], col=rc[1])

    # Update axis labels, match x-axis range to the storm opacity data plot
    fig.update_xaxes(title_text="Rel. Sol", range=[storm_min.index.min(), storm_min.index.max()],
                     row=rc[0], col=rc[1])
    fig.update_yaxes(title_text="-log10(p-value)", range=-np.log10([config.stats['max_pvalue'] + 1e-2, config.stats['min_pvalue'] - 1e-6]),
                     row=rc[0], col=rc[1])

    ###################################
    # Add surface temperature plot
    rc = (3, 5)
    storm_subset = storm_profiles_df[['rel_sol_int', 'T_surf', 'level']]
    control_subset = control_profiles_df_smoothed[['rel_sol_int', 'T_surf', 'level']]
    min_sol = min(storm_subset['rel_sol_int'].min(), control_subset['rel_sol_int'].min())
    max_sol = max(storm_subset['rel_sol_int'].max(), control_subset['rel_sol_int'].max())
    rel_sol_range = range(min_sol, max_sol + 1)

    # Plot the mean, median, min, max values
    grouped = storm_subset.groupby(['rel_sol_int'])['T_surf']
    storm_min = grouped.min().reindex(rel_sol_range)
    storm_max = grouped.max().reindex(rel_sol_range)
    storm_mean = grouped.mean().reindex(rel_sol_range)
    #storm_median = grouped.median().reindex(rel_sol_range)
    
    #TODO: Do we want to use the smoothed or unsmoothed control data here?
    grouped = control_subset.groupby(['rel_sol_int'])['T_surf']
    control_min = grouped.min().reindex(rel_sol_range)
    control_max = grouped.max().reindex(rel_sol_range)
    control_mean = grouped.mean().reindex(rel_sol_range)
    #control_median = grouped.median().reindex(rel_sol_range)
    
    # Save this for frequency analysis
    surf_temp_perturbation_mean_trace = storm_mean - control_mean
    
    # Get sol inds where we can subtract control mean from storm mean
    for rel_sol in rel_sol_range:
        extracted_metadata[f'T_surf_min_sol_{rel_sol:03}'] = storm_min[rel_sol]
        extracted_metadata[f'T_surf_max_sol_{rel_sol:03}'] = storm_max[rel_sol]
        extracted_metadata[f'T_surf_mean_sol_{rel_sol:03}'] = storm_mean[rel_sol]
        extracted_metadata[f'perturbation_T_surf_min_sol_{rel_sol:03}'] = storm_min[rel_sol] - control_min[rel_sol]
        extracted_metadata[f'perturbation_T_surf_max_sol_{rel_sol:03}'] = storm_max[rel_sol] - control_max[rel_sol]
        extracted_metadata[f'perturbation_T_surf_mean_sol_{rel_sol:03}'] = storm_mean[rel_sol] - control_mean[rel_sol]
        
    # Add perturbation data. Only show perturbation as actual values are 100s of degrees K and can't be easily plotted with small perturbation differences
    fig.add_trace(go.Scatter(x=storm_min.index, y=storm_min - control_min, mode='lines+markers', name='Min Perturbation', line=dict(color='Indigo'), marker=dict(size=4, color='Indigo')), row=rc[0], col=rc[1])
    fig.add_trace(go.Scatter(x=storm_max.index, y=storm_max - control_max, mode='lines+markers', name='Max Perturbation', line=dict(color='DarkOrange'), marker=dict(size=4, color='DarkOrange')), row=rc[0], col=rc[1])
    fig.add_trace(go.Scatter(x=storm_mean.index, y=storm_mean - control_mean, mode='lines+markers', name='Mean Perturbation', line=dict(color='black'), marker=dict(size=4, color='black')), row=rc[0], col=rc[1])
    #fig.add_trace(go.Scatter(x=storm_median.index, y=storm_median - control_median, mode='lines', name='Median Perturbation', line=dict(color='gray')), row=rc[0], col=rc[1])

    # Add horizontal line at zero and vertical line at storm sol
    fig.add_hline(y=0, line=dict(color='red', dash='dash'), showlegend=False, row=rc[0], col=rc[1])
    fig.add_vline(x=0, line=dict(color='red', dash='dash', width=1.5), layer="below", row=rc[0], col=rc[1])

    fig.update_xaxes(title_text="Rel. Sol", row=rc[0], col=rc[1], range=[storm_min.index.min(), storm_min.index.max()])
    fig.update_yaxes(title_text="Surface Temperature (K)", row=rc[0], col=rc[1])

    ###################################
    # Add surface temperature p-value plot
    rc = (4, 5)

    # Extract the storm profiles and control profiles, dropping duplicate rows and NaNs
    storm_subset = (
        storm_profiles_df[['sol_int', 'rel_sol_int', 'T_surf', 'Profile_identifier']]
        .drop_duplicates()
        .dropna(subset=['T_surf'])
        .copy()
    )
    control_subset = (
        control_profiles_df[['sol_int', 'rel_sol_int', 'T_surf', 'Profile_identifier']]
        .drop_duplicates()
        .dropna(subset=['T_surf'])
        .copy()
    )
    storm_subset['level'] = -1  # Add back dummy level for compute_level_statistics_windowed function
    control_subset['level'] = -1

    # Compute surface temperature statistics for plotting in next section of code
    surf_temp_stats = compute_level_statistics_windowed(
        storm_subset, control_subset, 'T_surf', 
        test=config.stats['test_statistic'], 
        min_points=config.stats['min_points'],
        rolling_window_storm=config.stats['storm_window_size'], 
        rolling_window_control=config.stats['controls_window_size'],
        timing_col='rel_sol_int')

    if 'pvalue' not in surf_temp_stats.columns:
        surf_temp_stats['pvalue'] = np.nan
        surf_temp_stats['-log_pvalue'] = np.nan
    else:
        surf_temp_stats['-log_pvalue'] = -np.log10(surf_temp_stats['pvalue'].clip(config.stats['min_pvalue'], config.stats['max_pvalue']))

    for rel_sol in surf_temp_stats['center_sol']:
        extracted_metadata[f'T_surf_pval_{rel_sol:03}'] = surf_temp_stats[surf_temp_stats['center_sol'] == rel_sol]['pvalue'].values[0]
    
    # Reset the index to the sol for plotting reasons and reindex to fill in missing sols with NaNs (to break up line plots properly)
    if not surf_temp_stats.empty:
        surf_temp_stats.set_index('center_sol', inplace=True)
        surf_temp_stats = surf_temp_stats.reindex(np.arange(surf_temp_stats.index.min(), surf_temp_stats.index.max() + 1))

        # Plot the p-values
        fig.add_trace(go.Scatter(x=surf_temp_stats.index, y=surf_temp_stats['-log_pvalue'],
                                 mode='lines+markers', 
                                 name='Surface Temperature p-value', 
                                 line=dict(width=1, color='purple'),
                                 marker=dict(size=4, color='purple')),
                      row=rc[0], col=rc[1])

    # Add vertical line at storm sol
    fig.add_vline(x=0, line=dict(color='red', dash='dash', width=1.5), layer="below", row=rc[0], col=rc[1])

    # Update axis labels, match x-axis range to the surface temperature data plot
    fig.update_xaxes(title_text="Rel. Sol", range=[storm_mean.index.min(), storm_mean.index.max()],
                     row=rc[0], col=rc[1])
    fig.update_yaxes(title_text="-log10(p-value)", range=-np.log10([config.stats['max_pvalue'] + 1e-2, config.stats['min_pvalue'] - 1e-6]), 
                     row=rc[0], col=rc[1])

    ###################################
    # Add plots Lomb-Scargle amplitude and phase
    rc_amp = (3, 4)
    rc_phase = (4, 4)
    ang_freqs = np.linspace(config.freq_analysis['min_freq'],
                            config.freq_analysis['max_freq'],
                            config.freq_analysis['n_freqs'])
    
    ##############
    # Dust opacity
    dust_opacity_perturbation_mean_trace_cleaned = dust_opacity_perturbation_mean_trace.dropna()
    if len(dust_opacity_perturbation_mean_trace_cleaned) >= config.freq_analysis['min_data_pts']:
        dust_pgram = lombscargle(dust_opacity_perturbation_mean_trace_cleaned.index.values, 
                                dust_opacity_perturbation_mean_trace_cleaned.values,
                                ang_freqs,
                                normalize='amplitude',  # Gives amplitude and phase info
                                floating_mean=True)
        fig.add_trace(go.Scatter(x=ang_freqs, y=np.abs(dust_pgram), mode='lines', name='Dust Opacity Amplitude', line=dict(color='orange')), row=rc_amp[0], col=rc_amp[1])
        fig.add_trace(go.Scatter(x=ang_freqs, y=np.angle(dust_pgram), mode='lines', name='Dust Opacity Angle', opacity=0.5, line=dict(color='orange', dash='dot')), row=rc_phase[0], col=rc_phase[1])
        fig.update_yaxes(title_text="Dust Opacity Amplitude", 
                         title_font_color="darkorange",
                         row=rc_amp[0], col=rc_amp[1])

    #####################
    # Surface temperature
    surf_temp_perturbation_mean_trace_cleaned = surf_temp_perturbation_mean_trace.dropna()
    if len(surf_temp_perturbation_mean_trace_cleaned) >= config.freq_analysis['min_data_pts']:
        temp_pgram = lombscargle(surf_temp_perturbation_mean_trace_cleaned.index.values, 
                                surf_temp_perturbation_mean_trace_cleaned.values,
                                ang_freqs,
                                normalize='amplitude',  # Gives amplitude and phase info
                                floating_mean=True)

        # Add surface temperature amplitude on secondary y-axis
        fig.add_trace(go.Scatter(x=ang_freqs, y=np.abs(temp_pgram), mode='lines', name='Surf. Temp. Amplitude', line=dict(color='purple')), secondary_y=True, row=rc_amp[0], col=rc_amp[1])
        fig.add_trace(go.Scatter(x=ang_freqs, y=np.angle(temp_pgram), mode='lines', name='Surf. Temp. Angle', opacity=0.5, line=dict(color='purple', dash='dot')), row=rc_phase[0], col=rc_phase[1])
        
        # Update the secondary y-axis properties
        fig.update_yaxes(title_text="Surf. Temp. Amplitude", 
                         title_font_color="purple",
                         secondary_y=True,
                         row=rc_amp[0], col=rc_amp[1])
        
    if len(dust_opacity_perturbation_mean_trace_cleaned) >= config.freq_analysis['min_data_pts'] and len(surf_temp_perturbation_mean_trace_cleaned) >= config.freq_analysis['min_data_pts']:
        fig.add_trace(go.Scatter(x=ang_freqs, y=np.angle(dust_pgram - temp_pgram), mode='lines', name='Phase Diff.', line=dict(color='red')), row=rc_phase[0], col=rc_phase[1])

    fig.update_xaxes(title_text="Freq [rad/sol]", row=rc_amp[0], col=rc_amp[1])
    fig.update_xaxes(title_text="Freq [rad/sol]", row=rc_phase[0], col=rc_phase[1])
    fig.update_yaxes(title_text="Phase (rad)", row=rc_phase[0], col=rc_phase[1], range=[-np.pi, np.pi])
    
    ###################################
    # Add gradients quartiles
    rc = (5, 2)

    merged_ddr_filtered_subset = storm_profiles_df[['rel_sol_int', 'T', 'level']].copy()
    # Ensure deterministic order before computing differences across levels
    merged_ddr_filtered_subset = merged_ddr_filtered_subset.sort_values(['rel_sol_int', 'level'])

    # Compute gradient as within-sol diff across increasing level
    temp_gradients = merged_ddr_filtered_subset.copy()
    temp_gradients['T_grad'] = temp_gradients.groupby('rel_sol_int')['T'].diff()
    
    #####
    grad_min_level, grad_max_level = 0, 40  # Filter inclusive

    quartile_temp_grad = temp_gradients.copy()
    quartile_temp_grad = quartile_temp_grad.loc[(grad_min_level <= quartile_temp_grad['level']) & (quartile_temp_grad['level'] <= grad_max_level), :]
    grouped = quartile_temp_grad.groupby('rel_sol_int')['T_grad'].apply(list)

    # Add each quartile component
    for rel_sol_int, group in grouped.items():
        trace = go.Box(y=group, x=[rel_sol_int] * len(group), name=rel_sol_int, width=0.5, marker_color='#636EFA', showlegend=False)
        fig.add_trace(trace, row=rc[0], col=rc[1])
        
    fig.update_xaxes(title_text="Rel. Sol", row=rc[0], col=rc[1])
    fig.update_yaxes(title_text="Temp. Gradient", row=rc[0], col=rc[1])
    fig.add_vline(x=0, line=dict(color='red', dash='dash', width=1.5), layer="below", row=rc[0], col=rc[1])

    ###################################
    # Add gradient plots
    rc = (5, 1)
    df_agg_cols = ['T_grad']
    aggregated_data = aggregate_heatmap_data(temp_gradients, cols=df_agg_cols, uniform_MY_required=False, group_by_col_name='rel_sol_int')
    graph_objs = generate_heatmap_graph_objs_sol(aggregated_data, col_colorbar_ys=[0.07], cols=df_agg_cols, 
                                                 center_colorbar=True, colorscale='RdBu_r', colorbar_xpos=0.125,
                                                 sol_col_name='rel_sol_int')

    # Add temperature heatmap
    fig.add_trace(graph_objs[0], row=rc[0], col=rc[1])

    # Add vertical line at storm sol
    fig.add_vline(x=0, line=dict(color='red', dash='dash', width=1.5), layer="below", row=rc[0], col=rc[1])

    # Update xaxis and yaxis titles for each subplot
    fig.update_xaxes(title_text="Sol", row=rc[0], col=rc[1])
    fig.update_yaxes(title_text="Level", row=rc[0], col=rc[1])

    ###################################
    # Add geospatial context plot
    rc = (5, 4)
    geospatial_fig = generate_geospatial_overview(storm_metadata, storm_profiles_df)

    # TODO: Fix hack where we export to image
    try:
        img_bytes = safe_plotly_to_image(geospatial_fig, storm_metadata['storm_db_id'], scale=1.5, format='png')
        buf = io.BytesIO(img_bytes)
        pil_image = Image.open(buf)
        fig.add_trace(px.imshow(pil_image).data[0], row=rc[0], col=rc[1])
        fig.update_xaxes(showticklabels=False, row=rc[0], col=rc[1])
        fig.update_yaxes(showticklabels=False, row=rc[0], col=rc[1])
    except Exception as e:
        logging.error(f"Error generating geospatial plot for DB ID {storm_metadata['storm_db_id']}: {e}. Skipping.")
    
    ###################################
    # Plot atmospheric profile p-values
    # Add colorbar plot showing distribution distance between storm/controls
    rc_t = (1, 4)  # Position of temperature subplot
    rc_dust = (2, 4)  # Position of dust subplot
    
    ####
    # Single, static window
    if config.stats['storm_window_size'] is None:
        temp_stats = compute_level_statistics(storm_profiles_df, control_profiles_df, 'T', test=config.stats['test_statistic'], min_points=config.stats['min_points'])
        dust_stats = compute_level_statistics(storm_profiles_df, control_profiles_df, 'Dust', test=config.stats['test_statistic'], min_points=config.stats['min_points'])

    ####
    # Compute statistics with rolling window
    else:
        storm_profiles_df_filtered = storm_profiles_df[['rel_sol_int', 'T', 'Dust', 'level']].copy()
        control_profiles_df_filtered = control_profiles_df[['rel_sol_int', 'T', 'Dust', 'level']].copy()

        temp_stats = compute_level_statistics_windowed(
            storm_profiles_df_filtered, control_profiles_df_filtered, 'T', 
            test=config.stats['test_statistic'], 
            min_points=config.stats['min_points'],
            rolling_window_storm=config.stats['storm_window_size'], 
            rolling_window_control=config.stats['controls_window_size'],
            timing_col='rel_sol_int')
        dust_stats = compute_level_statistics_windowed(
            storm_profiles_df_filtered, control_profiles_df_filtered, 'Dust', 
            test=config.stats['test_statistic'], 
            min_points=config.stats['min_points'],
            rolling_window_storm=config.stats['storm_window_size'], 
            rolling_window_control=config.stats['controls_window_size'],
            timing_col='rel_sol_int')
        

    # Process rolling statistics for metadata
    def extract_pval_stats(df, compute_column):
        """Helper to extract rolling statistics (usually for temp/dust)"""
        
        temp_metadata = {}
        cc = compute_column.lower()
        
        # Filter levels <= 40
        if 'level' in df.columns:
            sub_L40_df = df[df['level'] <= config.stats['max_level']]
            if not sub_L40_df.empty:
                # Group by center sol
                grouped = sub_L40_df.groupby('center_sol')
                
                # Save out p-values and indicate sol window size
                for center_sol, group in grouped:
                    key_stub = f'subL{config.stats["max_level"]}_{cc}_sol_window_{config.stats["storm_window_size"]:02d}_sol_{center_sol:03d}'

                    temp_metadata[f'pval_min_{key_stub}'] = group['pvalue'].min()
                    temp_metadata[f'pval_median_{key_stub}'] = group['pvalue'].median()
                    temp_metadata[f'pval_mean_{key_stub}'] = group['pvalue'].mean()
                    
        return temp_metadata

    # Extract metadata for Temperature and Dust
    extracted_metadata.update(extract_pval_stats(temp_stats, 'T'))
    extracted_metadata.update(extract_pval_stats(dust_stats, 'Dust'))
    

    def plot_pvalues(df, fig, row, col, col_colorbar_y, col_colorbar_x=0.74, min_pvalue=1e-5, max_pvalue=0.1):
        """Plot p-values within plotly subplot."""

        if df.empty:
            return

        df['-log_pvalue'] = -np.log10(df['pvalue'].clip(min_pvalue, max_pvalue))

        aggregated_data = aggregate_heatmap_data(df, cols=['-log_pvalue'], uniform_MY_required=False, group_by_col_name='rel_sol_int')
        pvalue_subplots = generate_heatmap_graph_objs_sol(aggregated_data, col_colorbar_ys=[col_colorbar_y],
                                                          cols=['-log_pvalue'], colorscale='Plasma', colorbar_xpos=col_colorbar_x,
                                                          sol_col_name='rel_sol_int') 
        fig.add_trace(pvalue_subplots[0], row=row, col=col) 

        # Update xaxis and yaxis titles for each subplot
        fig.update_xaxes(title_text=f"Rel. Sol", row=row, col=col)
        fig.update_yaxes(title_text="Level", range=[-0.1, 105.1], row=row, col=col)
        
        # Set colorbar range to match the -log of min and max p-values
        fig.update_traces(selector=dict(type='heatmap'), zmin=-np.log10(max_pvalue), zmax=-np.log10(min_pvalue), row=row, col=col)

    # Rename these in prep for `plot_pvalues`, which will call `aggregate_heatmap_data`
    temp_stats.rename({'center_sol': 'rel_sol_int'}, axis='columns', inplace=True)
    dust_stats.rename({'center_sol': 'rel_sol_int'}, axis='columns', inplace=True)
    plot_pvalues(temp_stats, fig, rc_t[0], rc_t[1], col_colorbar_y=0.94, min_pvalue=config.stats['min_pvalue'], max_pvalue=config.stats['max_pvalue'])
    plot_pvalues(dust_stats, fig, rc_dust[0], rc_dust[1], col_colorbar_y=0.725, min_pvalue=config.stats['min_pvalue'], max_pvalue=config.stats['max_pvalue'])
    
    # Add annotation above the top plot
    fig.add_vline(x=0, line=dict(color='red', dash='dash', width=1.5), layer="below", row=rc_t[0], col=rc_t[1])
    fig.add_vline(x=0, line=dict(color='red', dash='dash', width=1.5), layer="below", row=rc_dust[0], col=rc_dust[1])

    ###################################
    # Add bar plot showing Obs_qual vals
    rc = (5, 3)

    # Preprocess the data to ensure unique rows based on Profile_identifier
    storm_profiles_df_unique = storm_profiles_df.drop_duplicates(subset='Profile_identifier')
    control_profiles_df_unique = control_profiles_df.drop_duplicates(subset='Profile_identifier')

    # Define the bins and aggregate the data
    bins = {'0, 1, 7, 10, 11, 17': [0, 1, 7, 10, 11, 17], 
            '2, 3': [2, 3], 
            '4, 5': [4, 5]}

    def aggregate_obs_qual(df, bins):
        counts = {}
        if 'Obs_qual' in df.columns:
            for bin_name, bin_values in bins.items():
                counts[bin_name] = df['Obs_qual'].isin(bin_values).sum()
        return counts

    # Create the bar plot, add to figure 
    storm_counts = aggregate_obs_qual(storm_profiles_df_unique, bins)
    control_counts = aggregate_obs_qual(control_profiles_df_unique, bins)

    storm_bar = go.Bar(x=list(storm_counts.keys()), y=list(storm_counts.values()), name='Storm Profiles', marker_color='#d62728')
    control_bar = go.Bar(x=list(control_counts.keys()), y=list(control_counts.values()), name='Control Profiles', marker_color='#1f77b4')
    fig.add_trace(storm_bar, row=rc[0], col=rc[1])
    fig.add_trace(control_bar, row=rc[0], col=rc[1])

    fig.update_xaxes(row=rc[0], col=rc[1], title_text='Obs_qual Bins')
    fig.update_yaxes(row=rc[0], col=rc[1], title_text='Count')
    
    ###################################
    # Add plots showing temp/dust whole-profile measurement failures (surface temp and dust column)
    rc = (3, 2)

    # Get levels where T_surf is NA, then grab the Profile_identifier and relative sol
    ddr1_data = storm_profiles_df.loc[:, ['T_surf', 'Dust_column', 'Profile_identifier', 'rel_sol_int']].drop_duplicates(subset='Profile_identifier')

    # Plot missing surface temps. We use size (instead of count) to make sure NaN values are included. 
    # Reindex to fill in missing values with nan so they plot properly (no interpolation) when possible
    tsurf_missing = ddr1_data[ddr1_data['T_surf'].isna()].groupby('rel_sol_int').size()
    if tsurf_missing.index.nunique() > 1:
        tsurf_missing = tsurf_missing.reindex(np.arange(tsurf_missing.index.min(), tsurf_missing.index.max() + 1))

    tsurf_valid = ddr1_data[~ddr1_data['T_surf'].isna()].groupby('rel_sol_int').size()
    if tsurf_valid.index.nunique() > 1:
        tsurf_valid = tsurf_valid.reindex(np.arange(tsurf_valid.index.min(), tsurf_valid.index.max() + 1))
                                                  
    fig.add_trace(go.Scatter(x=tsurf_missing.index, y=tsurf_missing.values,
                             mode='lines+markers',
                             name='Temp. Surface NaN',
                             line=dict(color='red'),
                             marker=dict(size=4, color='red')),
                             row=rc[0], col=rc[1])

    fig.add_trace(go.Scatter(x=tsurf_valid.index, y=tsurf_valid.values,
                             mode='lines+markers',
                             name='Temp. Surface Valid',
                             line=dict(color='gray'),
                             marker=dict(size=4, color='gray')),
                             row=rc[0], col=rc[1])
    fig.update_xaxes(title_text="Rel. Sol", row=rc[0], col=rc[1])
    fig.update_yaxes(title_text="Number of Profiles", row=rc[0], col=rc[1])

    # Add vertical line at storm sol
    fig.add_vline(x=0, line=dict(color='red', dash='dash', width=1.5), layer="below", row=rc[0], col=rc[1])

    # Record valid and invalid dust column opacities
    for rel_sol_int in tsurf_missing.loc[~tsurf_missing.isna()].index:
        extracted_metadata[f'T_surf_failure_{rel_sol_int:03}'] = tsurf_missing[rel_sol_int]
    for rel_sol_int in tsurf_valid.loc[~tsurf_valid.isna()].index:
        extracted_metadata[f'T_surf_valid_{rel_sol_int:03}'] = tsurf_valid[rel_sol_int]
    ###########
    rc = (4, 2)

    # Plot missing dust columns. We use size (instead of count) to make sure NaN values are included. 
    # Reindex to fill in missing values with nan so they plot properly (no interpolation) when possible
    dust_missing = ddr1_data[ddr1_data['Dust_column'].isna()].groupby('rel_sol_int').size()
    if dust_missing.index.nunique() > 1:
        dust_missing = dust_missing.reindex(np.arange(dust_missing.index.min(), dust_missing.index.max() + 1))

    dust_valid =  ddr1_data[~ddr1_data['Dust_column'].isna()].groupby('rel_sol_int').size()
    if dust_valid.index.nunique() > 1:
        dust_valid = dust_valid.reindex(np.arange(dust_valid.index.min(), dust_valid.index.max() + 1))

    fig.add_trace(go.Scatter(x=dust_missing.index, y=dust_missing.values,
                             mode='lines+markers',
                             name='Dust Column NaN',
                             line=dict(color='red'),
                             marker=dict(size=4, color='red')),
                             row=rc[0], col=rc[1])
    fig.add_trace(go.Scatter(x=dust_valid.index, y=dust_valid.values,
                             mode='lines+markers',
                             name='Dust Column Valid',
                             line=dict(color='gray'),
                             marker=dict(size=4, color='gray')),
                             row=rc[0], col=rc[1])
    
    # Update the subplot axes
    fig.update_xaxes(title_text="Rel. Sol", row=rc[0], col=rc[1])
    fig.update_yaxes(title_text="Number of Profiles", row=rc[0], col=rc[1])

    # Add vertical line at storm sol
    fig.add_vline(x=0, line=dict(color='red', dash='dash', width=1.5), layer="below", row=rc[0], col=rc[1])
    
    # Record valid and invalid dust column opacities
    for rel_sol_int in dust_missing.loc[~dust_missing.isna()].index:
        extracted_metadata[f'dust_column_opacity_failure_sol_{rel_sol_int:03}'] = dust_missing[rel_sol_int]
    for rel_sol_int in dust_valid.loc[~dust_valid.isna()].index:
        extracted_metadata[f'dust_column_opacity_valid_sol_{rel_sol_int:03}'] = dust_valid[rel_sol_int]
        
    ###################################
    # Add plots showing how low dust/temp retrievals went until they failed

    # Get levels where T/Dust is valid, then grab the level, Profile_identifier, and relative sol
    temp_valid_profs = storm_profiles_df.loc[~storm_profiles_df['T'].isna(), ['level', 'Profile_identifier', 'rel_sol_int']]
    dust_valid_profs = storm_profiles_df.loc[~storm_profiles_df['Dust'].isna(), ['level', 'Profile_identifier', 'rel_sol_int']]
    
    # Process the above to get the lowest level for each profile where there is an valid value
    if not temp_valid_profs.empty:
        temp_min_valid_levels = temp_valid_profs.groupby(['Profile_identifier', 'rel_sol_int'])['level'].min().reset_index()
    else:
        temp_min_valid_levels = pd.DataFrame(columns=['Profile_identifier', 'rel_sol_int', 'level'])

    if not dust_valid_profs.empty:
        dust_min_valid_levels = dust_valid_profs.groupby(['Profile_identifier', 'rel_sol_int'])['level'].min().reset_index()
    else:
        dust_min_valid_levels = pd.DataFrame(columns=['Profile_identifier', 'rel_sol_int', 'level'])

    ###############
    # Temp failures - Box plots
    rc = (3, 3)

    if not temp_min_valid_levels.empty:
        temp_grouped = temp_min_valid_levels.groupby('rel_sol_int')['level'].apply(np.array)
        
        # Add each box plot for temperature failures
        for rel_sol_int, failure_levels in temp_grouped.items():
            # Convert to failure levels (min valid level - 1) where 0 is the lowest possible level (retrieved to surface)
            failure_levels_adjusted = np.maximum(failure_levels - 1, 0)

            trace = go.Box(y=failure_levels_adjusted, x=[rel_sol_int] * len(failure_levels_adjusted), width=0.5, name=rel_sol_int, marker_color='purple', showlegend=False)
            fig.add_trace(trace, row=rc[0], col=rc[1])
            
            # Save out to our extracted metadata for saving in filtersheet CSV
            extracted_metadata[f'T_retrieval_failure_level_min_sol_{rel_sol_int:03}'] = np.min(failure_levels_adjusted)
            extracted_metadata[f'T_retrieval_failure_level_median_sol_{rel_sol_int:03}'] = np.round(np.median(failure_levels_adjusted), 1)
            extracted_metadata[f'T_retrieval_failure_level_mean_sol_{rel_sol_int:03}'] = np.round(np.mean(failure_levels_adjusted), 2)
            extracted_metadata[f'T_retrieval_failure_level_max_sol_{rel_sol_int:03}'] = np.max(failure_levels_adjusted)
            
        fig.add_vline(x=0, line=dict(color='red', dash='dash', width=1.5), layer="below", row=rc[0], col=rc[1])
         
    fig.update_xaxes(title_text="Rel. Sol", row=rc[0], col=rc[1])
    fig.update_yaxes(title_text="Highest Failure Level", row=rc[0], col=rc[1])
    
    ###############
    # Dust failures - Box plots
    rc = (4, 3)

    if not dust_min_valid_levels.empty:
        dust_grouped = dust_min_valid_levels.groupby('rel_sol_int')['level'].apply(np.array)
        
        # Add each box plot for dust failures
        for rel_sol_int, failure_levels in dust_grouped.items():
            # Convert to failure levels (min valid level - 1)
            failure_levels_adjusted = failure_levels - 1
            trace = go.Box(y=failure_levels_adjusted, x=[rel_sol_int] * len(failure_levels_adjusted), width=0.5, name=rel_sol_int, marker_color='darkorange', showlegend=False)
            fig.add_trace(trace, row=rc[0], col=rc[1])

            # Save out to our extracted metadata for saving in filtersheet CSV
            extracted_metadata[f'Dust_retrieval_failure_level_min_sol_{rel_sol_int:03}'] = np.min(failure_levels_adjusted)
            extracted_metadata[f'Dust_retrieval_failure_level_median_sol_{rel_sol_int:03}'] = np.round(np.median(failure_levels_adjusted), 1)
            extracted_metadata[f'Dust_retrieval_failure_level_mean_sol_{rel_sol_int:03}'] = np.round(np.mean(failure_levels_adjusted), 2)
            extracted_metadata[f'Dust_retrieval_failure_level_max_sol_{rel_sol_int:03}'] = np.max(failure_levels_adjusted)
            
        fig.add_vline(x=0, line=dict(color='red', dash='dash', width=1.5), layer="below", row=rc[0], col=rc[1])

    # Update the subplot axes
    fig.update_xaxes(title_text="Rel. Sol", row=rc[0], col=rc[1])
    fig.update_yaxes(title_text="Highest Failure Level", row=rc[0], col=rc[1])

    ###################################
    # Add plots showing controls
    rc_t = (1, 2)
    rc_dust = (2, 2)
    
    graph_objs = generate_heatmap_graph_objs_sol(control_profiles_df_smoothed, col_colorbar_ys=[0.94, 0.725], 
                                                 cols=['T', 'Dust'], colorbar_xpos=0.33, sol_col_name='rel_sol_int')

    # Add temperature heatmap
    fig.add_trace(graph_objs[0], row=rc_t[0], col=rc_t[1])
    fig.add_trace(graph_objs[1], row=rc_dust[0], col=rc_dust[1])

    # Update xaxis and yaxis titles for each subplot
    fig.update_xaxes(title_text="Rel. Sol", row=rc_t[0], col=rc_t[1])
    fig.update_yaxes(title_text="Level", row=rc_t[0], col=rc_t[1])
    fig.update_xaxes(title_text="Rel. Sol", row=rc_dust[0], col=rc_dust[1])
    fig.update_yaxes(title_text="Level", row=rc_dust[0], col=rc_dust[1])

    ###################################
    # Add plots showing storm - control
    col=3

    # Aggregate data by sol_int and level and set multindex. Do this for min, max, and mean
    aggregated_min_storm_data_multindex = aggregate_heatmap_data(storm_profiles_df, cols=['T', 'Dust'], uniform_MY_required=False, group_by_col_name='rel_sol_int', agg_func='min').set_index(['rel_sol_int', 'level'])  # Set multindex
    aggregated_max_storm_data_multindex = aggregate_heatmap_data(storm_profiles_df, cols=['T', 'Dust'], uniform_MY_required=False, group_by_col_name='rel_sol_int', agg_func='max').set_index(['rel_sol_int', 'level'])  # Set multindex
    aggregated_mean_storm_data_multindex = aggregate_heatmap_data(storm_profiles_df, cols=['T', 'Dust'], uniform_MY_required=False, group_by_col_name='rel_sol_int', agg_func='mean').set_index(['rel_sol_int', 'level'])  # Set multindex
    control_profiles_df_smoothed_multindex = control_profiles_df_smoothed.drop(columns=['T_surf', 'Dust_column']).set_index(['rel_sol_int', 'level'])

    # With indices in place, subtract the mean control profile from each sol's mean profile in storm data 
    perturbation_min_storm_data_multindex = (aggregated_min_storm_data_multindex - control_profiles_df_smoothed_multindex).rename(columns={'T': 'T_diff', 'Dust': 'Dust_diff'})
    perturbation_max_storm_data_multindex = (aggregated_max_storm_data_multindex - control_profiles_df_smoothed_multindex).rename(columns={'T': 'T_diff', 'Dust': 'Dust_diff'})
    perturbation_mean_storm_data_multindex = (aggregated_mean_storm_data_multindex - control_profiles_df_smoothed_multindex).rename(columns={'T': 'T_diff', 'Dust': 'Dust_diff'})

    aggregated_min_storm_data = aggregated_min_storm_data_multindex.reset_index()
    aggregated_max_storm_data = aggregated_max_storm_data_multindex.reset_index()
    aggregated_mean_storm_data = aggregated_mean_storm_data_multindex.reset_index()
    
    perturbation_min_storm_data = perturbation_min_storm_data_multindex.reset_index()
    perturbation_max_storm_data = perturbation_max_storm_data_multindex.reset_index()
    perturbation_mean_storm_data = perturbation_mean_storm_data_multindex.reset_index()

    # Plot just the mean
    heatmap_subplots_diffs = generate_heatmap_graph_objs_sol(perturbation_mean_storm_data, col_colorbar_ys = [0.94, 0.725], cols=['T_diff', 'Dust_diff'], 
                                                             center_colorbar=True, colorscale='RdBu_r', colorbar_xpos=0.53, sol_col_name='rel_sol_int') 
    fig.add_trace(heatmap_subplots_diffs[0], row=1, col=col)
    fig.add_trace(heatmap_subplots_diffs[1], row=2, col=col)

    # Update xaxis and yaxis titles for each subplot
    fig.update_xaxes(title_text="Rel. Sol", row=2, col=col)
    fig.update_yaxes(title_text="Level", row=1, col=col)
    fig.update_yaxes(title_text="Level", row=2, col=col)

    # Add vertical line at storm sol
    fig.add_vline(x=0, line=dict(color='red', dash='dash', width=1.5), layer="below", row=1, col=col)
    fig.add_vline(x=0, line=dict(color='red', dash='dash', width=1.5), layer="below", row=2, col=col)
    
    ############################### 
    # Extract original dust/temp data for specific levels

    for li in config.per_level_analysis['level_to_extract']:
        level_data_mean = aggregated_mean_storm_data.loc[aggregated_mean_storm_data['level']==li, :].groupby('rel_sol_int')
        level_data_min = aggregated_min_storm_data.loc[aggregated_min_storm_data['level']==li, :].groupby('rel_sol_int')
        level_data_max = aggregated_max_storm_data.loc[aggregated_max_storm_data['level']==li, :].groupby('rel_sol_int')
        
        for rel_sol_int, data in level_data_mean:
            extracted_metadata[f'T_mean_L{li:02d}_sol_{rel_sol_int:03d}'] = data['T'].mean()
            extracted_metadata[f'Dust_mean_L{li:02d}_sol_{rel_sol_int:03d}'] = data['Dust'].mean()
        
        for rel_sol_int, data in level_data_min:
            extracted_metadata[f'T_min_L{li:02d}_sol_{rel_sol_int:03d}'] = data['T'].min()
            extracted_metadata[f'Dust_min_L{li:02d}_sol_{rel_sol_int:03d}'] = data['Dust'].min()
            
        for rel_sol_int, data in level_data_max:
            extracted_metadata[f'T_max_L{li:02d}_sol_{rel_sol_int:03d}'] = data['T'].max()
            extracted_metadata[f'Dust_max_L{li:02d}_sol_{rel_sol_int:03d}'] = data['Dust'].max()
    
    '''
    # TODO: If using this, need to decide if calcs should be made on original disagregated data or the min/max/mean after aggregation
    # Extract original data for ranges (0-20, 20-40)
    aggregated_data_L0_20 = aggregated_storm_data.loc[aggregated_storm_data['level']<=20, :].groupby('rel_sol')
    for rel_sol, data in aggregated_data_L0_20:
        extracted_metadata[f'T_min_L00_20_sol_{rel_sol:03d}'] = data['T'].min()
        extracted_metadata[f'T_max_L00_20_sol_{rel_sol:03d}'] = data['T'].max()
        extracted_metadata[f'T_mean_L00_20_sol_{rel_sol:03d}'] = data['T'].mean()
        extracted_metadata[f'Dust_min_L00_20_sol_{rel_sol:03d}'] = data['Dust'].min()
        extracted_metadata[f'Dust_max_L00_20_sol_{rel_sol:03d}'] = data['Dust'].max()
        extracted_metadata[f'Dust_mean_L00_20_sol_{rel_sol:03d}'] = data['Dust'].mean()
    
    aggregated_data_L20_40 = aggregated_storm_data.loc[(aggregated_storm_data['level']>20) & (aggregated_storm_data['level']<=40), :].groupby('rel_sol')
    for rel_sol, data in aggregated_data_L20_40:
        extracted_metadata[f'T_min_L20_40_sol_{rel_sol:03d}'] = data['T'].min()
        extracted_metadata[f'T_max_L20_40_sol_{rel_sol:03d}'] = data['T'].max()
        extracted_metadata[f'T_mean_L20_40_sol_{rel_sol:03d}'] = data['T'].mean()
        extracted_metadata[f'Dust_min_L20_40_sol_{rel_sol:03d}'] = data['Dust'].min()
        extracted_metadata[f'Dust_max_L20_40_sol_{rel_sol:03d}'] = data['Dust'].max()
        extracted_metadata[f'Dust_mean_L20_40_sol_{rel_sol:03d}'] = data['Dust'].mean()
    '''
    ############################### 
    # Extract perturbation data for specific levels

    for li in config.per_level_analysis['level_to_extract']:
        level_data_mean = perturbation_mean_storm_data.loc[perturbation_mean_storm_data['level']==li, :].groupby('rel_sol_int')
        level_data_min = perturbation_min_storm_data.loc[perturbation_min_storm_data['level']==li, :].groupby('rel_sol_int')
        level_data_max = perturbation_max_storm_data.loc[perturbation_max_storm_data['level']==li, :].groupby('rel_sol_int')
        
        for rel_sol_int, data in level_data_mean:
            extracted_metadata[f'perturbation_T_mean_L{li:02d}_sol_{rel_sol_int:03d}'] = data['T_diff'].mean()
            extracted_metadata[f'perturbation_Dust_mean_L{li:02d}_sol_{rel_sol_int:03d}'] = data['Dust_diff'].mean()
        
        for rel_sol_int, data in level_data_min:
            extracted_metadata[f'perturbation_T_min_L{li:02d}_sol_{rel_sol_int:03d}'] = data['T_diff'].min()
            extracted_metadata[f'perturbation_Dust_min_L{li:02d}_sol_{rel_sol_int:03d}'] = data['Dust_diff'].min()
            
        for rel_sol_int, data in level_data_max:
            extracted_metadata[f'perturbation_T_max_L{li:02d}_sol_{rel_sol_int:03d}'] = data['T_diff'].max()
            extracted_metadata[f'perturbation_Dust_max_L{li:02d}_sol_{rel_sol_int:03d}'] = data['Dust_diff'].max()
    
    '''
    # TODO: If using this, need to decide if calcs should be made on original disagregated data or the min/max/mean after aggregation
    perturbation_data_L0_20 = perturbation_data.loc[perturbation_data['level']<=20, :].groupby('rel_sol')
    for g1_rel_sol, g1 in perturbation_data_L0_20:
        extracted_metadata[f'perturbation_T_min_L00_20_sol_{g1_rel_sol:03d}'] = g1['T_diff'].min()
        extracted_metadata[f'perturbation_T_max_L00_20_sol_{g1_rel_sol:03d}'] = g1['T_diff'].max()
        extracted_metadata[f'perturbation_T_mean_L00_20_sol_{g1_rel_sol:03d}'] = g1['T_diff'].mean()
        extracted_metadata[f'perturbation_Dust_min_L00_20_sol_{g1_rel_sol:03d}'] = g1['Dust_diff'].min()
        extracted_metadata[f'perturbation_Dust_max_L00_20_sol_{g1_rel_sol:03d}'] = g1['Dust_diff'].max()
        extracted_metadata[f'perturbation_Dust_mean_L00_20_sol_{g1_rel_sol:03d}'] = g1['Dust_diff'].mean()

    perturbation_data_L20_40 = perturbation_data.loc[(perturbation_data['level']>20) & (perturbation_data['level']<=40), :].groupby('rel_sol')
    for g2_rel_sol, g2 in perturbation_data_L20_40:
        extracted_metadata[f'perturbation_T_min_L20_40_sol_{g2_rel_sol:03d}'] = g2['T_diff'].min()
        extracted_metadata[f'perturbation_T_max_L20_40_sol_{g2_rel_sol:03d}'] = g2['T_diff'].max()
        extracted_metadata[f'perturbation_T_mean_L20_40_sol_{g2_rel_sol:03d}'] = g2['T_diff'].mean()
        extracted_metadata[f'perturbation_Dust_min_L20_40_sol_{g2_rel_sol:03d}'] = g2['Dust_diff'].min()
        extracted_metadata[f'perturbation_Dust_max_L20_40_sol_{g2_rel_sol:03d}'] = g2['Dust_diff'].max()
        extracted_metadata[f'perturbation_Dust_mean_L20_40_sol_{g2_rel_sol:03d}'] = g2['Dust_diff'].mean()
    '''
    ###################################
    # Update grid properties for all x and y axes
    fig.update_xaxes( showgrid=True, gridwidth=0.5,
        gridcolor='rgba(128, 128, 128, 0.3)')  # Gray with 30% opacity

    fig.update_yaxes(showgrid=True, gridwidth=0.5, 
                     gridcolor='rgba(128, 128, 128, 0.3)')  # Gray with 30% opacity

    fig.update_layout(height=2400, width=3000, 
                      title_text=plot_title,
                      margin=dict(t=250))  # Adjust the top margin to avoid overlap
    
    # Save to disk
    if png_save_fpath:
        try:
            Path(png_save_fpath).parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(png_save_fpath)
        except Exception as e:
            logging.error(f"Error saving figure to {png_save_fpath}: {e}")
    if html_save_fpath:
        Path(html_save_fpath).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(html_save_fpath)

    # Deleting larger variables to ensure they're released from memory
    storm_profiles_df = None
    control_profiles_df = None
    control_profiles_df_smoothed = None
    fig = None

    return extracted_metadata


def get_column_patterns(time_range, levels_to_extract):
    """Get column patterns for the given time range.
    
    Parameters
    ----------
    time_range : tuple
        (min, max) range for time points
        
    Returns
    -------
    dict
        Dictionary of column patterns
    """
    # Build the time_series dictionary dynamically
    time_series = {
        'dust_opacity_mean_sol_{:03d}': time_range,
        'perturbation_dust_opacity_mean_sol_{:03d}': time_range,
        'perturbation_T_surf_mean_sol_{:03d}': time_range,
        # Original data for level ranges 
        'T_min_L00_20_sol_{:03d}': time_range,
        'T_max_L00_20_sol_{:03d}': time_range,
        'T_mean_L00_20_sol_{:03d}': time_range,
        'T_min_L20_40_sol_{:03d}': time_range,
        'T_max_L20_40_sol_{:03d}': time_range,
        'T_mean_L20_40_sol_{:03d}': time_range,
        'Dust_min_L00_20_sol_{:03d}': time_range,
        'Dust_max_L00_20_sol_{:03d}': time_range,
        'Dust_mean_L00_20_sol_{:03d}': time_range,
        'Dust_min_L20_40_sol_{:03d}': time_range,
        'Dust_max_L20_40_sol_{:03d}': time_range,
        'Dust_mean_L20_40_sol_{:03d}': time_range,
        # Perturbation data for level ranges
        'perturbation_T_max_L00_20_sol_{:03d}': time_range,
        'perturbation_T_max_L20_40_sol_{:03d}': time_range,
        'perturbation_T_min_L00_20_sol_{:03d}': time_range,
        'perturbation_T_min_L20_40_sol_{:03d}': time_range,
        'perturbation_T_mean_L00_20_sol_{:03d}': time_range,
        'perturbation_T_mean_L20_40_sol_{:03d}': time_range,
        'pval_min_subL40_t_sol_window_01_sol_{:03d}': time_range,
        'pval_mean_subL40_t_sol_window_01_sol_{:03d}': time_range,
        'pval_median_subL40_t_sol_window_01_sol_{:03d}': time_range,
        # Perturbation data for level ranges
        'perturbation_Dust_max_L00_20_sol_{:03d}': time_range,
        'perturbation_Dust_max_L20_40_sol_{:03d}': time_range,
        'perturbation_Dust_min_L00_20_sol_{:03d}': time_range,
        'perturbation_Dust_min_L20_40_sol_{:03d}': time_range,
        'perturbation_Dust_mean_L00_20_sol_{:03d}': time_range,
        'perturbation_Dust_mean_L20_40_sol_{:03d}': time_range,
        'pval_min_subL40_dust_sol_window_01_sol_{:03d}': time_range,
        'pval_mean_subL40_dust_sol_window_01_sol_{:03d}': time_range,
        'pval_median_subL40_dust_sol_window_01_sol_{:03d}': time_range,
    }
    
    # Construct more column names using pattern and levels (likely from config)
    for level in levels_to_extract:
        # Original data at specific levels
        time_series[f'T_min_L{level:02d}_sol_{{:03d}}'] = time_range
        time_series[f'T_max_L{level:02d}_sol_{{:03d}}'] = time_range
        time_series[f'T_mean_L{level:02d}_sol_{{:03d}}'] = time_range
        time_series[f'Dust_min_L{level:02d}_sol_{{:03d}}'] = time_range
        time_series[f'Dust_max_L{level:02d}_sol_{{:03d}}'] = time_range
        time_series[f'Dust_mean_L{level:02d}_sol_{{:03d}}'] = time_range
        
        # Perturbation data for specific levels
        time_series[f'perturbation_T_max_L{level:02d}_sol_{{:03d}}'] = time_range
        time_series[f'perturbation_T_min_L{level:02d}_sol_{{:03d}}'] = time_range
        time_series[f'perturbation_T_mean_L{level:02d}_sol_{{:03d}}'] = time_range
        time_series[f'perturbation_Dust_max_L{level:02d}_sol_{{:03d}}'] = time_range
        time_series[f'perturbation_Dust_min_L{level:02d}_sol_{{:03d}}'] = time_range
        time_series[f'perturbation_Dust_mean_L{level:02d}_sol_{{:03d}}'] = time_range
    
    return {
        'basic_stats': [
            'storm_db_id', 'area', 'Ls', 'lat', 'storm_len_in_sols', 'lon', 
            'conflev', 'mars_year', 'n_control_profiles', 
            'n_storm_profiles_sol_of', 'n_storm_profiles_total'
        ],
        'time_series': time_series
    }


def filter_and_sort_data(df, ls_range, lat_range, min_conflev, storm_len, 
                         min_profiles_sol_of, area, time_range):
    """Filter and sort the dataframe based on input parameters.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing storm data
    ls_range : tuple
        (min, max) range for Ls values
    lat_range : tuple
        (min, max) range for latitude values
    min_conflev : float
        Minimum confidence level
    storm_len : tuple
        (min, max) range for storm length in sols
    min_profiles_sol_of : int
        Minimum number of storm profiles
    area : tuple
        (min, max) range for area in km^2
    time_range : tuple
        (min, max) range for time points
        
    Returns
    -------
    pd.DataFrame
        Filtered and sorted dataframe
    """
    # Create boolean mask for filtering
    mask = (
        (df['Ls'].between(ls_range[0], ls_range[1])) &  # Between is inclusive of edges by default
        (df['lat'].between(lat_range[0], lat_range[1])) &
        (df['conflev'] >= min_conflev) &
        (df['storm_len_in_sols'].between(storm_len[0], storm_len[1])) &
        (df['n_storm_profiles_sol_of'] >= min_profiles_sol_of) &
        (df['area'].between(area[0], area[1]))
    )
    
    # Apply filter using .loc
    filtered_df = df.loc[mask].copy()
    
    # Drop non-numeric columns
    filtered_df = filtered_df.drop(columns=['mdgm', 'storm_id', 'seq_id', 'exit_status'])
    
    # Get column patterns for the time range
    patterns = get_column_patterns(time_range, config.per_level_analysis['level_to_extract'])
    
    # Generate sorted columns list
    sorted_columns = []
    sorted_columns.extend(patterns['basic_stats'])
    
    for pattern, (start, end) in patterns['time_series'].items():
        sorted_columns.extend(pattern.format(i) for i in range(start, end + 1))
        
    return filtered_df.loc[:, sorted_columns]


def format_parameter_text(params):
    """Format parameter information for plot annotations.
    
    Parameters
    ----------
    params : dict
        Dictionary containing plot parameters
        
    Returns
    -------
    str
        Formatted parameter text
    """
    return (f"Parameters: Ls={params['ls_range']}<br>Lat={params['lat_range']}<br>"
            f"Min Conf={params['min_conflev']}<br>"
            f"Storm Len={params['storm_len']}<br>"
            f"Min Profiles Sol of={params['min_profiles_sol_of']}<br>"
            f"Area={params['area']}<br>"
            f"Time Range={params['time_range']}<br>N={params['n_storms']} storms")


def visualize_controls(control_profiles_df, pressure_levels=(40, 30, 20, 10), png_save_fpath=None, 
                       variability_alpha=0.1,
                       atmospheric_temp_lims=(160, 220),
                       atmospheric_dust_lims=(1e-6, 1e-2),
                       surface_temp_lims=(160, 270),
                       dust_column_lims=(1e-6, 1)):
    """Create visualization of control profiles across multiple Mars years.

    Parameters
    ----------
    control_profiles_df : pd.DataFrame
        DataFrame containing control profile data with columns:
        'sol_int', 'mars_year', 'T', 'Dust', 'level', 'T_surf', 'Dust_column'
    pressure_levels : tuple, optional
        Pressure levels to plot, by default (40, 30, 20, 10)
    png_save_fpath : str, optional
        Path to save the PNG file, by default None
    variability_alpha : float, optional
        Alpha value for fill_between shading
    pressure_temp_lims : tuple, optional
        (min, max) temperature limits for pressure level plots
    pressure_dust_lims : tuple, optional
        (min, max) dust limits for pressure level plots
    surface_temp_lims : tuple, optional
        (min, max) temperature limits for surface plot
    dust_column_lims : tuple, optional
        (min, max) dust limits for column plot
    """
    # Create figure with extra width to accommodate metadata text
    n_rows = len(pressure_levels) + 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    
    # Set up color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, len(control_profiles_df['mars_year'].unique())))
    sorted_years = sorted(control_profiles_df['mars_year'].unique())
    
    # Prepare metadata text
    total_profiles = len(control_profiles_df['Profile_identifier'].unique())
    metadata_text = f'Total Profiles: {total_profiles:,}\n\n'
    
    for year in sorted_years:
        year_data = control_profiles_df[control_profiles_df['mars_year'] == year]
        n_profiles = len(year_data['Profile_identifier'].unique())
        ls_min = year_data['L_s'].min()
        ls_max = year_data['L_s'].max()
        metadata_text += f'MY {year}: {n_profiles} profiles\n'
        metadata_text += f'Ls {ls_min:.1f}° to {ls_max:.1f}°\n\n'
    
    # Adjust text position to top right
    fig.text(0.98, 0.9, metadata_text, fontsize=8, va='top', ha='right', family='monospace')

    ###################################
    # Plot T/Dust at each pressure level
    for row, level in enumerate(pressure_levels):
        # Temperature plot
        for year, color in zip(sorted_years, colors):
            year_data = control_profiles_df[(control_profiles_df['mars_year'] == year) & 
                                            (control_profiles_df['level'] == level)].groupby('sol_int')
            
            mean_temp = year_data['T'].mean()
            std_temp = year_data['T'].std()
            
            axes[row, 0].plot(mean_temp.index, mean_temp.values, label=f'MY {year}', color=color)
            std_vals = np.nan_to_num(std_temp.values, nan=0.0)  # Guard against NaNs in std for fill_between
            lower = mean_temp.values - std_vals
            upper = mean_temp.values + std_vals
            axes[row, 0].fill_between(mean_temp.index, lower, upper, alpha=variability_alpha, color=color)
        
        # Dust plot
        for year, color in zip(sorted_years, colors):
            year_data = control_profiles_df[(control_profiles_df['mars_year'] == year) & 
                                            (control_profiles_df['level'] == level)].groupby('sol_int')
            
            mean_dust = year_data['Dust'].mean()
            std_dust = year_data['Dust'].std()
            
            axes[row, 1].plot(mean_dust.index, mean_dust.values, label=f'MY {year}', color=color)
            std_vals = np.nan_to_num(std_dust.values, nan=0.0)
            lower = mean_dust.values - std_vals
            upper = mean_dust.values + std_vals
            axes[row, 1].fill_between(mean_dust.index, lower, upper, alpha=variability_alpha, color=color)
        
        # Set titles and labels for pressure level plots
        axes[row, 0].set_title(f'Temperature at Level {level}')
        axes[row, 1].set_title(f'Dust at Level {level}')
        axes[row, 1].set_yscale('log')
    
    ###################################
    # Plot surface measurements in bottom row
    for year, color in zip(sorted_years, colors):
        year_data = control_profiles_df[control_profiles_df['mars_year'] == year].groupby('sol_int')
        
        # Surface temperature
        mean_surf_temp = year_data['T_surf'].mean()
        std_surf_temp = year_data['T_surf'].std()
        
        axes[-1, 0].plot(mean_surf_temp.index, 
                        mean_surf_temp.values, 
                        label=f'MY {year}', color=color)
        std_vals = np.nan_to_num(std_surf_temp.values, nan=0.0)
        lower = mean_surf_temp.values - std_vals
        upper = mean_surf_temp.values + std_vals
        axes[-1, 0].fill_between(mean_surf_temp.index, lower, upper, alpha=variability_alpha, color=color)
        
        # Dust column
        mean_dust_col = year_data['Dust_column'].mean()
        std_dust_col = year_data['Dust_column'].std()
        
        axes[-1, 1].plot(mean_dust_col.index, mean_dust_col.values,
                        label=f'MY {year}', color=color)
        std_vals = np.nan_to_num(std_dust_col.values, nan=0.0)
        lower = mean_dust_col.values - std_vals
        upper = mean_dust_col.values + std_vals
        lower = np.clip(lower, dust_column_lims[0], np.inf)
        upper = np.clip(upper, dust_column_lims[0], np.inf)
        axes[-1, 1].fill_between(mean_dust_col.index, lower, upper, alpha=variability_alpha, color=color)
    
    # Set titles and labels for surface measurement plots
    axes[-1, 0].set_title('Surface Temperature')
    axes[-1, 1].set_title('Dust Column')
    axes[-1, 1].set_yscale('log')
    
    # Add common labels and legend
    for ax in axes[:, 0]:
        ax.set_ylabel('Temperature (K)')
    for ax in axes[:, 1]:
        ax.set_ylabel('Dust')
    for ax in axes[-1]:
        ax.set_xlabel('Sol')
    
    # Add legend to top-right subplot
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set y-limits for pressure level plots
    for row in range(len(pressure_levels)):
        axes[row, 0].set_ylim(atmospheric_temp_lims)
        axes[row, 1].set_ylim(atmospheric_dust_lims)
        axes[row, 1].set_yscale('log')

    # Set y-limits for surface plots
    axes[-1, 0].set_ylim(surface_temp_lims)
    axes[-1, 1].set_ylim(dust_column_lims)
    axes[-1, 1].set_yscale('log')

    # Adjust layout with extra right margin for metadata text
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    if png_save_fpath:
        Path(png_save_fpath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(png_save_fpath, bbox_inches='tight', dpi=300)
    
    plt.close()


def make_storm_control_profile_comparison_plot(control_dfs: List, storm_dfs: List, storm_sol: float, storm_time_bounds, yaxis="Pres", save_path=None):
    """
    Plot control and storm profiles (and percentiles) before/during/after storm
    rows: before/during/after
    columns: Temperature profiles, Dust profiles, Dust columns, T_surfs
    """
    # Set quantiles and styles for bold lines in per-level profile plots
    quantiles = [0.1, 0.5, 0.9]
    styles = ['--', '-', '--']

    n_time_steps = len(storm_dfs)  # Number of time chunks (rows)
    fig, ax = plt.subplots(n_time_steps, 4, figsize=(22, 4*n_time_steps))
    # Make plots for each row
    for i in range(n_time_steps):
        if control_dfs[i] is None:
            continue

        ###########################################
        # Line plots showing temp and dust profiles

        # First plot all individual control profiles (thin, light)
        for cp, cp_group in control_dfs[i].groupby("Profile_identifier"):
            cp_data = cp_group.sort_values(yaxis)
            ax[i, 0].plot(cp_data["T"], cp_data[yaxis], c="lightblue", lw=0.1)
            ax[i, 1].plot(cp_data["Dust"], cp_data[yaxis], c="lightblue", lw=0.1)

        # Calculate some quantiles and plot (bolder)
        for quantile, style in zip(quantiles, styles):
            quant_group = control_dfs[i].groupby("level")[["T", "Dust", yaxis]].quantile(quantile)
            ax[i, 0].plot(quant_group["T"], quant_group[yaxis], c="blue", lw=2, ls=style)
            ax[i, 1].plot(quant_group["Dust"], quant_group[yaxis], c="blue", lw=2, ls=style)

        # Plot storm profiles
        if storm_dfs[i] is not None:
            for sp, sp_group in storm_dfs[i].groupby("Profile_identifier"):
                # Specify individual profile lw + colors if sol of storm or not
                lw = 0.1
                c="r"
                # If sol of storm, plot bolder
                if int(sp_group["sol"].unique().squeeze())==int(storm_sol):
                    lw=1
                    c="k"
                # Plot individual profiles
                ax[i, 0].plot(sp_group["T"], sp_group[yaxis], c=c, lw=lw, zorder=10000)
                ax[i, 1].plot(sp_group["Dust"], sp_group[yaxis], c=c, lw=lw, zorder=10000)

            for quantile, style in zip(quantiles, styles):
                storm_quant = storm_dfs[i].groupby("level")[["T", "Dust", yaxis]].quantile(quantile)
                ax[i, 0].plot(storm_quant["T"], storm_quant[yaxis], c="r", lw=2, ls=style, zorder=1000)
                ax[i, 1].plot(storm_quant["Dust"], storm_quant[yaxis], c="r", lw=2, ls=style, zorder=1000)

        ######################################
        # Histograms of Dust column and Tsurf

        # Control dust col/surf temp histograms
        # Can use 'first' for dust column and surf temp. since values will be stored as identical numbers for all levels/altitudes in each profile
        dust_data_arr = control_dfs[i].groupby("Profile_identifier")["Dust_column"].first().to_numpy()
        dust_data = np.where(dust_data_arr > 0, np.log10(dust_data_arr), np.nan)
        temp_data = control_dfs[i].groupby("Profile_identifier")["T_surf"].first().dropna()

        if not dust_data.size == 0 and not dust_data.isna().all():
            dust_weights = np.ones(len(dust_data)) / dust_data.count()  # weights to plot controls and storms together. Only counts non-nan vals
            ax[i, 2].hist(dust_data, weights=dust_weights, color="lightblue", density=False, bins=np.arange(-2.5, 1, 0.1))
        if not temp_data.size == 0 and not temp_data.isna().all():
            temp_weights = np.ones(len(temp_data)) / temp_data.count()
            ax[i, 3].hist(temp_data, weights=temp_weights, color="lightblue", density=False)

        # Storm dust col/surf temp histograms
        if storm_dfs[i] is not None:
            dust_data_arr = storm_dfs[i].groupby("Profile_identifier")["Dust_column"].first().to_numpy()
            dust_data = np.where(dust_data_arr > 0, np.log10(dust_data_arr), np.nan)
            dust_nan_count = dust_data.isna().sum()
            if dust_nan_count:
                ax[i, 2].annotate(f'{dust_nan_count} NaN(s)\n{dust_nan_count/len(dust_data):0.1%}', 
                                  xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10)

            if not dust_data.empty and not dust_data.isna().all():
                dust_weights = np.ones(len(dust_data)) / dust_data.count()
                ax[i, 2].hist(dust_data, weights=dust_weights, color="r", alpha=0.5, density=False, bins=np.arange(-2.5, 1, 0.1))

            temp_data = storm_dfs[i].groupby("Profile_identifier")["T_surf"].first().dropna()
            temp_nan_count = temp_data.isna().sum()
            if temp_nan_count:
                ax[i, 3].annotate(f'{temp_nan_count} NaN(s)\n{temp_nan_count/len(temp_data):0.1%}', 
                                  xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10)

            if not temp_data.empty and not temp_data.isna().all():
                temp_weights = np.ones(len(temp_data)) / temp_data.count()
                ax[i, 3].hist(temp_data, weights=temp_weights, color="r", alpha=0.5, density=False)
            
        ######################################
        # General figure parameters

        # Set axes limits        
        ax[i, 0].set_xlim(130, 230)
        ax[i, 1].set_xlim(1.e-6, 1.e-1)
        ax[i, 1].set_xscale("log")
        ax[i, 2].set_xlim(-2.5, -0.5)

        # Y-axis depends on variable
        for a in ax[i, 0:2]:
            if yaxis=="Alt":
                a.set_ylim(0, 60)
            else:
                a.set_ylim(2.e+3, 1.e-2)
                a.set_yscale("log")

        if yaxis=="Alt":
            ax[i, 0].set_ylabel(f"Altitude [km] ({storm_time_bounds[i][0]} sols before to {storm_time_bounds[i][1]} sols after)")
        else:
            ax[i, 0].set_ylabel("Pressure [Pa]")

    #fig.suptitle(f"Storm {STORM_ID}")
    ax[0, 0].set_title("Temperature")
    ax[0, 1].set_title("Dust Profile")
    ax[0, 2].set_title("Dust Opacity")
    ax[0, 3].set_title("Surface Temp")
    ax[-1, 0].set_xlabel("Temperature [K]")
    ax[-1, 1].set_xlabel(r"Dust [km$^{-1}$]")
    ax[-1, 2].set_xlabel(r"Dust Opacity [log]" )
    ax[-1, 3].set_xlabel(r"Surface Temp [K]" )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    plt.close(fig)

    return fig, ax


def normalize_columns(df, col_pattern, sol_range, suffix='_norm', method='min_max',
                      baseline_window=None, baseline_func='median',
                      pct_baseline_window=None, pct_baseline_func='mean'):
    """
    Normalize time series columns across the sol range for each row. Useful in time series plots
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the time series columns
    col_pattern : str
        Pattern for columns to normalize, with {} for sol number formatting
    sol_range : range or list
        Range of sol values to process
    suffix : str, default='_norm'
        Suffix to add to normalized column names
    method: str
        Whether to normalize to a min-max range, by z-score, percent, or absnorm
    baseline_window: tuple or list, optional
        Inclusive window (start, end) of sols to compute baseline for centering (applies to all methods except 'percent').
        Window is inclusive of both endpoints. If None, no baseline centering is applied.
    baseline_func: str, optional
        Baseline aggregation function for generic centering: 'mean' or 'median'. Default is 'median'.
    pct_baseline_window: tuple or list, optional
        When method=='percent', inclusive window (start, end) of sols to compute baseline. 
        Window is inclusive of both endpoints.
    pct_baseline_func: str, optional
        When method=='percent', baseline aggregation function: 'mean' or 'median'
        
    Returns
    -------
    pandas.DataFrame
        New DataFrame with original data plus normalized columns added
    """
    df = df.copy()
    
    # Create list of all columns matching the pattern
    cols = [col_pattern.format(i) for i in sol_range]

    # Skip missing columns
    if not all(col in df.columns for col in cols):
        logging.warning(f"Some columns matching '{col_pattern}' not found in dataframe")
        cols = [col for col in cols if col in df.columns]
        if not cols:
            logging.warning(f"No columns matching '{col_pattern}' found in dataframe")
            return df

    new_cols = [f"{col}{suffix}" for col in cols]
    n_rows = len(df)
    n_cols = len(cols)

    results = np.full((n_rows, n_cols), np.nan)

    baseline_col_indices = None
    if baseline_window is not None and method != 'percent':
        baseline_col_indices = [i for i, sol in enumerate(sol_range)
                                if baseline_window[0] <= sol <= baseline_window[1]
                                and col_pattern.format(sol) in df.columns]
        if len(baseline_col_indices) == 0:
            logging.warning(f"No baseline columns in desired baseline window {baseline_window}")

    pct_baseline_col_indices = None
    if method == 'percent':
        if not pct_baseline_window:
            raise ValueError("Percent normalization window is required for normalize_columns method `percent`")
        pct_baseline_col_indices = [i for i, sol in enumerate(sol_range)
                                    if pct_baseline_window[0] <= sol <= pct_baseline_window[1]
                                    and col_pattern.format(sol) in df.columns]
        if len(pct_baseline_col_indices) == 0:
            logging.warning(f"No baseline columns in percent window {pct_baseline_window}")

    data_matrix = df[cols].to_numpy(dtype=float)

    for row_idx in range(n_rows):
        values = data_matrix[row_idx].copy()
        storm_id = df.index[row_idx]

        if np.all(np.isnan(values)):
            logging.warning(f"Skipping storm {storm_id} because all values are NaN")
            continue

        if baseline_col_indices is not None and method != 'percent':
            if len(baseline_col_indices) == 0:
                logging.warning(f"Skipping storm {storm_id}: no baseline columns in desired baseline window {baseline_window}")
                continue

            baseline_values = values[baseline_col_indices]
            if baseline_func == 'median':
                baseline = np.nanmedian(baseline_values)
            elif baseline_func == 'mean':
                baseline = np.nanmean(baseline_values)
            else:
                raise ValueError(f'Baseline function not correct. Received "{baseline_func}"')

            if np.isnan(baseline):
                logging.warning(f"Skipping storm {storm_id}: baseline is NaN")
                continue

            values = values - baseline

        ##########################################################
        if method == 'min_max':
            # Calculate min and max, ignoring NaN values
            min_val = np.nanmin(values)
            max_val = np.nanmax(values)

            # Skip if min equals max (no variation to normalize)
            if min_val == max_val:
                normalized = np.full_like(values, np.nan)
            else:
                # Calculate the midpoint of the data range
                mid_val = (min_val + max_val) / 2
                # Scale values to -1 to 1 range
                normalized = 2 * (values - mid_val) / (max_val - min_val)

        ##########################################################
        elif method == 'zscore':
            std_val = np.nanstd(values)
            if std_val == 0:
                logging.warning(f"Skipping storm {storm_id} because std is 0")
                continue
            normalized = values / std_val

        elif method == 'percent':
            if len(pct_baseline_col_indices) == 0:
                logging.warning(f"Skipping storm {storm_id}: no baseline columns in window {pct_baseline_window}")
                continue

            baseline_values = values[pct_baseline_col_indices]
            if pct_baseline_func == 'median':
                baseline = np.nanmedian(baseline_values)
            else:
                baseline = np.nanmean(baseline_values)

            if np.isnan(baseline) or baseline == 0:
                logging.warning(f"Skipping storm {storm_id}: invalid baseline {baseline}")
                continue

            normalized = 100.0 * (values - baseline) / baseline

        ##########################################################
        elif method == 'absnorm':
            # Normalize by max absolute value (values may already be centered via baseline_window)
            max_abs = np.nanmax(np.abs(values))
            
            if max_abs == 0 or np.isnan(max_abs):
                logging.warning(f"Skipping storm {storm_id}: Center and absnorm normalization skipped for {col_pattern} as max_abs is {max_abs}")
                normalized = np.full_like(values, np.nan)
            else:
                normalized = values / max_abs
        else:
            raise ValueError(f"Invalid normalization method: {method}")

        results[row_idx] = normalized

    result_df = pd.DataFrame(results, index=df.index, columns=new_cols)
    df.loc[:, new_cols] = result_df

    return df