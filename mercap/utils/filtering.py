"""
Utils for filtering (often used when generating plots)
"""


def filter_storms(df, ls_min, ls_max, lat_min, lat_max, 
                  conflev_min, storm_len_max, n_profiles_min, 
                  area_max, pval_max, pval_window_start, pval_window_end,
                  storm_lifecycle_filter):
    """
    Apply filters to storm dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing storm data
    ls_min : float
        Minimum solar longitude
    ls_max : float
        Maximum solar longitude
    lat_min : float
        Minimum latitude
    lat_max : float
        Maximum latitude
    conflev_min : int
        Minimum confidence level
    storm_len_max : float
        Maximum storm length in sols
    n_profiles_min : int
        Minimum number of profiles
    area_max : float
        Maximum area
    pval_max : float or None
        Maximum p-value for dust opacity, if None no p-value filtering is applied
    pval_window_start : int
        Start sol for time series filtering
    pval_window_end : int
        End sol for time series filtering
    storm_lifecycle_filter : bool
        If True, filter storms to those that have an onset and end, and are not a sequence
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """

    # Basic metadata filtering
    filtered_df = df[(df['Ls'] >= ls_min) & (df['Ls'] <= ls_max) &        # Solar longitude filter
                     (df['lat'] >= lat_min) & (df['lat'] <= lat_max) &       # Latitude filter
                     (df['conflev'] >= conflev_min) &                       # Confidence level (likely want >= 2 or 3)
                     (df['storm_len_in_sols'] <= storm_len_max) &             # Total storm length. 1 or 2 sols for "local" storms
                     (df['n_storm_profiles_sol_of'] >= n_profiles_min) &       # Number of profiles on the sol the member was observed
                     (df['area'] <= area_max)]                         # Area limit (<=1.6e6 km^2 for "local" storms)

    # Timeseries metadata filtering - only if pval_max is not None
    if pval_max is not None:
        pval_window_of_interest = range(pval_window_start, pval_window_end)
        min_dust_opacity_pval = filtered_df[[f'dust_opacity_pval_{i:03d}' for i in pval_window_of_interest]]
        filtered_df = filtered_df[(min_dust_opacity_pval < pval_max).sum(axis='columns') >= 1]

    # Filter storms to those that have an onset and end, and are not a sequence
    if storm_lifecycle_filter:
        storms_with_onset = df.loc[df["storm_onset"]==1, "storm_id"].to_list()
        storms_with_end = df.loc[df["storm_end"]==1, "storm_id"].to_list()
        filtered_df = filtered_df[(filtered_df["storm_id"].isin(storms_with_onset) & 
                                   filtered_df["storm_id"].isin(storms_with_end) & 
                                   (filtered_df["seq_id"] == 'None'))]

    return filtered_df