# Configuration options

PI = 3.14159265359
N_JOBS = 64
SQL_ENGINE_STRING_MCS13 = 'postgresql://<path_to_geospatial_db>'

# Geoid parameters (in meters) for geodesic calculations
MARS_GEOID_A = 3396190
MARS_GEOID_B = 3376200

MDAD_CONFLEV_CONVERSION = {0: 0, 25: 1, 50: 2, 75: 3, 100: 4}

TIME_SUBPLOT_SOL_RANGE = (-10, 11)  # Used in profile plots to set the min/max sol
INT_SOLS_PER_MY = 669
SOL_WINDOW_BOUNDS = (-20, 21)  # Sols before/after storm. Add 1 to the right side to include the storm sol
IMPROVE_MARCI_TIMING_PRECISION = True  # False uses 'ls' from original mdad/mdssd file. True uses the more precise, swath-specific Ls/time values from MARCI cumindex.tab
MARCI_DT_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'

tolerances = {'LTST_DAY_START': 9/24,
              'LTST_NIGHT_START': 21/24}  # Local True Solar Times allowed (must be expressed on [0, 1])
stats = {'min_points': 2,  # Minimum number of data points required before we'll compute statistics
         'controls_window_size': 11,  # Total length of window for controls in sols for stats calculations. Must be odd
         'storm_window_size': 1,  # Total length of window for storm in sols for stats calculations. Must be odd or can be None for if using all data
         'max_level': 40,
         'test_statistic': 'cvm',  # 'cvm' or 'ks'
         'min_pvalue': 1e-5,  # Minimum p-value for color scale/plotting range
         'max_pvalue': 0.1}  # Maximum p-value for color scale/plotting range
    
controls = {'Ls_window_distribution_plots': 12.5,  # Used in plotting to find control profiles
            'collage_controls_smoothing_window_size': 11,  # Window size for smoothing controls with median filter. Want odd value so we include sol 0 and equal number of sols on either side. This will also be used to identify overlapping storms when searching in time in `find_intersecting_storms`
            'sol_exclusion_window_size': 4,  # +/- this many sols (inclusive) will be excluded for storms that overlapped in time and space. Set this in conjunction with `collage_controls_smoothing_window_size`
            'control_smoothing_method': 'one_step_aggregate_then_smooth'}  # Method for smoothing control profiles: 'two_step_median' (equal weight per Mars year) or 'one_step_aggregate_then_smooth' (original method, mean then rolling median)

good_obs_quals = (0, 1, 7, 10, 11, 17)  # Must be in tuple format for DB query
min_dust_permitted = 1e-6  # Lowest level of dust permitted when loading atmospheric data. Lower values will be set to NaN. Set this to None to avoid filtering. Used in `load_ddr2_from_profile_matches`
DROP_INVALID_DDR1 = True  # Whether or not to drop profiles in load_ddr2_from_profile_matches if invalid (when '1' column != 0)

freq_analysis = {'min_freq': PI/10,  # Minimum frequency to use in Lomb-Scargle periodograms
                 'max_freq': 4/5 * PI,  # Maximum frequency to use in Lomb-Scargle periodograms
                 'n_freqs': 50,  # Number of frequencies to use in Lomb-Scargle periodograms between min and max
                 'min_data_pts': 10}  # Minimum number of data points required to perform Lomb-Scargle periodogram

per_level_analysis = {'level_to_extract': [8, 10, 12, 14, 16, 18, 20, 22, 24, 30, 40, 50, 60]}

dust_database_time_spans = {'mdssd': ('2007-10-09', '2019-01-21'),
                            'mdad': ('2006-11-08', '2017-05-05')}