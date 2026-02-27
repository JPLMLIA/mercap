import click
import logging

import pandas as pd
import sqlalchemy
from joblib import Parallel, delayed
from shapely.geometry import Point
from geoalchemy2 import Geometry
from tqdm import tqdm
import mars_time
from sqlalchemy.exc import SQLAlchemyError

from mercap.config import N_JOBS


logging.basicConfig(level=logging.INFO,
                    datefmt='%H:%M:%S',
                    format='%(asctime)s | %(lineno)d | %(levelname)-5s | %(module)-15s | %(message)s')

@click.command()
@click.option('--mcs_fpath', type=click.Path(exists=True), required=True, help='Filepath of MCS data dump.')
@click.option('--db_url', type=str, required=True, help='Database engine path to save PostGRES+PostGIS.')
@click.option('--table_name', type=str, required=True, help='Database table name.')
@click.option('--mcs_ddr1_latlon', type=click.Choice(["Profile", "Surf"]), default="Surf", help='MCS DDR1 lat/lon columns to use for profile locations.')
@click.option('--write_conflict_behavior', type=click.Choice(['fail', 'replace', 'append']), default='fail', help='If database table already exists, specify what to do.')
@click.option('--smoke_test', is_flag=True, help='Enable smoke test to debug by processing only the first 10000 profiles.')
def cli(mcs_fpath, db_url, table_name, mcs_ddr1_latlon, write_conflict_behavior, smoke_test):

    logging.info('Reading MCS file: %s', mcs_fpath)
    nrows = 1000 if smoke_test else None
    mcs_df = pd.read_csv(mcs_fpath, sep='\t', engine='python', parse_dates=['dt'], 
                         index_col='dt', nrows=nrows)

    # Set the LTST column to lower case as SQL has issues otherwise
    mcs_df.rename(columns={'LTST': 'ltst', 
                           'T_surf': 't_surf',
                           'Obs_qual': 'obs_qual', 
                           'P_qual': 'p_qual', 
                           'T_qual': 't_qual',
                           'Dust_qual': 'dust_qual'}, inplace=True)

    logging.info('Adding Mars Datetime object')
    mars_dts = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(mars_time.datetime_to_marstime)(temp_dt) 
                                                  for temp_dt in mcs_df.index.to_pydatetime())
    mcs_df['mars_year'] = [temp_dt.year for temp_dt in mars_dts]
    mcs_df['sol'] = [temp_dt.sol for temp_dt in mars_dts]
 
    logging.info('Converting lon/lat to Geometry type')
    lon_col, lat_col = f'{mcs_ddr1_latlon}_lon', f'{mcs_ddr1_latlon}_lat'
    mcs_df['profile_loc'] = [Point([lon, lat]).wkt for lon, lat in 
                             tqdm(zip(mcs_df[lon_col], mcs_df[lat_col]), desc='Converting lon/lat to Geometry type')]

    logging.info('Exporting MCS info+geometry to database at: %s', db_url)
    column_types = {'profile_loc': Geometry('POINT')}
    mcs_df.to_sql(table_name, db_url, index=True, if_exists=write_conflict_behavior, dtype=column_types)
    logging.info('\nInfo on dataframe inserted into DB:')
    logging.info(mcs_df.info())

    logging.info('Attempting to add indices...')

    # Define the indices creation statements
    drop_index_1 = sqlalchemy.text(f'DROP INDEX IF EXISTS profile_loc_index;')
    create_index_1 = sqlalchemy.text(f'CREATE INDEX IF NOT EXISTS profile_loc_index ON {table_name} USING GIST (profile_loc);')

    drop_index_2 = sqlalchemy.text(f'DROP INDEX IF EXISTS ls_index;')
    create_index_2 = sqlalchemy.text(f'CREATE INDEX IF NOT EXISTS ls_index ON {table_name} (l_s);')

    drop_index_3 = sqlalchemy.text(f'DROP INDEX IF EXISTS mars_year_index;')
    create_index_3 = sqlalchemy.text(f'CREATE INDEX IF NOT EXISTS mars_year_index ON {table_name} (mars_year);')

    drop_index_4 = sqlalchemy.text(f'DROP INDEX IF EXISTS sol_index;')
    create_index_4 = sqlalchemy.text(f'CREATE INDEX IF NOT EXISTS sol_index ON {table_name} (sol);')

    drop_index_5 = sqlalchemy.text(f'DROP INDEX IF EXISTS obs_qual_index;')
    create_index_5 = sqlalchemy.text(f'CREATE INDEX IF NOT EXISTS obs_qual_index ON {table_name} (obs_qual);')

    # Initialize the database engine
    engine = sqlalchemy.create_engine(db_url, echo=True)

    # Execute the index statements
    with engine.connect() as conn:
        try:
            conn.execute(drop_index_1)
            conn.execute(drop_index_2)
            conn.execute(drop_index_3)
            conn.execute(drop_index_4)
            conn.execute(drop_index_5)

            conn.execute(create_index_1)
            conn.execute(create_index_2)
            conn.execute(create_index_3)
            conn.execute(create_index_4)
            conn.execute(create_index_5)

            conn.commit()  # Explicitly commit

        except SQLAlchemyError as e:
            logging.error(f'An error occured: {e}')

    
if __name__ == '__main__':
    cli()
