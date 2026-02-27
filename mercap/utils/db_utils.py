import logging

import pandas as pd
from sqlalchemy import create_engine, exc, text


logging.basicConfig(level=logging.INFO,
                    datefmt='%H:%M:%S',
                    format='%(asctime)s | %(lineno)d | %(levelname)-5s | %(module)-15s | %(message)s')
 
def sqlalchemy_engine_check(engine):
    """Helper to check if an engine string refers to a valid, live database"""
    try:
        # Create engine
        engine = create_engine(engine)
        
        # Attempt to connect to the database
        connection = engine.connect()
        
        # Close the connection
        connection.close()
    except exc.SQLAlchemyError as e:
        print(f"Error connecting to the database: {e}")
        return

def get_database_table_info(connection, verbose=False):
    """
    Retrieves information about tables in the database including row counts
    for key tables used in storm and profile analysis.
    
    Parameters
    ----------
    connection : SQLAlchemy connection
        Active connection to the database
    verbose : bool, optional
        Whether to log additional information, by default False
        
    Returns
    -------
    dict
        Dictionary containing database information with keys:
        - 'tables': list of all table names
        - 'counts': dictionary with row counts for specific tables
    """
    # Get list of all tables
    text_statement = text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    tables_df = pd.read_sql(text_statement, connection)
    table_names = tables_df['table_name'].tolist()
    
    # Get row counts for specific tables of interest
    count_data = {}
    column_data = {}
    tables_to_count = ['mdssd_table', 'mdad_table', 'mcs_profiles_2d']
    
    for table in [t for t in tables_to_count if t in table_names]:
        text_statement = text(f"SELECT COUNT(*) as count FROM {table}")
        count_df = pd.read_sql(text_statement, connection)
        count_data[table] = int(count_df['count'].iloc[0])

        column_data[table] = pd.read_sql(text(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'"), connection)

    result = {
        'tables': table_names,
        'counts': count_data
    }
    
    if verbose:
        logging.info(f"\n\nFound {len(table_names)} tables in database")
        for table_name in table_names:
            logging.info(f"  {table_name}")

        logging.info("\n\nTable row counts for tables of interest:")
        for table, count in count_data.items():
            logging.info(f"  {table}:\t\t{count:12,} rows")

        logging.info("\n\nTable column names for tables of interest:")
        for table, columns in column_data.items():
            logging.info(f"  {table}:{columns.to_string(index=False)}")
    
    return result