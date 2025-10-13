# MERCAP
Mars Examination foR ClimAte Patterns

Using Mars Climate Sounder observations and known dust storm locations from the Mars Dust Storm Sequence Dataset (MDSSD) and Mars Dust Activity Database (MDAD).

# Overview

MERCAP harmonizes atmospheric observations from MCS with MDSSD/MDAD using geospatially-enabled databases. This framework enables targeted extraction and analysis of atmospheric data (dust opacity, temperature, and surface temperature) associated with specific storm events. After extracting coincident atmospheric measurements, we can use that data to study of how local and regional dust storms impact the Martian atmosphere and surface.

Further, we can use "control" measurements from the same location and season (but different Mars years) to estimate how much any given storm event perturbed the atmosphere.

![MERCAP Overview](docs/source/_static/images/data_processing_v3.png)

# Installation

## Dependencies
Install the mcs_tools package:
```
git clone https://github.com/cloudspotting-on-mars/mcstools
cd mcstools
pip install -e .
```

## Install mercap
`pip install -e .`
# Usage
## Creating a local, geospatially enabled DB

The below shows how to ingest MCS profile data, MDSSD storms, and MDAD storms into a single database for search.
### Start a Postgres server (try Postgres App if on Mac)
### [First time] Enable geospatial extension
1. Connect to DB. If in Postgres app, double click on a DB (e.g., try the default `postgres` table)
1. After a terminal opens (and you can see `postgres=#`) add the postgis extension by entering `CREATE EXTENSION postgis;`. You only need to do this once per database.
### Transcribe MCS Data
1. Export some MCS data on one of the MCS machines. For example, get 10 years of a few DDR1 columns:
```mysql_lun -e "SELECT dt, Date, UTC, Profile_lon, Profile_lat, l_s, LTST from profiles_2d_v6 where dt >= '2007-10-01 00:00:00' and dt < '2017-07-01 00:00:00';" > mcs_ddr1_matching_mdssd.txt```
1. If data is remote, move it to your local machine like:
    ```
    rsync -avzhe ssh <user>@<remote_machine>:/remote/path/to/mcs_ddr1_matching_mdssd.txt /local/path/to/mcs_ddr1_matching_mdssd.txt --progress
    ```
1. Ingest MCS data using `transcribe_mcs_to_db.py`
    The below will copy your MCS data to the `postgres` database. Feel free to substitute in a different database if not using the default `postgres`. Note that the below will overwrite any previous data in the `mcs_profiles_2d` table.

    You can run a fast sanity check by passing the `--smoke_test` flag.

    ```
    python mercap/utils/transcribe_mcs_to_db.py \
    --mcs_fpath '/local/path/to/mcs_ddr1_matching_mdssd.txt' \
    --db_url 'postgresql://postgres:postgres@localhost:5432/postgres' \
    --table_name 'mcs_profiles_2d' \
    --write_conflict_behavior 'replace'
    ```

### Process/ingest MDSSD Data
When processing/ingesting the MDSSD data, you can save to a CSV and/or a database. The below example shows both.
```
python mercap/parse_mdssd.py \
--mdssd-head-dir '/path/to/mdssd' \
--csv-output-fpath '/path/to/mdssd_processed_geometry_augmented.csv' \
--db_url 'postgresql://postgres:postgres@localhost:5432/postgres' \
--n_jobs 64
```

### Process/ingest MDAD Data
Similar to MDSSD, you can save to a CSV and/or a database. The below example shows both. Note, still a WIP
```
python mercap/parse_mdad.py \
--mdssd-head-dir '/path/to/mdad' \
--csv-output-fpath '/path/to/mdad_processed_geometry_augmented.csv' \
--db_url 'postgresql://postgres:postgres@localhost:5432/postgres' \
```

# Run spatiotemporal queries

With all the information in one place, you can now query for profiles that align (or don't align) with storms. For examples on different database queries, see:
* `experiments/mdssd_target_data_processing/MDSSD_Queries_v1.ipynb`
* `experiments/bkgd_haze_viz/bkgd_haze_plots_v1.ipynb`