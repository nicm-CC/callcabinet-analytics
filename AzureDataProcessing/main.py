# Todo - add functionality in 'params' to reprocess all files within 'offset' days

# Todo - Set up CallCabinet GitHub repo
# Todo - Investigate use of 'from fastparquet import ParquetFile' to read parquet files
# Todo - General code cleanup and stability improvements
# Todo - Ensure suitability across all environments & clients
# Todo - Ensure all functions are documented
# Todo - Create Azure Function from this script
"""
Script Details:
    - This script will process all json files in a given Azure blob container
    - The script will process files in the following order:
        1. Process recovery file if it exists
        2. Get list of base directories
        3. Get list of directories to process
        4. Get list of files to process
        5. Check if files exist in master already
        6. Process files
    - Data processing steps:
        1. Streaming json files into memory
        2. Transforming data
        3. Save new data to Azure
        4. Clear memory after each run
    - Parameters:
        - offset: number of days to offset from today
        - num_days_per_run: number of days to process per run
        - test_files: boolean to determine if test files should be processed
        - connection_string: Azure connection string
        - container_name: Azure container name
"""

# standard dependencies
import pandas as pd
import numpy as np
import logging
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

# import custom dependencies
from ProcessData import (
    process_recovery_file,
    get_master_data_parquet,
    get_base_dirs,
    get_directory_list,
    get_file_names,
    check_processed_files,
    get_json_data,
    transform_data,
    save_new_data_local,
    load_local_data,
    save_new_data_parquet,
    clean_memory,
)

logging.basicConfig(level=logging.DEBUG)

def run(params):

    # load master data before recovery
    master_data = get_master_data_parquet(params["connection_string"], params["container_name"])

    # Process recovery file if it exists and append to master
    process_recovery_file(params["connection_string"], params["container_name"])

    # load master data after recovery
    master_data = get_master_data_parquet(params["connection_string"], params["container_name"])

    # Get list of base directories
    base_dirs = get_base_dirs(params["connection_string"], params["container_name"])

    for i in range(1, params['offset'] + 1):

        # Set offset
        offset = params['offset'] - i

        # Get list of directories to process
        directory = get_directory_list(1, offset)

        # Get list of files to process
        file_names = get_file_names(base_dirs, directory, params["connection_string"], params["container_name"])

        # check if files exist in master already
        file_names = check_processed_files(file_names, directory, master_data, params["exception_sites"])

        # Process files
        if len(file_names) > 0:

            # Streaming json files into memory
            data = get_json_data(file_names, params["connection_string"], params["container_name"])

            # Transforming data
            cleaned_data = transform_data(data)

            # clean data
            cleaned_df = pd.DataFrame.from_records(cleaned_data)
            replace_dict = {float('nan'): None,
                            'nan': None,
                            np.nan: None,
                            "": None,
                            "None": None}

            cleaned_df.replace(replace_dict, inplace=True)
            cleaned_df = cleaned_df.astype(str)

            # Save new data to local
            save_new_data_local(cleaned_df)

            # Clear memory
            clean_memory(False, file_names, data, cleaned_data, cleaned_df)

        else:
            logging.info("No new files to process")
            pass

    # Loads intermediate data from local
    data = load_local_data()

    # Standardize CallID
    if 'CallID' in data.columns:
        data['CallID'] = data['CallID'].fillna(data['callId'])

    # Appends to master and saves to Azure
    save_new_data_parquet(data, params["connection_string"], params["container_name"])

    # Clear memory
    clean_memory(True)


if __name__ == "__main__":
    params_LS = {
        "connection_string": config['LifeStorage']['connection_string'],
        "container_name": config['PPS']['container_name'],
        "num_days_per_run": 1,
        "offset": 2,
        "exception_sites": [{"site_id": "897929a0-b4c8-446f-924f-1649c402c978",
                             "field": "CallSummary"}],
        "test_files": False}

    params_PPS = {
        "connection_string": config['PPS']['connection_string'],
        "container_name": config['PPS']['container_name'],
        "num_days_per_run": 1,
        "offset": 2,
        "exception_sites": None,
        "test_files": False}

    print("Starting CallCabinet Data Processing for LifeStorage")
    run(params_LS)

    print("Starting CallCabinet Data Processing for PPS")
    run(params_PPS)
