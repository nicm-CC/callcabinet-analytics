import json
import concurrent.futures
import datetime
import uuid
import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import numpy as np


# Azure sdk library
from azure.storage.blob import BlobClient, ContainerClient, BlobBlock
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
from azure.core.pipeline.policies import HttpLoggingPolicy
# Configure the logging module to show WARNING messages
logging.basicConfig(level=logging.WARNING)

# Create an instance of the HTTPLoggingPolicy with a custom logging level
http_logging_policy = HttpLoggingPolicy(log_level=logging.WARNING)

# Set up console logging
logging.basicConfig(level=logging.INFO)

"""
This loads the master data from Azure for later use. The parquet file is loaded
by saving chunks to a temp file and then reading the file into a pandas dataframe.
The temp file is then deleted when the data is loaded into memory.
"""
def get_master_data_parquet(connection_string, container_name):

    print("Getting master parquet file from Azure")

    container_name = container_name + '-master'
    blob_name = 'master_data.parquet'

    # get blob client
    blob_client = BlobClient.from_connection_string(conn_str=connection_string,
                                                    container_name=container_name,
                                                    blob_name=blob_name,
                                                    policies=[http_logging_policy])

    # download blob
    try:

        chunk_size = 1 * 1024 * 1024
        chunks = []

        # delete temp file if it exists
        if os.path.exists('temp.parquet'):
            os.remove('temp.parquet')

        with open("temp.parquet", "wb") as f:
            download_stream = blob_client.download_blob(timeout=720)
            print("Downloading master data...")
            while True:
                chunk = download_stream.read(chunk_size)

                chunks.append(chunk_size)

                # total percentage downloaded
                total = sum(chunks)
                percent = total / blob_client.get_blob_properties().size * 100

                if percent <= 100:
                    print(f"Streaming into memory: {percent}%")

                if not chunk:
                    break
                f.write(chunk)

        # read parquet file into pandas dataframe
        master_data = pd.read_parquet('temp.parquet')

        master_data.to_parquet('master_data.parquet')

        print("Master data retrieved from Azure")

        # delete temp file
        os.remove("temp.parquet")

        return master_data

    except ResourceNotFoundError:
        logging.warning("Master data not found in Azure")

        # delete temp file if it exists
        if os.path.exists('temp.parquet'):
            os.remove('temp.parquet')

        return None


"""
This function returns the top-level directories in the Azure container.
"""
def get_base_dirs(conn_str, container_name):
    base_dirs = []

    # create the container client
    container_client = ContainerClient.from_connection_string(conn_str=conn_str,
                                                              container_name=container_name,
                                                              policies=[http_logging_policy])

    # Get top level folders in container
    for folder in container_client.walk_blobs(name_starts_with="", delimiter='/'):
        base_dirs.append(folder.name)

    # if element in list contains 'master' in the name remove it
    base_dirs = [x for x in base_dirs if 'master' not in x]
    base_dirs = [x for x in base_dirs if 'backup' not in x]

    print("Base directories to process: ")
    print(base_dirs)

    return base_dirs


"""
This function returns a list of directories to process. The number of days to
process is passed in as a parameter. The offset is used to skip the most recent
days. For example, if the offset is 1, the most recent day will not be processed.
Generates a list of directories to process.
"""
def get_directory_list(num_days, offset):
    # Get list of directories to process
    directories_list = []
    for i in range(0, num_days):
        date = datetime.date.today() - datetime.timedelta(days=i + offset)
        directories_list.append(date.strftime("%Y/%m/%d"))

    # format the dates as strings
    directories_list = [date.replace("/0", "/") for date in directories_list]

    # reverse the list so that the oldest dates are processed first
    directories_list.reverse()

    print("============================================")
    print("Days to process: ")
    print(directories_list)

    return directories_list


"""
This function returns a list of file names from the Azure container. The function
uses a thread pool to retrieve the file names in parallel. The function returns
a list of file names given a list of directories and a list of base directories
to look in.
"""
def get_file_names(base_dirs, directories, connection_string, container_name):
    print("Retrieving file names from Azure container...")

    file_names = []
    failed = []

    # create the container client
    container_client = ContainerClient.from_connection_string(conn_str=connection_string,
                                                              container_name=container_name,
                                                              policies=[http_logging_policy])

    def retrieve_file_names(entry, base_dir):
        path = base_dir + entry + "/"
        names = []
        try:
            for file in container_client.walk_blobs(path, delimiter='/'):
                names.append(file.name)
        except HttpResponseError as e:
            logging.warning(f"Failed to retrieve file names for {path} with error: {e}")
            failed.append(path)
        return names

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(retrieve_file_names, entry, base_dir) for entry in directories for base_dir in base_dirs]
        for future in concurrent.futures.as_completed(futures):
            file_names.extend(future.result())

    print(f"Retrieved {len(file_names)} files from Azure container")

    return file_names


"""
This function returns the list of file names that need to be processed. The
function checks the master data to see if the file has already been processed.
If the file has already been processed, it is not added to the list of files
to be processed. The function also checks if the file needs to be updated. If
the file needs to be updated, it is added to the list of files to be processed.
"""
def check_processed_files(file_names, directory, master_data=None, exception_sites=None):
    current_list = pd.DataFrame(file_names, columns=['file_name'])

    # check if the dataframe 'master_data' is not a NoneType

    out = []
    update_count = 0
    new_count = 0

    if master_data is not None:

        """
        Extract ID's from master data
        """
        # extract the last part (ID) from file_names
        current_list['id'] = current_list['file_name'].str.split('/').str[-1]

        # remove file extension
        current_list['id'] = current_list['id'].str.split('.').str[0]

        # retrieve the part of the url after the last '/'
        master_data['id'] = master_data['url'].str.split('/').str[-1]

        """
        Identify new files to be processed if ID's not present in master data
        """
        # all files in current_list that are not in master_data (1)
        new_files = current_list[~current_list['id'].isin(master_data['id'])]
        out.extend(new_files['file_name'])
        new_count = len(new_files)

        """
        Get master data subset from current list
        """
        # all files in master_data that are in the current_list
        current_master_data = master_data[master_data['id'].isin(current_list['id'])]

        """
        Identify files to be updated if 'Field' is empty and listed in exception_sites
        """
        if exception_sites is not None:
            date_base_str = directory[0]
            for site in exception_sites:

                site_id = site["site_id"]
                field = site["field"]

                current_master_data = current_master_data[current_master_data[field] == "None"]
                need_updating = current_master_data[current_master_data['SiteID'] == site_id]

                # Create file name from SiteID, Date and ID
                files_for_updating = need_updating['SiteID'] + "/" + date_base_str + "/" + need_updating['id'] + ".json"

                out.extend(files_for_updating)
                update_count += len(files_for_updating)

        else:
            print("No exception sites provided, skipping update check")

    else:
        print("master_data.parquet not found, processing all files in current list")
        print(f"Total files to process: {len(out)}")

        out = current_list['file_name'].tolist()

    print(f"Total files to process: {len(out)}")
    print(f"New files to process: {new_count}")
    print(f"Files to update: {update_count}")

    # Check cases where few files are processed
    if new_count < 5:
        print("Files to process: ")

    return out


"""
This function is a helper function that returns the data from a file in the
Azure container given the file name and the connection string.
"""
def get_file_data(
        entry,
        conn_str,
        container_name
):
    blob_client = BlobClient.from_connection_string(conn_str=conn_str,
                                                    container_name=container_name,
                                                    blob_name=entry,
                                                    policies=[http_logging_policy])
    download_stream = blob_client.download_blob()

    return json.load(download_stream)


"""
This function returns the JSON data from the files in the Azure container. The
function uses a thread pool to retrieve the data in parallel. The function returns
a list of JSON data using the helper function 'get_file_data'.
The function saves the data to a record file in case the process is interrupted.
"""
def get_json_data(
        json_files,
        connection_string,
        container_name,
):
    print(f"Retrieving JSON data from Azure container for {len(json_files)} files")
    print("--------------------------------------------")

    total = len(json_files)
    start_time = datetime.datetime.now()
    count = 1
    num_files = len(json_files)

    json_data = []
    entries = json_files[-num_files:-1]

    # concurrently retrieve the data from the files using a thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_entry = {executor.submit(get_file_data, entry, connection_string, container_name): entry for entry in
                           entries}
        for future in concurrent.futures.as_completed(future_to_entry):

            # time since start_time
            time = round((datetime.datetime.now() - start_time).total_seconds(), 2)

            if count % 100 == 0:
                print(f"Time elapsed: {time} seconds")
                print(f"Files processed: {count} of {num_files}")
                print(f"Files remaining: {total - count}")
                print(f"Average time per file: {round((datetime.datetime.now() - start_time).total_seconds() / count, 2)} seconds")
                print(f"Estimated time remaining: {round((datetime.datetime.now() - start_time).total_seconds() / count * (total - count), 2)} seconds")
                print("--------------------------------------------")

            # Save data to recovery file every 1000 files
            if count % 1000 == 0:

                json_string = json.dumps(json_data)

                with open("recovery.txt", "w") as outfile:
                    outfile.write(json_string)

            entry = future_to_entry[future]
            try:
                data = future.result()
            except Exception as exc:
                logging.warning(f'{entry} generated an exception: {exc}')
            else:
                json_data.append(data)
                count += 1

    print("Process Complete")
    print("Average time per file: " + str(
        (datetime.datetime.now() - start_time).total_seconds() / count) + "seconds")

    return json_data


"""
This function handles the standard data cleaning and transformation for the JSON data
as it comes out of the VOCI analytics pipeline. The function returns a list of cleaned
dictionary objects. There are various handlers for different objects in the JSON data.
"""
def transform_data(
        json_files
):
    results = []

    for json_file in json_files:
        cleaned = {}
        try:

            """
            Remove transcript from file
            """
            json_file.pop("transcript", None)

            """
            Handle 'CallAnalytic' object
            """
            try:
                for entry in json_file["CallAnalytic"]:
                    if type(json_file["CallAnalytic"][entry]) is not dict:
                        cleaned[entry] = json_file["CallAnalytic"][entry]

            except KeyError as ke:
                print(ke)

            """
            Handle 'PlainEVSResponce' object
            """
            evs = json.loads(json_file["PlainEVSResponce"])
            evs_client_data = evs["client_data"]

            for entry in evs_client_data:
                if type(evs_client_data[entry]) is not dict:
                    cleaned[entry] = evs_client_data[entry]

            """
            Handle 'app_data' object
            """
            for entry in json_file["vociAdditionalData"]:
                if type(json_file["vociAdditionalData"][entry]) is not dict:
                    cleaned[entry] = json_file["vociAdditionalData"][entry]

            """
            Handle 'vociAdditionalData' object
            """
            for entry in json_file["vociAdditionalData"]['app_data']:
                if type(json_file["vociAdditionalData"]['app_data'][entry]) is not dict:
                    cleaned[entry] = json_file["vociAdditionalData"]['app_data'][entry]

            """
            Handle 'scorecard' object
            """
            scorecard = json_file["vociAdditionalData"]['app_data']['scorecard']
            for entry in scorecard:
                for k, v in scorecard[entry].items():
                    l1 = scorecard[entry][k]

                    if not l1['subcategories']:
                        path = entry + "." + k + '.score'
                        cleaned[path] = l1['score']

                        # print("Saving score at path: " + path)

                    else:
                        for k1, v1 in l1['subcategories'].items():
                            l2 = l1['subcategories'][k1]

                            if not l2['subcategories']:
                                path1 = entry + "." + k + '.' + k1 + '.score'
                                cleaned[path1] = l2['score']

                                # print("Saving score at path: " + path1)

                            else:
                                for k2, v2 in l2['subcategories'].items():
                                    l3 = l2['subcategories'][k2]

                                    if not l3['subcategories']:
                                        path2 = entry + "." + k + '.' + k1 + '.' + k2 + '.score'
                                        cleaned[path2] = l3['score']

                                        # print("Saving score at path: " + path2)

                                    else:
                                        for k3, v3 in l3['subcategories'].items():
                                            l4 = l3['subcategories'][k3]

                                            if not l4['subcategories']:
                                                path3 = entry + "." + k + '.' + k1 + '.' + k2 + '.' + k3 + '.score'
                                                cleaned[path3] = l4['score']

                                                # print("Saving score at path: " + path3)

                                            else:
                                                try:
                                                    for k4, v4 in l4['subcategories'].items():
                                                        l5 = l4['subcategories'][k3]

                                                        if not l5['subcategories']:
                                                            path4 = entry + "." + k + '.' + k1 + '.' + k2 + '.' + k3 + '.' + k4 + '.score'
                                                            cleaned[path4] = l5['score']

                                                            # print("Saving score at path: " + path4)

                                                except KeyError:
                                                    logging.warning("Maximum of five levels")

            results.append(cleaned)

        except Exception as e:
            logging.warning(e)

    print("Data transformation complete")

    return results


def save_new_data_local(data):

    file_name = 'new_master_data.parquet'
    if os.path.exists(file_name):
        print("Local file exists, appending new data")

        df = pd.read_parquet(file_name)
        df = pd.concat([df, data]).drop_duplicates(keep='last')
        df.to_parquet(file_name)

    else:
        print("Local file does not exist, creating new file")

        df = pd.DataFrame(data)
        df.to_parquet(file_name)

    print("Local file saved")


"""
This function reads the intermediate data file from the local machine. 
If the file does not exist, it returns None.
"""
def load_local_data():
    file_name = 'new_master_data.parquet'
    if os.path.exists(file_name):
        print("Local file exists, loading data")

        df = pd.read_parquet(file_name)
        return df
    else:
        print("Local file does not exist, returning None")
        return None


"""
This loads the master data from local storage. If the master data does not exist, it loads the backup data.
It appends newly transformed data to the master data and saves it to Azure Blob Storage. If no backup data exists,
it creates a master data file from the newly transformed data. 
"""
def save_new_data_parquet(data, connection_string, container_name):
    container_name = container_name + '-master'
    master_blob_name = 'master_data.parquet'
    backup_blob_name = 'backup_data.parquet'

    print("Fetching master parquet file from local storage")

    if os.path.exists(master_blob_name):
        print("Master parquet file exists, loading data")

        master_data = pd.read_parquet(master_blob_name)

        # check if both 'CallID' and 'callId' exist in master_data:
        if 'CallID' in master_data.columns and 'callId' in master_data.columns:
            master_data['CallID'] = master_data['CallID'].fillna(master_data['callId'])

        new_data = data

        print("Appending new data to master data")
        master_data = pd.concat([master_data, new_data]).drop_duplicates(subset='CallID', keep='last')

    elif os.path.exists(backup_blob_name):
        print("Master parquet file does not exist, loading backup data")

        backup_data = pd.read_parquet(backup_blob_name)

        # check if both 'CallID' and 'callId' exist in master_data:
        if 'CallID' in backup_data.columns and 'callId' in backup_data.columns:
            backup_data['CallID'] = backup_data['CallID'].fillna(backup_data['callId'])

        new_data = data

        print("Appending new data to backup data")
        master_data = pd.concat([backup_data, new_data]).drop_duplicates(subset='CallID', keep='last')

    else:
        print("Master & Backup parquet file does not exist, creating new file")

        master_data = data

    print("Converting to acceptable format")

    # drop 'utterances' column if it exists
    if 'utterances' in master_data.columns:
        master_data = master_data.drop(columns=['utterances'])

    # convert float columns to str
    master_data = master_data.astype(str)

    # Convert the Pandas DataFrame to a PyArrow Table
    table = pa.Table.from_pandas(master_data)

    # Write the PyArrow Table to a Parquet file
    pq.write_table(table, "temp_master_data.parquet")

    print("Saving master & backup data to Azure Blob Storage")
    # upload master data to Azure
    blob_client = BlobClient.from_connection_string(conn_str=connection_string,
                                                    container_name=container_name,
                                                    blob_name=master_blob_name,
                                                    policies=[http_logging_policy])

    block_list = []
    chunk_size = 1 * 1024 * 1024

    with open("temp_master_data.parquet", "rb") as pf:
        # blob_client.upload_blob(data=pf, overwrite=True)

        while True:
            read_data = pf.read(chunk_size)
            if not read_data:
                break  # done
            blk_id = str(uuid.uuid4())
            blob_client.stage_block(block_id=blk_id, data=read_data)
            block_list.append(BlobBlock(block_id=blk_id))

    blob_client.commit_block_list(block_list)

    print("Master data updated")

    # upload master as new backup data to Azure
    blob_client = BlobClient.from_connection_string(conn_str=connection_string,
                                                    container_name=container_name,
                                                    blob_name=backup_blob_name,
                                                    policies=[http_logging_policy])

    block_list = []
    chunk_size = 1 * 1024 * 1024

    with open("temp_master_data.parquet", "rb") as pf:
        # blob_client.upload_blob(data=pf, overwrite=True)

        while True:
            read_data = pf.read(chunk_size)
            if not read_data:
                break  # done
            blk_id = str(uuid.uuid4())
            blob_client.stage_block(block_id=blk_id, data=read_data)
            block_list.append(BlobBlock(block_id=blk_id))

    blob_client.commit_block_list(block_list)

    print("Backup data updated")

    return


"""
This function cleans up the memory by deleting all temporary files.
final = True means that the function is called at the end of the program.
"""
def clean_memory(final, *args):
    for arg in args:
        del arg

    if os.path.exists('temp.csv'):
        os.remove('temp.csv')

    if os.path.exists('temp.parquet'):
        os.remove('temp.parquet')

    if os.path.exists('temp_master_data.parquet'):
        os.remove('temp_master_data.parquet')

    if final is True:

        if os.path.exists('new_master_data.parquet'):
            os.remove('new_master_data.parquet')

        if os.path.exists('master_data.parquet'):
            os.remove('master_data.parquet')

        if os.path.exists('recovery.txt'):
            os.remove('recovery.txt')

    print("Memory cleaned")


def process_recovery_file(connection_string, container_name):

    # check if recovery json file exists
    if os.path.exists("recovery.txt"):
        print("Recovery file found")
        print("=====================================")

        # load recovery file
        with open("recovery.txt", "r") as file:
            # Load the JSON data from the file
            json_string = file.read()

        data = json.loads(json_string)

        # process recovery file
        cleaned_data = transform_data(data)
        cleaned_df = pd.DataFrame.from_records(cleaned_data)
        replace_dict = {float('nan'): None,
                        'nan': None,
                        np.nan: None,
                        "": None,
                        "None": None}
        cleaned_df.replace(replace_dict, inplace=True)

        print("Master data exists, appending new data")

        # save recovered data to Azure
        save_new_data_parquet(cleaned_df, connection_string, container_name)

        clean_memory(data, cleaned_data, cleaned_df)

        print("Recovered data processed successfully, proceeding with normal run")

        os.remove("recovery.txt")

    else:
        print("No recovery file found, proceeding with normal run")
        print("=====================================")

    return
