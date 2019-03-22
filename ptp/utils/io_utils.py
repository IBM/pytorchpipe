# Copyright (C) tkornuta, IBM Corporation 2019
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Tomasz Kornuta"

import os
import sys
import shutil
import zipfile
import time
import csv
import urllib
from pathlib import Path



def save_string_list_to_txt_file(folder, filename, data):
    """ 
    Writes list of strings to txt file.

    :param data: List containing strings (sententes, words etc.).
    """
    # Make sure directory exists.
    os.makedirs(os.path.dirname(folder +'/'), exist_ok=True)

    # Write elements in separate lines.        
    with open(os.path.join(folder, filename), mode='w+') as txtfile:
        txtfile.write('\n'.join(data))


def load_string_list_from_txt_file(folder, filename):
    """
    Loads data from txt file.

    :return: List of strings (e.g. list of sententes).
    """
    data = []
    with open(os.path.join(folder, filename), mode='rt') as txtfile:
        for line in txtfile:
            # Remove next line char.
            if line[-1] == '\n':
                line = line[:-1]
            data.append(line)
    return data


def load_mappings_from_csv_file(folder, filename):
    """
    Loads mappings (word:id) from csv file.

    .. warning::
            There is an assumption that file will contain key:value pairs (no content checking for now!)

    :param filename: File with encodings (absolute path + filename).
    :return: dictionary with word:index keys
    """        
    file_path = os.path.join(os.path.expanduser(folder), filename)

    with open(file_path, mode='rt') as csvfile:
        # Check the presence of the header.
        sniffer = csv.Sniffer()
        first_bytes = str(csvfile.read(256))
        has_header = sniffer.has_header(first_bytes)
        # Rewind.
        csvfile.seek(0)  
        reader = csv.reader(csvfile)
        # Skip the header row.
        if has_header:
            next(reader)  
        # Read the remaining rows.
        ret_dict = {rows[0]:int(rows[1]) for rows in reader}
    return ret_dict


def save_mappings_to_csv_file(folder, filename, word_to_ix, fieldnames = []):
    """
    Saves mappings dictionary to a file.

    :param filename: File with encodings (absolute path + filename).
    :param word_to_ix: dictionary with word:index keys
    """
    # Make sure directory exists.
    os.makedirs(os.path.dirname(folder +'/'), exist_ok=True)

    file_path = os.path.join(os.path.expanduser(folder), filename)

    with open(file_path, mode='w+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Create header.
        writer.writeheader()

        # Write word-index pairs.
        for (k,v) in word_to_ix.items():
            #print("{} : {}".format(k,v))
            writer.writerow({fieldnames[0]:k, fieldnames[1]: v})


def get_project_root() -> Path:
    """
    Returns project root folder.
    """
    return Path(__file__).parent.parent


def check_file_existence(directory, filename):
    """
    Checks if file exists.

    :param directory: Relative path to to folder.
    :type directory: str

    :param filename: Name of the file.
    :type directory: str
    """
    file_to_check = os.path.join(os.path.expanduser(directory), filename)
    if os.path.isdir(directory) and os.path.isfile(file_to_check):
        return True
        #self.logger.info('Dataset found at {}'.format(file_folder_to_check))
    else:
        return False

def check_files_existence(directory, filenames):
    """
    Checks if all files in the list exist.

    :param directory: Relative path to to folder.
    :type directory: str

    :param filename: List of files
    :type directory: List of strings or a single string with filenames separated by spaces)
    """
    # Check directory existence.
    if not os.path.isdir(directory):
        return False

    # Process list of files.
    if type(filenames) == str:
        filenames = filenames.split(" ")

    # Check files one by one.
    for file in filenames:
        file_to_check = os.path.join(os.path.expanduser(directory), file)
        if not os.path.isfile(file_to_check):
            return False
    # Ok.
    return True
    

def download(directory, filename, url):
    """
    Checks whether a file or folder exists at given path (relative to storage folder), \
    otherwise downloads files from the given URL.

    :param directory: Relative path to to folder.
    :type directory: str

    :param filename: Name of the file.
    :type directory: str

    :param url: URL to download the file from.
    :type url: str
    """
    if check_file_existence(directory, filename):
        return
    
    # Make sure directory exists.
    os.makedirs(os.path.dirname(directory +'/'), exist_ok=True)

    # Download.
    file_result = os.path.join(os.path.expanduser(directory), filename)
    urllib.request.urlretrieve(url, os.path.expanduser(file_result), reporthook)
    #self.logger.info('Downloading {}'.format(url))

def reporthook(count, block_size, total_size):
    """
    Progress bar function.
    """
    global start_time
    if count == 0:
            start_time = time.time()
            return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def download_extract_zip_file(logger, data_folder, url, zipfile_name):
    """
    Method downloads zipfile from given URL and extracts it.

    :param logger: Logger object used for logging information during download.

    :param data_folder: Folder in which data will be stored.

    :param url: URL from which file will be downloaded.

    :param zipfile_name: Name of the zip file to be downloaded and extracted.

    """
    logger.info("Initializing download in folder {}".format(data_folder))

    if not check_file_existence(data_folder, zipfile_name):
        logger.info("Downloading file {} from {}".format(zipfile_name, url))
        download(data_folder, zipfile_name, url)
    else:
        logger.info("File {} found in {}".format(zipfile_name, data_folder))

    # Extract data from zip.
    logger.info("Extracting data from {}".format(zipfile_name))
    with zipfile.ZipFile(os.path.join(data_folder, zipfile_name), 'r') as zip_ref:
        zip_ref.extractall(data_folder)
    
    logger.info("Download and extraciton finished")

def move_files_between_dirs(logger, source_folder, dest_folder, filenames):
    """
    Moves files between directories
    """
    # Process list of files.
    if type(filenames) == str:
        filenames = filenames.split(" ")

    for f in filenames:
        shutil.move(os.path.join(source_folder, f) , os.path.join(dest_folder, f))

    logger.info("Moved {} from {} to {}".format(filenames, source_folder, dest_folder))

