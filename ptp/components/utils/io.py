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
import numpy as np
import shutil
import zipfile
import time
import requests
from pathlib import Path


def save_nparray_to_csv_file(folder, filename, nparray, sep=','):
    """ 
    Writes numpy array to csv file.

    :param folder: Relative path to to folder.
    :type folder: str

    :param filename: Name of the file.
    :type filename: str

    :param nparray: Numpy array.

    :param sep: Separator (DEFAULT: ',')
    """
    # Make sure folder exists.
    os.makedirs(os.path.dirname(os.path.expanduser(folder) +'/'), exist_ok=True)

    name = os.path.join(os.path.expanduser(folder), filename)
    print(name)
    
    # Write array to file, separate elements with commas.
    nparray.tofile(name, sep=sep, format="%s")


def load_nparray_from_csv_file(folder, filename, dtype=float, sep=','):
    """ 
    Loads numpy array from csv file.

    :param folder: Relative path to to folder.
    :type folder: str

    :param filename: Name of the file.
    :type filename: str

    :param dtype: Type of the array to load (DEFAULT: float)

    :return: Numpy array.
    """
    # Absolute pathname of the file.
    name = os.path.join(os.path.expanduser(folder), filename)
    
    # Load array from file
    nparray = np.fromfile(name, dtype, count=-1, sep=',')

    # Return it.
    return nparray



def save_string_list_to_txt_file(folder, filename, data):
    """ 
    Writes list of strings to txt file.

    :param folder: Relative path to to folder.
    :type folder: str

    :param filename: Name of the file.
    :type filename: str

    :param data: List containing strings (sententes, words etc.).
    """
    # Make sure folder exists.
    os.makedirs(os.path.dirname(os.path.expanduser(folder) +'/'), exist_ok=True)

    # Write elements in separate lines.        
    with open(os.path.join(os.path.expanduser(folder), filename), mode='w+') as txtfile:
        txtfile.write('\n'.join(data))


def load_string_list_from_txt_file(folder, filename):
    """
    Loads data from txt file.

    :return: List of strings (e.g. list of sententes).
    """
    data = []
    with open(os.path.join(os.path.expanduser(folder), filename), mode='rt') as txtfile:
        for line in txtfile:
            # Remove next line char.
            if line[-1] == '\n':
                line = line[:-1]
            data.append(line)
    return data


def get_project_root() -> Path:
    """
    Returns project root folder.
    """
    return Path(__file__).parent.parent


def check_file_existence(folder, filename):
    """
    Checks if file exists.

    :param folder: Relative path to to folder.
    :type folder: str

    :param filename: Name of the file.
    :type filename: str
    """
    file_to_check = os.path.join(os.path.expanduser(folder), filename)
    if os.path.isdir(folder) and os.path.isfile(file_to_check):
        return True
        #self.logger.info('Dataset found at {}'.format(file_folder_to_check))
    else:
        return False

def check_files_existence(folder, filenames):
    """
    Checks if all files in the list exist.

    :param folder: Relative path to to folder.
    :type folder: str

    :param filename: List of files
    :type lst: List of strings or a single string with filenames separated by spaces)
    """
    # Check folder existence.
    if not os.path.isdir(folder):
        return False

    # Process list of files.
    if type(filenames) == str:
        filenames = filenames.split(" ")

    # Check files one by one.
    for file in filenames:
        file_to_check = os.path.join(os.path.expanduser(folder), file)
        if not os.path.isfile(file_to_check):
            return False
    # Ok.
    return True
    

def download(folder, filename, url):
    """
    Checks whether a file or folder exists at given path (relative to storage folder), \
    otherwise downloads files from the given URL.

    :param folder: Relative path to to the folder.
    :type folder: str

    :param filename: Name of the file.
    :type filename: str

    :param url: URL to download the file from.
    :type url: str
    """
    if check_file_existence(folder, filename):
        return
    
    # Make sure folder exists.
    os.makedirs(os.path.dirname(folder +'/'), exist_ok=True)

    # Download.
    file_result = os.path.join(os.path.expanduser(folder), filename)

    with open(os.path.expanduser(file_result), "wb") as f:
        global start_time
        start_time = time.time()
        r = requests.get(url)
        content_length = int(r.headers.get('content-length', None))
        count = 0

        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                count += 1
                reporthook(count, 1024, content_length)

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

def download_extract_zip_file(logger, folder, url, zipfile_name):
    """
    Method downloads zipfile from given URL and extracts it.

    :param logger: Logger object used for logging information during download.

    :param folder: Folder in which data will be stored.

    :param url: URL from which file will be downloaded.

    :param zipfile_name: Name of the zip file to be downloaded and extracted.

    """
    logger.info("Initializing download in folder {}".format(folder))

    if not check_file_existence(folder, zipfile_name):
        logger.info("Downloading file {} from {}".format(zipfile_name, url))
        download(folder, zipfile_name, url)
    else:
        logger.info("File {} found in {}".format(zipfile_name, folder))

    # Extract data from zip.
    logger.info("Extracting data from {}".format(zipfile_name))
    with zipfile.ZipFile(os.path.join(folder, zipfile_name), 'r') as zip_ref:
        zip_ref.extractall(folder)
    
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

