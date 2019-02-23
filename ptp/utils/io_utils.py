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
import time
import errno
import csv
import urllib
from pathlib import Path



def save_list_to_txt_file(folder, filename, data):
    """ 
    Writes data to txt file.
    """
    # Check directory existence.
    if not os.path.exists(os.path.dirname(folder)):
        try:
            os.makedirs(os.path.dirname(folder))
        except OSError as exc:
            # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise        
    # Write elements in separate lines.        
    with open(folder+'/'+filename, mode='w+') as txtfile:
        txtfile.write('\n'.join(data))


def load_list_from_txt_file(folder, filename):
    """
    Loads data from txt file.
    """
    data = []
    with open(folder+'/'+filename, mode='rt') as txtfile:
        for line in txtfile:
            if line[-1] == '\n':
                line = line[:-1]
            data.append(line)
    return data


def load_dict_from_csv_file(folder, filename):
    """
    Loads data from csv file.

    .. warning::
            There is an assumption that file will contain key:value pairs (no content checking for now!)

    :param filename: File with encodings (absolute path + filename).
    :return: dictionary with word:index keys
    """        
    file_path = os.path.expanduser(folder) + "/" + filename

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


def save_dict_to_csv_file(folder, filename, word_to_ix, fieldnames = []):
    """
    Saves dictionary to a file.

    :param filename: File with encodings (absolute path + filename).
    :param word_to_ix: dictionary with word:index keys
    """
    file_path = os.path.expanduser(folder) + "/" + filename

    # Check directory existence.
    if not os.path.exists(os.path.dirname(folder)):
        try:
            os.makedirs(os.path.dirname(folder))
        except OSError as exc:
            # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise        

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



# Function to make check and download easier
def check_and_download(file_folder_to_check, url=None, download_name='~/data/downloaded'):
    """
    Checks whether a file or folder exists at given path (relative to storage folder), \
    otherwise downloads files from the given URL.

    :param file_folder_to_check: Relative path to a file or folder to check to see if it exists.
    :type file_folder_to_check: str

    :param url: URL to download files from.
    :type url: str

    :param download_name: What to name the downloaded file. (DEFAULT: "downloaded").
    :type download_name: str

    :return: False if file was found, True if a download was necessary.

    """

    file_folder_to_check = os.path.expanduser(file_folder_to_check)
    if not (os.path.isfile(file_folder_to_check) or os.path.isdir(file_folder_to_check)):
        if url is not None:
            #self.logger.info('Downloading {}'.format(url))
            urllib.request.urlretrieve(url, os.path.expanduser(download_name), reporthook)
            return True
        else:
            return True
    else:
        #self.logger.info('Dataset found at {}'.format(file_folder_to_check))
        return False

# Progress bar function
def reporthook(count, block_size, total_size):
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
