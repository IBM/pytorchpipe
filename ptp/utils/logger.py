# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2019
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

import logging
import logging.config as logging_config

from ptp.utils.app_state import AppState

def initialize_logger(name, add_file_handler = True):
    """
    Initializes the logger, with a specific configuration.
    Requires that AppState has the following variable already set:
        - AppState().args.log_level -- log level (from command line argsuments)

    :param name: Name of the entity that "owns" the logger.

    :return: Logger object.

    """
    # Load the default logger configuration.
    logger_config = {'version': 1,
                        'disable_existing_loggers': False,
                        'formatters': {
                            'simple': {
                                'format': '[%(asctime)s] - %(levelname)s - %(name)s >>> %(message)s',
                                'datefmt': '%Y-%m-%d %H:%M:%S'}},
                        'handlers': {
                            'console': {
                                'class': 'logging.StreamHandler',
                                'level': 'INFO',
                                'formatter': 'simple',
                                'stream': 'ext://sys.stdout'}},
                        'root': {'level': 'DEBUG',
                                'handlers': ['console']}}

    logging_config.dictConfig(logger_config)

    # Create the Logger, set its label and logging level.
    logger = logging.getLogger(name=name)

    # Add file handler - when the file is initialized...
    if add_file_handler:
        add_file_handler_to_logger(logger)

    # Set logger level depending on the settings.
    if AppState().args is not None and AppState().args.log_level is not None:
        logger.setLevel(getattr(logging, AppState().args.log_level.upper(), None))
    else:
        logger.setLevel('INFO')

    return logger


def add_file_handler_to_logger(logger):
    """
    Add a ``logging.FileHandler`` to the logger.
    Requires that AppState has the following variable already set:

        - AppState().logfile - name (with path) of the file that everything will be used as destination by the ``FileHandler``.

    :param logger: Logger object.

    """
    # This makes 
    if AppState().log_file == None:
        return

    # Create file handler which logs even DEBUG messages.
    fh = logging.FileHandler(AppState().log_file)

    # Set logging level for this file.
    fh.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers.
    formatter = logging.Formatter(fmt='[%(asctime)s] - %(levelname)s - %(name)s >>> %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)

    # Add the handler to the logger.
    logger.addHandler(fh)
