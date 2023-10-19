import logging
from logging.handlers import TimedRotatingFileHandler
from tf_seg.config import LOGGER_NAME, LOG_FOLDER
import os

# Constants for logging setup
log_file = f"{LOGGER_NAME.lower()}.log"


def set_logger(logger_name, log_file, log_folder):
    """
    Sets up a logger with specified configurations.

    Returns:
        logger: Configured logger object.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Formatter to display log messages
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.set_name(f"{logger_name}_stream_handler")

    # Ensure log folder exists
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, log_file)

    file_handler = TimedRotatingFileHandler(log_file, when="M", interval=1, backupCount=0)
    file_handler.setFormatter(formatter)
    file_handler.set_name(f"{logger_name}_file_handler")

    # Clear existing handlers to avoid duplication
    logger.handlers = []

    # Add handlers to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


class Logger:
    """
    A Logger class to handle application logging.

    Attributes
    ----------
    logger : logging.Logger
        Python logger instance

    Methods
    -------
    debug(message)
        Logs a debug message.
    info(message)
        Logs an info message.
    warning(message)
        Logs a warning message.
    error(message)
        Logs an error message.
    critical(message)
        Logs a critical message.
    """

    def __init__(self, logger_name=LOGGER_NAME, log_file=log_file, log_folder=LOG_FOLDER, level=logging.INFO):
        """
        Parameters
        ----------
        name : str
            Name of the logger
        log_file : str, optional
            File to log messages to (default is None)
        level : int, optional
            Logging level (default is logging.INFO)
        """
        self.logger_name = logger_name
        self.level = level
        self.log_file = log_file

        self.logger = set_logger(logger_name, log_file, log_folder)

        # File handler
        # if log_file:
        #     if not os.path.exists('logs'):
        #         os.makedirs('logs')
        #     fh = logging.FileHandler(f'logs/{log_file}')
        #     fh.setLevel(level)
        #     fh.setFormatter(formatter)
        #     self.logger.addHandler(fh)

    def debug(self, message):
        """
        Logs a debug message.

        Parameters
        ----------
        message : str
            Message to log
        """
        self.logger.debug(message)

    def info(self, message):
        """
        Logs an info message.

        Parameters
        ----------
        message : str
            Message to log
        """
        self.logger.info(message)

    def warning(self, message):
        """
        Logs a warning message.

        Parameters
        ----------
        message : str
            Message to log
        """
        self.logger.warning(message)

    def error(self, message):
        """
        Logs an error message.

        Parameters
        ----------
        message : str
            Message to log
        """
        self.logger.error(message)

    def critical(self, message):
        """
        Logs a critical message.

        Parameters
        ----------
        message : str
            Message to log
        """
        self.logger.critical(message)
