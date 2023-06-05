import logging
import os


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

    def __init__(self, name, log_file=None, level=logging.INFO):
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
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Formatter for our log messages
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

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
