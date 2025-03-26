import datetime
import os
import sys

from loguru import logger

from src.config import cfg

logger_cfg = cfg.logger


def create_logger():
    """Create logger object
    Returns: a logger object
    """
    # Reset all handlers to ensure consistent configuration
    logger.remove()
    
    # Define a different format for console with vertical bars and colors
    console_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <cyan>{name:>15}:{line:<3}</cyan> | <level>{level:>8}</level> | <level>{message}</level>"
    
    # Add console handler with custom format
    logger.add(
        sys.stderr,
        format=console_format,
        level=logger_cfg.level,
        colorize=True
    )
    
    # Get the current date and time
    current_time = datetime.datetime.now()
    # Format the current date and time as a string
    timestamp = current_time.strftime(logger_cfg.time_format)
    os.makedirs(logger_cfg.log_dir, exist_ok=True)
    # Construct log file name
    filepath = os.path.join(logger_cfg.log_dir, f"{timestamp}{logger_cfg.log_file_extension}")
    # Create logger file with original format
    logger.add(
        filepath, 
        format=logger_cfg.log_format,
        level=logger_cfg.level, 
        rotation=logger_cfg.rotation
    )
    # delete old log files 
    delete_old_logs(logger_cfg.log_dir)
    print(f"Create log file at: {filepath}")
    return logger


def reset_logger():
    """Reset logger configuration after it may have been modified by external libraries
    Returns: reconfigured logger object
    """
    # Reset all handlers
    logger.remove()
    
    # Define a different format for console with vertical bars and colors
    console_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <cyan>{name:>15}:{line:<3}</cyan> | <level>{level:>8}</level> | <level>{message}</level>"
    
    # Add console handler with custom format
    logger.add(
        sys.stderr,
        format=console_format,
        level=logger_cfg.level,
        colorize=True
    )
    
    # Add file handler if needed
    for handler_id, handler in logger._core.handlers.items():
        if hasattr(handler, '_name') and logger_cfg.log_dir in handler._name:
            # File handler already exists
            return logger
    
    # If no file handler exists, add one
    # Get the current date and time
    current_time = datetime.datetime.now()
    # Format the current date and time as a string
    timestamp = current_time.strftime(logger_cfg.time_format)
    # Construct log file name
    filepath = os.path.join(logger_cfg.log_dir, f"{timestamp}{logger_cfg.log_file_extension}")
    # Create logger file with original format
    logger.add(
        filepath, 
        format=logger_cfg.log_format,
        level=logger_cfg.level, 
        rotation=logger_cfg.rotation
    )
    print(f"Logger reset and reconfigured. Log file at: {filepath}")
    return logger


def delete_old_logs(log_dir):
    """Delete log files which created before last 30 days

    Args:
        log_dir (str): directory of logs folder
    """
    today = datetime.date.today()
    for file_name in os.listdir(log_dir):
        if file_name.endswith(logger_cfg.log_file_extension):
            try:
                log_date = datetime.datetime.strptime(file_name.split(".")[0], logger_cfg.time_format).date()
                if (today - log_date).days > logger_cfg.log_existence_days:
                    file_path = os.path.join(log_dir, file_name)
                    os.remove(file_path)
                    print(f"Deleted file: {file_name}")
            except ValueError:
                continue

# Create and configure the logger
logger = create_logger()
