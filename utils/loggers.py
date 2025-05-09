import logging
import logging.config
import json
import os
from pathlib import Path
from dotenv import find_dotenv

def init_logger(file_var: str, config_file: str = "logging_config.json") -> logging.Logger:
    """
    Initializes and returns a module-specific logger with both console and file handlers.

    This function determines the project's root directory using the location of the `.env` file,
    then creates a structured log file path under a `logs/` directory that mirrors the module's
    relative path. It loads a base logging configuration from a JSON file, adds a file handler 
    dynamically for the calling module, and applies the configuration using `dictConfig`.

    Args:
        file_var (str): The `__file__` variable from the calling module, used to derive the log path.
        config_file (str): Path to a JSON logging configuration file (default: "logging_config.json").

    Returns:
        logging.Logger: A configured logger instance for the calling module.

    Raises:
        FileNotFoundError: If the `.env` file cannot be located to determine the project root.
        json.JSONDecodeError: If the logging config file is malformed.
        OSError: If log directories cannot be created or written to.
    """
    # file_path is usually __file__
    path_obj = Path(file_var).resolve()
    
    # Use location of .env file as project root
    env_path = find_dotenv()
    if not env_path:
        raise FileNotFoundError("Could not locate .env file to determine project root.")
    project_root = Path(env_path).parent

    # Calculate module name from file relative to project root
    try:
        relative_path = path_obj.with_suffix('').relative_to(project_root)
        module_parts = relative_path.parts
    except ValueError:
        module_parts = path_obj.with_suffix('').parts[-3:]  # fallback

    name_var = ".".join(module_parts)

    # Load base config
    with open(config_file, "r") as f:
        config = json.load(f)

    # Create dynamic log file path
    module_parts = name_var.split(".")
    log_dir = Path("logs").joinpath(*module_parts[:-1])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{module_parts[-1]}.log"

    # Add a file handler dynamically
    file_handler_name = f"{name_var}_file"
    config["handlers"][file_handler_name] = {
        "class": "logging.FileHandler",
        "formatter": "default",
        "level": "DEBUG",
        "filename": str(log_file)
    }

    # Add logger entry
    config["loggers"] = config.get("loggers", {})
    config["loggers"][name_var] = {
        "handlers": ["console", file_handler_name],
        "level": "DEBUG",
        "propagate": False
    }

    # Apply config and return logger
    logging.config.dictConfig(config)
    return logging.getLogger(name_var)
