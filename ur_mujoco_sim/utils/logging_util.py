import logging
import os
from datetime import datetime
import sys


def setup_logging(log_dir="logs"):
    # Remove all existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Get the script name
    script_name = os.path.basename(sys.argv[0]).replace(".py", "")

    # Define the log file name with the script name and current date and time
    log_file = os.path.join(log_dir, f"{script_name}_{datetime.now().strftime('d%Y-%m-%d_t%H-%M-%S')}.log")

    # Configure logging
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG,
        force=True  # This forces reconfiguration
    )

    # Log a message indicating the logging setup is complete
    logging.info(f"Logging setup complete for {script_name}")


# Usage in your main script or at the top of your module:
setup_logging()