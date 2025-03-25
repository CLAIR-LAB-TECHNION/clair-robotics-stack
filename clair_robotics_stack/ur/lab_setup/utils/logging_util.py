import logging
import os
import logging
import os
from datetime import datetime
import sys

def setup_logging(log_dir="logs"):
    # Check if logging is already configured
    if len(logging.getLogger().handlers) > 0:
        logging.info("Logging is already set up. Skipping reconfiguration.")
        return

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
        level=logging.DEBUG
    )

    # Log a message indicating the logging setup is complete
    logging.info(f"Logging setup complete for {script_name}")


if __name__ == "__main__":
    setup_logging()
    setup_logging()

    logging.info("This is an info message")
    logging.debug("This is a debug message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")
    logging.critical("This is a critical message")