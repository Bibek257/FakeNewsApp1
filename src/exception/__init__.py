# importing libraries
import sys
import types
import warnings
import os
# Get the absolute path of the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the project root to the system path
sys.path.insert(0, project_root)

# Now, your import will work
from src.logger import logging
from src.logger import logging
# warnings
warnings.filterwarnings("ignore")

# custom error message function
def error_message_details(error: Exception, error_details: types.ModuleType) -> str:
    """
    Returns a formatted error message with file name and line number.
    """
    _, _, tb = sys.exc_info()
    if tb is None:
        return f"An unknown error occurred: {str(error)}"

    file_name = tb.tb_frame.f_code.co_filename
    line_number = tb.tb_lineno
    return (
        f"Error occurred in script [{file_name}] "
        f"at line number [{line_number}] "
        f"with message [{str(error)}]"
    )
    
class CustomException(Exception):
    """
    Custom exception class that logs the error when raised.
    """

    def __init__(self, error: Exception, error_details: types.ModuleType):
        super().__init__(str(error))
        self.error_message = error_message_details(error, error_details)

        # Log the error automatically
        logging.error(self.error_message)

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 0  # Intentional error for testing
    except Exception as e:
        logging.info("An error occurred in the main block.")
        raise CustomException(e, sys)
