import os
import pandas as pd
import argparse
import glob



def check_write_permission(output_path: str) -> bool:
    """
    Checks whether the given output_path has write permission.

    This function is useful to avoid running a long job that might fail
    because it cannot write to the specified output file (e.g., if the
    file is open in another application like Excel, or if there are
    directory permission issues).

    Args:
        output_path (str): The full path to the file you intend to write.

    Returns:
        bool: True if the path is writable, False otherwise.
    """
    output_dir = os.path.dirname(output_path)
    if not output_dir:
        output_dir = os.getcwd()

    if not os.path.exists(output_dir):
        print(f"Error: Directory '{output_dir}' does not exist.")
        return False

    if not os.path.isdir(output_dir):
        print(f"Error: '{output_dir}' is not a directory.")
        return False

    if not os.access(output_dir, os.W_OK):
        print(f"Error: Directory '{output_dir}' is not writable. Check permissions.")
        return False

    if os.path.exists(output_path):
        # If the file exists, check if it's writable
        if not os.path.isfile(output_path):
            print(f"Error: '{output_path}' exists but is not a regular file.")
            return False
        if not os.access(output_path, os.W_OK):
            print(f"Error: File '{output_path}' exists but is not writable. "
                  "It might be open in another application or you lack permissions.")
            return False
        else:
            print(f"Success: File '{output_path}' exists and is writable (can be overwritten).")
            return True
    else:
        print(f"Success: Directory '{output_dir}' is writable, and '{output_path}' does not exist. "
              "A new file can be created.")
        return True
