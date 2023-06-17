import os
from pathlib import Path

def modify_file_name(file_path, prefix):
    # Convert the file path to a Path object
    file_path = Path(file_path)
    
    # Extract the directory and the file name
    parent_dir = file_path.parent
    file_name = file_path.name

    # Add the prefix to the file name
    new_file_name = prefix + file_name

    # Create the new file path
    new_file_path = os.path.join(parent_dir, new_file_name)

    return new_file_path
