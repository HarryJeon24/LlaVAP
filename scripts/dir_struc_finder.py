import os

def report_directory_structure(directory, indent=0):
    # List all entries in the directory
    try:
        entries = os.listdir(directory)
    except PermissionError:
        print(" " * indent + f"[Permission Denied]: {directory}")
        return

    for entry in entries:
        entry_path = os.path.join(directory, entry)
        if os.path.isdir(entry_path):
            print(" " * indent + f"[DIR] {entry}")
            report_directory_structure(entry_path, indent + 2)
        else:
            print(" " * indent + f"[FILE] {entry}")

# Example usage
if __name__ == "__main__":
    directory_to_explore = "C:/Users/Harry/PycharmProjects/LLA-VAP\datasets\ICC"  # Replace with your directory
    print(f"Directory structure for: {directory_to_explore}")
    report_directory_structure(directory_to_explore)

import os
import wave

from pydub.utils import mediainfo

def get_wav_length(file_path):
    """Get the duration of a .wav file in seconds using pydub."""
    try:
        info = mediainfo(file_path)
        duration = float(info['duration'])
        return duration
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def list_directory_structure(base_path, indent=0):
    """Recursively list the directory structure of the given path."""
    try:
        # Get all files and directories in the current path
        entries = os.listdir(base_path)
        for entry in entries:
            entry_path = os.path.join(base_path, entry)
            # Check if the entry is a .wav file
            if os.path.isfile(entry_path) and entry_path.lower().endswith('.wav'):
                length = get_wav_length(entry_path)
                if length is not None:
                    print("  " * indent + f"- {entry} (Length: {length:.2f} seconds)")
                else:
                    print("  " * indent + f"- {entry} (Invalid .wav file)")
            # If entry is a directory, recursively list its contents
            elif os.path.isdir(entry_path):
                print("  " * indent + f"- {entry}")
                list_directory_structure(entry_path, indent + 1)
            else:
                print("  " * indent + f"- {entry}")
    except PermissionError:
        # Handle permission issues gracefully
        print("  " * indent + "- [Permission Denied]")

def main():
    # Specify the path
    path = r"C:/Users/Harry/PycharmProjects/LLA-VAP/datasets/ICC"
    if os.path.exists(path):
        print(f"Directory structure of '{path}':")
        list_directory_structure(path)
    else:
        print(f"The specified directory does not exist: {path}")

if __name__ == "__main__":
    main()

