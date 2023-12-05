import os

def remove_spaces_from_filenames(directory):
    # Get all files in the directory
    for filename in os.listdir(directory):
        # Check if the filename contains spaces
        if ' ' in filename:
            new_filename = filename.replace(' ', '')
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f"Renamed: '{filename}' to '{new_filename}'")

if __name__ == "__main__":
    # Run the function for the current directory
    remove_spaces_from_filenames(os.getcwd())

