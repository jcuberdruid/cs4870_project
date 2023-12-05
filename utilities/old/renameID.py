import os
import csv
import uuid

def generate_unique_id():
    """
    Generate a unique alphanumeric ID.
    """
    return uuid.uuid4().hex

def main():
    # Mapping of new ID to original file name
    mapping = {}

    # Walking through the directory tree to find .wav files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.wav'):
                original_path = os.path.join(root, file)
                unique_id = generate_unique_id()
                new_path = os.path.join(root, f"{unique_id}.wav")
                
                os.rename(original_path, new_path)
                
                # Store mapping
                mapping[unique_id] = file

    # Saving the mapping to a CSV file
    with open('file_mapping.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['ID', 'Original Filename'])
        for unique_id, original_filename in mapping.items():
            writer.writerow([unique_id, original_filename])

if __name__ == '__main__':
    main()

