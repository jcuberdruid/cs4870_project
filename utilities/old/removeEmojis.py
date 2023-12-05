import os
import emoji

def remove_emojis(text):
    return ''.join(c for c in text if c not in emoji.UNICODE_EMOJI['en'])

def main():
    root_dir = os.getcwd()  # Current directory
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            clean_name = remove_emojis(filename)
            if filename != clean_name:
                original_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, clean_name)
                os.rename(original_path, new_path)
                print(f"Renamed {filename} to {clean_name}")

if __name__ == "__main__":
    main()

