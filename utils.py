import os
import shutil

def remove_extension(file_name):
    return file_name.split(".")[0]

def get_metadata_from_filename(file_name):
    bias = file_name.split("_")[1]
    xray_flux = file_name.split("_")[3]
    # led_flux = file_name.split("_")[5]
    return bias, xray_flux

def prepend_to_png_filenames(folder_path, prefix):
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .png file
        if filename.lower().endswith('.png'):
            original_path = os.path.join(folder_path, filename)
            new_filename = prefix + filename
            new_path = os.path.join(folder_path, new_filename)

            # Copy the file with the new name
            shutil.copy2(original_path, new_path)
            print(f"Copied: {filename} -> {new_filename}")

def remove_string_from_filenames(folder_path, string_to_remove):
    for filename in os.listdir(folder_path):
        if string_to_remove in filename:
            original_path = os.path.join(folder_path, filename)
            new_filename = filename.replace(string_to_remove, '')
            new_path = os.path.join(folder_path, new_filename)
            shutil.copy2(original_path, new_path)
            print(f"Copied: {filename} -> {new_filename}")

if __name__ == "__main__":
    # Example usage:
    # Replace 'your_folder_path' with the actual folder path
    # Replace 'prefix_' with the string you want to prepend
    folder_path = "R:/Pockels_data/NEXT GEN POCKELS/Photo-Pockels_D420222_2025-06-20/Filter_1pct"
    # prepend_to_png_filenames(folder_path, 'Filter1pct_')
    remove_string_from_filenames(folder_path, '_0p01')

