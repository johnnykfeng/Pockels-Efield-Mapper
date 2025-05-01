
def remove_extension(file_name):
    return file_name.split(".")[0]

def get_metadata_from_filename(file_name):
    bias = file_name.split("_")[1]
    xray_flux = file_name.split("_")[3]
    # led_flux = file_name.split("_")[5]
    return bias, xray_flux