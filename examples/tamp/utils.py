import os
from PIL import Image

def find_next_filename(dir_path: str):
    """
    Saves the given image in the specified directory using the next available filename
    in the format 'frame_<number>.png'.

    :param image: PIL Image object to save
    :param dir_path: Path to the directory where the image should be saved
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    existing_numbers = []
    for filename in os.listdir(dir_path):
        if filename.startswith("frame_") and filename.endswith(".png"):
            try:
                number = int(filename[len("frame_"):-len(".png")])
                existing_numbers.append(number)
            except ValueError:
                continue

    next_number = 1
    if existing_numbers:
        next_number = max(existing_numbers) + 1

    new_filename = f"frame_{next_number}.png"
    return new_filename