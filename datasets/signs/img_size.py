import os
from PIL import Image

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

def process_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print(f"No image files found in {folder_path}")
        return

    max_width, max_height = float('-inf'), float('-inf')
    min_width, min_height = float('inf'), float('inf')

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        width, height = get_image_dimensions(image_path)

        max_width = max(max_width, width)
        max_height = max(max_height, height)

        min_width = min(min_width, width)
        min_height = min(min_height, height)

    print(f"Folder: {folder_path}")
    print(f"Max Dimensions: {max_width} x {max_height}")
    print(f"Min Dimensions: {min_width} x {min_height}")
    print()

def traverse_folders(root_folder):
    for foldername, subfolders, filenames in os.walk(root_folder):
        process_folder(foldername)

if __name__ == "__main__":
    root_folder = "/home/wtingting/Downloads/traffic_sign/demo/datasets/signs/data_china" 
    traverse_folders(root_folder)
