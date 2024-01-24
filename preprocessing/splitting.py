import os
import re
import time

import numpy as np
from PIL import Image


def split_image(image_path, split_width, split_height):
    # Open the image
    img = Image.open(image_path)

    # Calculate the number of splits
    img_width, img_height = img.size
    rows = img_height // split_height
    cols = img_width // split_width

    # Create a list to hold the split images
    split_images = []
    # image path has a form like this: labelled_images_as_masks/swissimage-dop10_2022_2674-1251_0_1_2056_tif_json/image_0.png. I want to get the two numbers 2674 and 1251
    file_path_coords = re.search(r'(?<=labelled_images_as_masks/swissimage-dop\w\w_\d\d\d\d_)\d{4}-\d{4}', image_path).group(0).split("-")

    # Split the image and add to the list
    for i in range(rows):
        for j in range(cols):
            left = j * split_width
            top = i * split_height
            right = (j + 1) * split_width
            bottom = (i + 1) * split_height

            left_incremented= int(int(file_path_coords[0])*10 + left*0.001)
            bottom_incremented= int(int(file_path_coords[1])*10 - (bottom*0.001))+10

            split_images.append((img.crop((left, top, right, bottom)),f"{left_incremented}_{bottom_incremented}"))

    return split_images

import matplotlib.pyplot as plt
def convert_png_to_array(image):
    # Open the PNG image
    np_array = np.array(image)
    return np_array


# Example usage
# encoded_string = "your_base64_encoded_string_here"
# encoded_chunks = split_encoded_image_to_base64(encoded_string)
# Now, encoded_chunks is a list of base64 encoded strings of 100x100 pixel chunks
if __name__ == '__main__':
    if not os.path.exists("labelled_images_splitted"):
        os.mkdir("labelled_images_splitted")
    for folder in os.listdir("labelled_images_as_masks"):
        if folder.endswith(".json") or folder.endswith(".tif") or folder.endswith(".txt") or folder.endswith(".DS_Store"):
            continue
        if os.path.exists("labelled_images_splitted/" + folder):
            continue
        print(folder)
        for file in os.listdir("labelled_images_as_masks/" + folder):
            if (file.endswith("img.png") or file.endswith("label.png")) and folder.startswith("swissimage-dop"):
                split_images = split_image("labelled_images_as_masks/" + folder + "/" + file, 1000, 1000)
                for i in range(len(split_images)):
                    if not os.path.exists("labelled_images_splitted/"  + str(split_images[i][1])):
                        os.mkdir("labelled_images_splitted/" + str(split_images[i][1]))
                    if file.endswith("img.png"):
                        split_images[i][0].save(
                            "labelled_images_splitted/" +str(split_images[i][1]) + "/" + file[:-4] + ".png")
                    elif file.endswith("label.png"):
                        np_array = convert_png_to_array(split_images[i][0])
                        np.save("labelled_images_splitted/" +str(split_images[i][1]) + "/" + file[:-4] + "_street_mask.npy", np_array)

