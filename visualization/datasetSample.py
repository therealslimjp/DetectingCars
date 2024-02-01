import gradio as gr
import os
import random
import PIL.Image
from fastai.vision.all import *

image_folder = '.\\images'

def find_files_by_pattern(folder, extension):
    matching_files = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            # check if file ends with extension
            if file.endswith(extension):
                matching_files.append(root + "\\" + file)

    return matching_files

def load_random_image():
    images = []

    for _ in range(3):  # Load 3 random images
        random_image = random.choice(find_files_by_pattern(image_folder, ".jpg"))

        img = PIL.Image.open(random_image)
        img = img.resize((256, 256))  # Resize the image
        images.append(img)
    return images


with gr.Blocks() as datasetSample:
    with gr.Row():
        gallery = gr.Gallery(load_random_image, label="Dataset Sample", columns=3)
    button = gr.Button("Load new Images")
    button.click(fn=load_random_image, inputs=[], outputs=gallery)
