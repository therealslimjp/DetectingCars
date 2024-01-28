import gradio as gr
import os
import random
import PIL.Image

image_folder = '../modeling/images'
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]


def load_random_image():
    images = []
    for _ in range(3):  # Load 3 random images
        random_image = random.choice(image_files)
        img_path = os.path.join(image_folder, random_image)
        img = PIL.Image.open(img_path)
        img = img.resize((256, 256))  # Resize the image
        images.append(img)
    return images


with gr.Blocks() as datasetSample:
    with gr.Row():
        gallery = gr.Gallery(load_random_image, label="Dataset Sample", columns=3)
    button = gr.Button("Load new Images")
    button.click(fn=load_random_image, inputs=[], outputs=gallery)
