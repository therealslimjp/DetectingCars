import random

import PIL.Image
import numpy as np
import gradio as gr
from fastai.vision.all import *

# define a function to get the images
def get_images(name):
    return get_image_files("images/")
# define a function to get the numpy mask for the given path
def get_mask(path):
    #remove the file name and extension from the path
    path = path.parent
    # add the name mask.npy to the path
    path = path.joinpath("label.npy")
    return np.load(path)
class Model:

    def __init__(self, path, is_transformer=False):
        self.is_transformer = is_transformer
        if is_transformer:
            self.model = Model._get_model_transformer(path)

        else:
            data_block = DataBlock(blocks=(ImageBlock, MaskBlock(codes=["0", "1"])),
                                   splitter=GrandparentSplitter(train_name='train', valid_name='test'),
                                   get_items=get_images,
                                   get_y=get_mask,
                                   batch_tfms=aug_transforms(size=500, max_lighting=0.3))
            dataloader = data_block.dataloaders("./", bs=4)
            learner = unet_learner(dataloader, resnet34, metrics=Dice)
            learner = learner.load(path)
            self.model= learner



    def predict(self, image):
        if self.is_transformer:
            return self.model.predict(image)[0]
        return self.model.predict(image)[0].numpy()


model_car = None


def model1(image: PIL.Image) -> PIL.Image:
    return image


def model2(image: PIL.Image) -> PIL.Image:
    return image


def model3(image: PIL.Image) -> PIL.Image:
    return image


def process_coordinates(x: int, y: int, model:int=0) -> PIL.Image:
    width, height = 400, 300
    color = random.choice(['red', 'green', 'blue', 'yellow', 'black'])
    image = PIL.Image.new('RGB', (width, height), color)
    draw = image.resize((500, 500))
    return draw

def process_coordinates_with_all_models_at_once(x: int, y: int) -> PIL.Image:
    width, height = 400, 300
    color = random.choice(['red', 'green', 'blue', 'yellow', 'black'])
    image = PIL.Image.new('RGB', (width, height), color)
    draw = image.resize((500, 500))
    return [draw,draw,draw]


def identifyCars(image: np.ndarray, model: int = 0) -> list:
    image= process_image(image, model)
    return image



def process_image(image_array: np.ndarray, model: int = 0) -> PIL.Image:
    print(model)

    if image_array is None:
        raise gr.Error("No Image Uploaded!")
    try:
        # Convert the numpy array to a PIL Image
        image = PIL.Image.fromarray(image_array)
        # resize it to 1000x1000
        image = image.resize((1000, 1000))
        if model == 0:
            image_np = model_car.predict(image)
        elif model == 1:
            image_np = model_transformer_car.predict(image)[0].numpy()
        elif model == 2:
            image_np = model_street.predict(image)[0].numpy()
        image = PIL.Image.fromarray(image_np)
        # Resize the image to 500x500
        resized_image = image.resize((500, 500))
    except Exception:
        raise gr.Error("Error processing!")
    return resized_image


def process_image_with_all_models_at_once(image_array: np.ndarray) -> [PIL.Image, PIL.Image, PIL.Image]:
    if image_array is None:
        raise gr.Error("No Image Uploaded!")
    try:
        results = []
        global model_car
        for model in [model_car]:
            image = PIL.Image.fromarray(image_array)
            # resize it to 1000x1000
            image = image.resize((1000, 1000))
            image_arr = model.predict(image)[0].numpy()
            image = PIL.Image.fromarray(image_arr)
            # Resize the image to 500x500
            resized_image = image.resize((500, 500))
            results.append(resized_image)
    except Exception:
        raise gr.Error("Error processing!")
    return results
