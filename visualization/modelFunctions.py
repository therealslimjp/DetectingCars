import PIL.Image
import numpy as np
import gradio as gr
from fastai.vision.all import *
from modelFactory import get_all_models
from CoordImageLoader import get_image_for_coords

models = get_all_models()

def identifyCars(image: np.ndarray, model: int = 0) -> list:
    image= process_image(image, model)
    return image


def process_image(image_array: np.ndarray, model: int = 0) -> PIL.Image:
    global models

    if image_array is None:
        raise gr.Error("No Image Uploaded!")
    try:
        # Convert the numpy array to a PIL Image
        image = PIL.Image.fromarray(image_array)

        prediction_image, prediction_mask, preditcion_np = models[model].segment_image(image)

    except Exception as e:
        raise gr.Error("Error processing!")
    return prediction_image


def process_image_with_all_models_at_once(image_array: np.ndarray) -> [PIL.Image]:
    global models

    if image_array is None:
        raise gr.Error("No Image Uploaded!")
    try:
        results = []
        # Convert the numpy array to a PIL Image
        image = PIL.Image.fromarray(image_array)

        for model in models:
            prediction_image, prediction_mask, preditcion_np = model.segment_image(image)
            results.append(prediction_image)

    except Exception:
        raise gr.Error("Error processing!")
    return results


def inference_coords_single(model: int , lat:float, lon:float) -> PIL.Image:
    pil_image = get_image_for_coords(lat, lon)

    #check if image is None
    if pil_image is None:
        raise gr.Error("No Image found for coords!")
    #convert to numpy array
    image_array = np.array(pil_image)

    #process image
    return process_image(image_array, model)


def inference_coords_all(lat:float, lon:float) -> PIL.Image:
    pil_image = get_image_for_coords(lat, lon)

    # check if image is None
    if pil_image is None:
        raise gr.Error("No Image found for coords!")

    # convert to numpy array
    image_array = np.array(pil_image)

    #process image
    return process_image_with_all_models_at_once(image_array)


def get_loaded_models():
    global models
    return models