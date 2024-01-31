import PIL.Image
import numpy as np
import gradio as gr
from fastai.vision.all import *
from modelFactory import get_all_models
from CoordImageLoader import get_image_for_coords

models = get_all_models()


def identify_cars(image: np.ndarray, model: int = 0) -> list:
    prediction_image, prediction_mask, preditcion_np = models[model].segment_image(image)
    return preditcion_np


def process_image(image_array: np.ndarray, selected_models: []) -> [PIL.Image]:
    global models

    if image_array is None:
        raise gr.Error("No Image Uploaded!")
    try:
        results = []
        # Convert the numpy array to a PIL Image
        image = PIL.Image.fromarray(image_array)

        #filter models with indexes in selected_models
        for model in [models[i] for i in selected_models]:
            prediction_image, prediction_mask, preditcion_np = model.segment_image(image)
            results.append((prediction_image, model.name))

        return results

    except Exception as e:
        raise gr.Error("Error processing!")


def inference_coords(models: [] , lat:float, lon:float) -> PIL.Image:
    pil_image = get_image_for_coords(lat, lon)

    #check if image is None
    if pil_image is None:
        raise gr.Error("No Image found for coords!")
    #convert to numpy array
    image_array = np.array(pil_image)

    #process image
    return process_image(image_array, models)


def get_loaded_models():
    global models
    return models