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


def inference_coords(models: [] , lat:float, lon:float) -> [PIL.Image]:
    pil_image = get_image_for_coords(lat, lon)

    #check if image is None
    if pil_image is None:
        raise gr.Error("No Image found for coords!")
    #convert to numpy array
    image_array = np.array(pil_image)

    #process image
    return process_image(image_array, models)


def process_image_annotated(image_array: np.ndarray) -> [PIL.Image]:
    global models

    if image_array is None:
        raise gr.Error("No Image Uploaded!")
    try:
        # Convert the numpy array to a PIL Image
        image = PIL.Image.fromarray(image_array)

        prediction_image_car, prediction_mask_car, preditcion_np_car = models[0].segment_image(image)
        prediction_image_street, prediction_mask_street, preditcion_np_street = models[2].segment_image(image)
        #resize image
        image = image.resize((500, 500))

        sections = [(preditcion_np_car, "car"), (preditcion_np_street, "street")]
        return (image, sections)

    except Exception as e:
        raise gr.Error("Error processing!")





def combined_model_image(image_array: np.ndarray) -> PIL.Image:
    global models

    if image_array is None:
        raise gr.Error("No Image Uploaded!")
    try:
        results = []
        # Convert the numpy array to a PIL Image
        image = PIL.Image.fromarray(image_array)

        prediction_image_car, prediction_mask_car, preditcion_np_car = models[0].segment_image(image)
        prediction_image_street, prediction_mask_street, preditcion_np_street = models[2].segment_image(image)

        return combine_two_masks(image, prediction_mask_car, prediction_mask_street)

    except Exception as e:
        raise gr.Error("Error processing!")


def combined_model_coords(lat:float, lon:float) -> PIL.Image:
    global models

    image = get_image_for_coords(lat, lon)

    #check if image is None
    if image is None:
        raise gr.Error("No Image found for coords!")
    try:
        results = []

        prediction_image_car, prediction_mask_car, preditcion_np_car = models[0].segment_image(image)
        prediction_image_street, prediction_mask_street, preditcion_np_street = models[2].segment_image(image)

        return combine_two_masks(image, prediction_mask_car, prediction_mask_street)

    except Exception as e:
        raise gr.Error("Error processing!")


def combine_two_masks(image, preditcion_mask_1, prediction_mask2):

    combined_mask = PIL.Image.blend(preditcion_mask_1, prediction_mask2, alpha=0.5)

    image_resized = image.resize((500, 500)).convert("RGBA")

    combined_image = PIL.Image.blend(image_resized, combined_mask, alpha=0.5)

    return combined_image


def get_loaded_models():
    global models
    return models