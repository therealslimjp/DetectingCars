
import gradio as gr


from visualization.datasetSample import datasetSample
from visualization.modelComparison import modelCompare
from visualization.modelInformation import modelInfo


def mock_detect_streets(image):
    # Simulate street detection by adding a red overlay
    red_overlay = np.full_like(image, (0, 0, 255), dtype=np.uint8)
    combined_image = cv2.addWeighted(image, 0.7, red_overlay, 0.3, 0)
    return combined_image


def mock_detect_cars(image):
    # Simulate car detection by adding green dots
    car_count = 0
    for x in range(50, 450, 100):
        for y in range(50, 450, 100):
            cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
            car_count += 1
    return image, car_count


def process_image(image):
    streets_image = mock_detect_streets(image)
    cars_image, car_count = mock_detect_cars(np.copy(streets_image))
    return image, cars_image, image


iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type='numpy'),
    outputs=[
        gr.Image(type="numpy", label="Original Image"),
        gr.Image(type="numpy", label="Processed Image"),
        gr.Textbox(label="Detection Info")
    ],
    title="Car Detection in Satellite Images",
    description="Upload a satellite image to detect cars. The app first highlights streets and then detects cars on those streets.",
)

import gradio as gr
import numpy as np
import cv2
import pandas as pd


def process_image_single_steps(image):
    streets_image = mock_detect_streets(image)
    cars_image, _ = mock_detect_cars(np.copy(streets_image))
    return streets_image, cars_image


# Mock data for tables
def create_mock_table():
    data = {"Car ID": [1, 2, 3], "Distance to Next Car": ["5m", "10m", "15m"]}
    return pd.DataFrame(data)


iface_tab1 = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type='numpy'),
    outputs=[
        gr.Image(type="numpy", label="Original Image"),
        gr.Image(type="numpy", label="Processed Image"),
        gr.Dataframe(label="Detection Info")
    ],
    title="Detect Cars - Combi",
    description="Upload an image to detect cars. Processed image will be displayed along with a table of detected cars."
)

iface_tab2 = gr.Interface(
    fn=process_image_single_steps,
    inputs=gr.Image(type='numpy'),
    outputs=[
        gr.Image(type="numpy", label="Streets Image"),
        gr.Image(type="numpy", label="Cars Image")
    ],
    title="SingleStepsModel",
    description="View intermediate steps in the car detection model."
)




# Create the tabbed interface
tabbed_interface = gr.TabbedInterface(
    [modelCompare, modelInfo, datasetSample],
    tab_names=["Try different Models", "Model Information", "Dataset Sample"]
)

if __name__ == '__main__':
    tabbed_interface.launch()
