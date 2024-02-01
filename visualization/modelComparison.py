import gradio as gr
import pandas as pd
from modelFunctions import *
from CoordImageLoader import *
from datasetSample import *


with gr.Blocks() as modelCompare:

        models = get_loaded_models()
        #get list of model names
        model_names = [model.name for model in models]
        selected_models = [model_names[0]]

        with gr.Row():
            model_choice = gr.CheckboxGroup(model_names, value=model_names[0], type="index", label="Select Models",
                                            interactive=True)
            selected_models = model_choice.value
        with gr.Row():
            with gr.Column():
                with gr.Tab("Select Coordinates"):
                    with gr.Row():
                        lat_coordinates = gr.Number(value=47.53224, label="Latitude Coordinate", interactive=True, step=0.0001, precision=5)
                        lon_coordinates = gr.Number(value=8.79798, label="Logitude Coordinate", interactive=True, step=0.0001, precision=5)
                        btn = gr.Button(value="Reload Map")
                    map = gr.Plot()
                    modelCompare.load(fn=filter_map, inputs=[lat_coordinates, lon_coordinates], outputs=map)
                    btn.click(fn=filter_map, inputs=[lat_coordinates, lon_coordinates], outputs=map)
                    detect_from_coords_button = gr.Button("Run Inference for map")
                with gr.Tab("Upload Image"):
                    with gr.Row():
                        load_sample_image_button = gr.Button("Load sample image")
                        clear_image_button = gr.Button("Reset Image")

                    input_image = gr.Image(type="numpy", label="Upload Image, best with 1000*1000px")

                    with gr.Row():
                        detect_from_image_button = gr.Button("Run Inference for Image")

            with gr.Column():
                outputs = gr.Gallery(type="numpy", label="Output Images", show_label=False, columns=1)
                clear_output_image_button = gr.Button("Discard Results")

            clear_output_image_button.click(fn=lambda: [], inputs=None, outputs=outputs)
            clear_image_button.click(fn=lambda: None, inputs=None, outputs=input_image)

            load_sample_image_button.click(fn=get_sample_image, inputs=[], outputs=input_image)

            detect_from_image_button.click(
                fn=process_image,
                inputs=[input_image, model_choice],
                outputs=outputs
            )

            detect_from_coords_button.click(
                fn=inference_coords,
                inputs=[model_choice, lat_coordinates, lon_coordinates],
                outputs=outputs
            )
