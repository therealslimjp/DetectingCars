import gradio as gr
import pandas as pd
from modelFunctions import *
from CoordImageLoader import *

with gr.Blocks() as modelCompare:

        models = get_loaded_models()
        #get list of model names
        model_names = [model.name for model in models]

        with gr.Tab("Model by Model"):
            with gr.Row():
                with gr.Column():
                    model_choice = gr.Radio(model_names, value=model_names[0],type="index", label="Select Model", interactive=True)
                    with gr.Tab("Select Coordinates"):
                        with gr.Row():
                            lat_coordinates = gr.Number(value=47.53224, label="Latitude Coordinate", interactive=True, step=0.0001, precision=5)
                            lon_coordinates = gr.Number(value=8.79798, label="Logitude Coordinate", interactive=True, step=0.0001, precision=5)
                            btn = gr.Button(value="Reload Map")
                        map = gr.Plot()
                        modelCompare.load(fn=filter_map, inputs=[lat_coordinates, lon_coordinates], outputs=map)
                        btn.click(fn=filter_map, inputs=[lat_coordinates, lon_coordinates], outputs=map)
                        detect_from_coords_button = gr.Button("Run Inference for map")
                        clear_coords_button = gr.Button("Reset Coordinates")
                    with gr.Tab("Upload Image"):
                        input_image = gr.Image(type="numpy", label="Upload Image, best with 1000*1000px")
                        detect_from_image_button = gr.Button("Run Inference")
                        clear_image_button = gr.Button("Reset Image")
                with gr.Column():
                    output_image = gr.Image(interactive=False, type="numpy", label="Prediction: ")
                    clear_output_image_button = gr.Button("Discard Image")
                clear_output_image_button.click(fn=lambda: None, inputs=None, outputs=output_image)
                clear_image_button.click(fn=lambda:None, inputs=None, outputs=input_image)
                detect_from_image_button.click(
                    fn=process_image,
                    inputs=[input_image,model_choice],
                    outputs=output_image
                )
                detect_from_coords_button.click(
                    fn=inference_coords_single,
                    inputs=[model_choice, lat_coordinates, lon_coordinates],
                    outputs=output_image
                )
        with gr.Tab("All at once"):
            with gr.Row():
                with gr.Column():
                    with gr.Tab("Select Coordinates"):
                        with gr.Row():
                            lat_coordinates = gr.Number(value=47.53224, label="Latitude Coordinate", interactive=True,
                                                        step=0.0001, precision=5)
                            lon_coordinates = gr.Number(value=8.79798, label="Logitude Coordinate", interactive=True,
                                                        step=0.0001, precision=5)
                            btn = gr.Button(value="Reload Map")
                        map = gr.Plot()
                        modelCompare.load(fn=filter_map, inputs=[lat_coordinates, lon_coordinates], outputs=map)
                        btn.click(fn=filter_map, inputs=[lat_coordinates, lon_coordinates], outputs=map)
                        detect_from_coords_button = gr.Button("Run Inference for map")
                        clear_coords_button = gr.Button("Reset Coordinates")
                    with gr.Tab("Upload Image"):
                        input_image = gr.Image(type="numpy", label="Upload Image, best with resulution of 0,1m per pixel and 1000*1000px")
                        detect_from_image_button = gr.Button("Run inference for all Models")
                        clear_image_button = gr.Button("Reset Image")
                with gr.Column():
                    outputs = []
                    for model in models:
                        outputs.append(gr.Image(interactive=False, type="numpy", label=model.name + " Result:"))

                    clear_output_image_button = gr.Button("Discard Images")
                clear_output_image_button.click(fn=lambda: [None,None,None], inputs=None, outputs=outputs)
                clear_image_button.click(fn=lambda:None, inputs=None, outputs=input_image)
                detect_from_image_button.click(
                    fn=process_image_with_all_models_at_once,
                    inputs=input_image,
                    outputs=outputs
                )
                detect_from_coords_button.click(
                    fn=inference_coords_all,
                    inputs=[lat_coordinates, lon_coordinates],
                    outputs=outputs
                )