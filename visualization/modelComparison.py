import gradio as gr
import pandas as pd
from modelFunctions import *  # Assuming you have defined relevant functions here

with gr.Blocks() as modelCompare:


        with gr.Tab("Model by Model"):
            with gr.Row():
                with gr.Column():
                    model_choice=gr.Radio(["Model1", "Model2", "Model3"], value="Model2" ,type="index", label="Select Model", interactive=True)
                    with gr.Tab("Select Coordinates"):
                        x_coordinates = gr.Number(value=20000, label="X Coordinate", interactive=True)
                        y_coordinates = gr.Number(value=20000, label="Y Coordinate", interactive=True)
                        detect_from_coords_button = gr.Button("Detect Cars")
                        clear_coords_button = gr.Button("Reset Coordinates")
                    with gr.Tab("Upload Image"):
                        input_image = gr.Image(type="numpy", label="Upload Image, best with 1000*1000px")
                        detect_from_image_button = gr.Button("Detect Cars")
                        clear_image_button = gr.Button("Reset Image")
                with gr.Column():
                    output_image = gr.Image(interactive=False, type="numpy", label="Processed Image")
                    clear_output_image_button = gr.Button("Discard Image")
                clear_output_image_button.click(fn=lambda: None, inputs=None, outputs=output_image)
                clear_image_button.click(fn=lambda:None, inputs=None, outputs=input_image)
                detect_from_image_button.click(
                    fn=process_image,
                    inputs=[input_image,model_choice],
                    outputs=output_image
                )
                detect_from_coords_button.click(
                    fn=process_coordinates,
                    inputs=[x_coordinates,y_coordinates,model_choice],
                    outputs=output_image
                )
        with gr.Tab("All at once"):
            with gr.Row():

                with gr.Column():
                    with gr.Tab("Select Coordinates"):
                        x_coordinates = gr.Number(value=20000, label="X Coordinate", interactive=True)
                        y_coordinates = gr.Number(value=20000, label="Y Coordinate", interactive=True)
                        detect_from_coords_button = gr.Button("Detect Cars")
                        clear_coords_button = gr.Button("Reset Coordinates")
                    with gr.Tab("Upload Image"):
                        input_image = gr.Image(type="numpy", label="Upload Image, best with 1000*1000px")
                        detect_from_image_button = gr.Button("Detect Cars")
                        clear_image_button = gr.Button("Reset Image")
                with gr.Column():
                    output_image_model1 = gr.Image(interactive=False, type="numpy", label="Model1 Image")
                    output_image_model2 = gr.Image(interactive=False, type="numpy", label="Model2 Image")
                    output_image_model3 = gr.Image(interactive=False, type="numpy", label="Model3 Image")
                    clear_output_image_button = gr.Button("Discard Images")
                clear_output_image_button.click(fn=lambda: [None,None,None], inputs=None, outputs=[output_image_model1,output_image_model2,output_image_model3])
                clear_image_button.click(fn=lambda:None, inputs=None, outputs=input_image)
                detect_from_image_button.click(
                    fn=process_image_with_all_models_at_once,
                    inputs=input_image,
                    outputs=[output_image_model1,output_image_model2,output_image_model3]
                )
                detect_from_coords_button.click(
                    fn=process_coordinates_with_all_models_at_once,
                    inputs=[x_coordinates,y_coordinates],
                    outputs=[output_image_model1, output_image_model2, output_image_model3]
                )