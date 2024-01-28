import gradio as gr
import pandas as pd
from modelFunctions import *

with gr.Blocks() as identifyAllCars:
    with gr.Row():
        with gr.Column():
            model_choice=gr.Radio(["Model1", "Model2", "Model3"], value="Model2" ,type="index", label="Select Model", interactive=True)
            input_image = gr.Image(type="numpy", label="Upload Image, best with 1000*1000px")
            detect_from_image_button = gr.Button("Detect Cars")
            clear_image_button = gr.Button("Reset Image")
        with gr.Column():
            output_image = gr.Image(interactive=False, type="numpy", label="Processed Image")
            clear_output_image_button = gr.Button("Discard Image")

            # Markdown text for header and description
            # Create Gradio components
        clear_output_image_button.click(fn=lambda: None, inputs=None, outputs=output_image)
        clear_image_button.click(fn=lambda:None, inputs=None, outputs=input_image)


        detect_from_image_button.click(
            fn=identifyCars,
            inputs=[input_image,model_choice],
            outputs=output_image
        )
