import gradio as gr
import pandas as pd
from modelFunctions import *
from CoordImageLoader import *
from datasetSample import *

with gr.Blocks() as modelCombined:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    load_sample_image_button = gr.Button("Load sample image")
                    clear_image_button = gr.Button("Reset Image")
                input_image = gr.Image(type="numpy", label="Upload Image, best with 1000*1000px")
                detect_from_image_button_annotated = gr.Button("Run Inference")
            with gr.Column():
                annotated_image = gr.AnnotatedImage( label="Prediction: ")
                clear_output_image_button = gr.Button("Discard Result")

            clear_output_image_button.click(fn=lambda: [], inputs=None, outputs=annotated_image)
            clear_image_button.click(fn=lambda: None, inputs=None, outputs=input_image)
            load_sample_image_button.click(fn=get_sample_image, inputs=[], outputs=input_image)

            detect_from_image_button_annotated.click(
                fn=process_image_annotated,
                inputs=[input_image],
                outputs=annotated_image
            )


