import gradio as gr
from Layout import Layout


if __name__ == '__main__':
    print("Starting Gradio Server...")
    Layout.launch(server_name="0.0.0.0", server_port=7860)
