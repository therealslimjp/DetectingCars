import gradio as gr


from datasetSample import datasetSample
from identifyCars import identifyAllCars
from modelComparison import modelCompare
from modelInformation import modelInfo




# Create the tabbed interface
tabbed_interface = gr.TabbedInterface(
    [modelCompare, identifyAllCars, modelInfo, datasetSample],
    tab_names=["Try different Models", "Identify all Cars", "Model Information", "Dataset Sample"]
)

if __name__ == '__main__':
    print("Starting Gradio Server...")
    tabbed_interface.launch(server_name="0.0.0.0", server_port=7860)
