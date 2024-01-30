import gradio as gr


from visualization.datasetSample import datasetSample
from visualization.identifyCars import identifyAllCars
from visualization.modelComparison import modelCompare
from visualization.modelInformation import modelInfo




# Create the tabbed interface
tabbed_interface = gr.TabbedInterface(
    [modelCompare, identifyAllCars, modelInfo, datasetSample],
    tab_names=["Try different Models", "Identify all Cars", "Model Information", "Dataset Sample"]
)

if __name__ == '__main__':
    tabbed_interface.launch()
