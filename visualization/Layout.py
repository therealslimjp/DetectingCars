import gradio as gr

from datasetSample import datasetSample
from identifyCars import identifyAllCars
from modelComparison import modelCompare
from modelInformation import modelInfo
from CombinedModels import modelCombined

theme = gr.themes.Monochrome(
    primary_hue="stone",
    neutral_hue="gray",
    radius_size="sm",
    text_size="lg",
)
theme.set(
    button_primary_background_fill='*primary_600',
    button_primary_background_fill_hover='*primary_400'
)

css= """
.caption-label{
    background-color: transparent;
}
.tab-nav button {
    font-size: 1.3rem;
}
"""

with gr.Blocks(theme=theme, css=css) as Layout:
    gr.Markdown("# Car Detection with Deep Learning")

    with gr.Row(): # Tabbed Interface
        tabbed_interface = gr.TabbedInterface(
            [modelCompare, identifyAllCars, modelCombined, modelInfo, datasetSample],
            tab_names=["Try different Models", "Identify all Cars", "Combined Models" , "Model Information", "Dataset Sample"],
        )

    gr.Markdown("### Created By: Jonas Erbacher, Jan-Philip Töpfer and Benedikt Voß")
