import gradio as gr
import pandas as pd+
from modelFactory import get_all_models

models = get_all_models()

# -----------MODEL INFORMATION -------------#
with gr.Blocks() as modelInfo:
    global models
    # combine all model metrics into dataframe
    df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
    for model in models:
        metrci_dict = model.metric.get_metrics_dict()


    # Applying style to highlight the maximum value in each row
    styler = df.style.highlight_max(color='lightgreen', axis=0)

    # Convert the styled dataframe to HTML
    styled_html = styler.to_html()
    styled_html = styled_html.replace('<table id=', '<table style="width:100%;" id=')

    # Markdown text for header and description
    markdown_text = "# Model Information\nHere is the description of what this table shows."

    # Create Gradio components
    markdown_component = gr.Markdown(markdown_text)
    styled_table = gr.HTML(styled_html)
# ------------------------------------------#