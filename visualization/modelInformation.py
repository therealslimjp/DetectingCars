import gradio as gr
import pandas as pd


# -----------MODEL INFORMATION -------------#
with gr.Blocks() as modelInfo:
    df = pd.DataFrame({
        "A": [14, 4, 5, 4, 1],
        "B": [5, 2, 54, 3, 2],
        "C": [20, 20, 7, 3, 8],
        "D": [14, 3, 6, 2, 6],
        "E": [23, 45, 64, 32, 23]
    })

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