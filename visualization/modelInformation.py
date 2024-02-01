import gradio as gr
import pandas as pd
from modelFactory import get_all_models

models = get_all_models()

# -----------MODEL INFORMATION -------------#
with gr.Blocks() as modelInfo:

    # combine all model metrics into dataframe
    list_of_dicts = [model.metrics.get_metrics_dict() for model in models]
    #set index to model name
    for i in range(len(list_of_dicts)):
        list_of_dicts[i]['Model'] = models[i].name

    df = pd.DataFrame(list_of_dicts)
    df.set_index('Model', inplace=True)

    # Applying style to highlight the maximum value in each column for iou dice and accuracy
    styler = df.style.apply(lambda x: ["background: lightgreen" if v == x.max() else "" for v in x], axis=0,
                            subset=['Dice score', 'IoU score', 'Accuracy'])

    # Convert the styled dataframe to HTML
    styled_html = styler.to_html()
    styled_html = styled_html.replace('<table id=', '<table style="width:100%;" id=')

    # Markdown text for header and description
    markdown_text = "# Models \n" \
                    "In the table below one can see the metrics of the different models, calculated on the validation dataset with 190 different images of size 1000x100px." \
                    "The Metrics were only calculated once after all training and hyperparametertuning was final." \

    # Create Gradio components
    markdown_component = gr.Markdown(markdown_text)
    styled_table = gr.HTML(styled_html)

    # dataset markdown text
    dataset_markdown_text = "# Dataset \n" \
                            "The dataset used for training and validation was manually label and extracted from the swisstopo dataset." \
                            "Overall approx 3.000 cars where marked in the images by using labelme." \

    dataset_markdown_component = gr.Markdown(dataset_markdown_text)
# ------------------------------------------#