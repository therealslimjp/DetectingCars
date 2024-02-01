# DetectingCars

## Introduction
Our project aims to detect cars in aerial images, utilizing various machine learning models to identify streets and cars within spatial data. 
Some envisioned use cases for our application include:
1. Early detection of traffic jams. 
2. Analysis of road utilization to identify streets with expansion potential due to high utilization.

## Process
We structured our project into three main parts:
1. Preprocessing
2. Modelling
3. Visualization

### Preprocessing
Within this section, we offer code for data procurement and preprocessing. 
We obtained images from swisstopo and generated masks delineating cars and streets using labelme. 
Additionally, to evaluate model performance on external data, we acquired images from the University of Twente. 
Detailed preprocessing steps are outlined in the README.md within the preprocessing folder.

### Modelling
This section includes code for model development. 
We created multiple models for our project: two for cars and one for streets. 
Our approach aimed to evaluate model performance when trained for specific tasks and when integrated. 
Detailed modelling steps are provided in the README.md within the modelling folder.

### Visualization
Here, we provide code for visualization. We utilized gradio to develop an app that integrates all project components. 
Detailed visualization steps are outlined in the README.md within the visualization folder.

**Due to large files the models and images are not included in the vizualizations those have to be downloaded from the following kaggle datasets and notebooks:** 
- https://www.kaggle.com/datasets/jonas06/tds-models
- https://www.kaggle.com/datasets/jonas06/sample-data
- https://www.kaggle.com/datasets/jonas06/car-street-data
- https://www.kaggle.com/datasets/jonas06/tds-cars-twente

- https://www.kaggle.com/benediktvoss/fastai-carsegmentation-filtereddata
- https://www.kaggle.com/benediktvoss/transformercarsegmentation
- https://www.kaggle.com/jonas06/street-modell


A Zipfolder that can be run without any modifications can be found here: https://drive.google.com/file/d/1QVlmwP5-4zZNELSPIBckwus6WBnavkZg/view?usp=sharing
