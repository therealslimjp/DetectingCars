# DetectingCars

## Introduction
In our project we want to detect cars in aerial images.
We decided to train different Machine-Learning models that help in identifying streets and cars in spatial data.
A few use cased we thought of when using our application were:
1. Early traffic jam detection
2. Analysis of road utilization. Therefore detecting streets that have expansion potential cause of high utilization.

## Process
We decided to split our project into 3 main parts:
1. Preprocessing
2. Modelling
3. Visualization

### Preprocessing
In this folder, we provide the code for data procurement and preprocessing. We downloaded images from swisstopo and created masks with cars and streets via labelme.
To evaluate the performance of our model on external data, we also downloaded images from the University of Twente.
The preprocessing steps in detail are described in the README.md in the preprocessing folder.

### Modelling
In this folder, we provide the code for the modelling. For our project, we developed multiple models: two for cars and one for streets. 
The rationale was to assess model performance when trained on specific tasks and when combined. 
The modelling steps in detail are described in the README.md in the modelling folder.

### Visualization
In this folder, we provide the code for the visualization. We used gradio to create an app that integrates all components into one.
The visualization steps in detail are described in the README.md in the visualization folder.
