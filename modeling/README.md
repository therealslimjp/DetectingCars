# Modelling

We opted to develop multiple models for our project, resulting in the creation of two models for cars and one for streets. 
The rationale behind this decision was to assess whether the model performs better when trained for specific tasks and to evaluate its performance when both models are integrated.

## Cars
We trained two models for cars:

1. We utilized a FastAI Model trained on our dataset for 40 Epochs, with 20 Epochs frozen and 20 unfrozen. 
The model employed a Resnet 34 backbone, achieving a Dice Score of 0.729. (Notebook: "FastAI_CarSegmentation_FilteredData.ipynb")
2. To explore potential improvements, we experimented with a Transformer Model from Huggingface, yielding a Dice Score of 0.794. (Notebook: "Transformer_Car_Segmentation_b5.ipynb")

Additionally, we assessed the performance of our FastAi Model on external data from the University of Twente Dataset to gauge its performance on unseen data (Notebook: "test-twente-our-data.ipynb").
Furthermore, we enhanced the model's capabilities to identify cars on the street and compute the area of street coverage by cars. 
Our approach involves labeling each car with a unique identifier and calculating both the area of the street covered by cars and the distances between neighboring cars. (Notebook: "CarsCounter.ipynb")

## Streets
For streets, we trained one model:

1. Initially, we attempted to leverage OSMX Street Masks for our model but found their accuracy insufficient, resulting in poor performance. 
Subsequently, we trained a new model using our own masks, employing a FastAi Model trained on our dataset for 40 Epochs, with 20 Epochs frozen and 20 unfrozen, utilizing a Resnet 34 backbone. 
This approach yielded a Dice Score of 0.796. Despite experimenting with various hyperparameters, we ultimately opted for consistency with the car model due to performance and time constraints. (Notebook: "street_-modell.ipynb")

## Combined
We also explored combining both models to potentially enhance performance, leveraging the noise reduction from the street model on the car model. 
Initially, we utilized the street model to detect streets and subsequently employed the car model to identify cars on the streets. 
The process involves passing images through the street model, blacking out non-street areas, and then using the car model to detect cars in the modified images. 
For more detailed information, refer to the notebook "CombinedStreetandCars.ipynb".