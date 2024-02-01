# Modelling

We opted to develop multiple models for our project, resulting in the creation of two models for cars and one for streets. 
The rationale behind this decision was to assess whether the model performs better when trained for specific tasks and to evaluate its performance when both models are integrated.

## Cars
We trained two models for cars:

1. We utilized a FastAI Model trained on our dataset for 30 Epochs, with 20 Epochs frozen and 10 unfrozen. 
The model employed a Resnet 34 backbone, achieving a Dice Score of 0.729 on the validation Set. The Model was trained on the entire Dataset in the frozen epochs and was afterwards finetuned on just the images where the mask actually contains a car. (Notebook: "FastAI_CarSegmentation_FilteredData.ipynb")

2. To explore potential improvements, we experimented with a Segformer /Transformer Model from Huggingface, yielding a Dice Score of 0.794. For this Model we tried different DataAugmentation techniques like ColorJitter and Albumentations. 
In the end we selected a combination of different transformations from the Albumentations library where the model was most robust and had a good performance.
The training for this model was performed just on those images containing cars to reduce training time for the large Segformer model.
The initially chosen checkpoint was the [nvidia\mit_b5](https://huggingface.co/nvidia/mit-b5) which is the largest one and results in the best performance. (Notebook: "Transformer_Car_Segmentation_b5.ipynb")

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

## Evaluation

We evaluated all of our model on the same validation split that was just once used after all hyperparametertuning and testing around with different models was done.
To make the metrics more comparable accross the different libraries we defined one calculations for iou, dice and accuracy and implemented those for each model.