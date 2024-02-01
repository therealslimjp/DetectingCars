# Modelling

We decided to create more than one model for our project. We therefore created two models for cars and a model for streets. 
The thaught behind this was that we wanted to see if the model performs better if it is trained on a specific task. 
We also wanted to see how it performs when we lay both models on top of each other.

## Cars
We trained two models for cars. 
1. We used a FastAi Model that was trained on our Dataset for 40 Epochs. 20 Epochs Frozen and 20 unfrozen. 
We used a Resnet 34 as a backbone. Which lead to a Dice Score of 0,78. (Notebook: "cars_fastai.ipynb")
2. To see if we can improve the model with a different Model we used a Transformer Modell from Huggingface. (Notebook: "cars_transformer.ipynb")

We also tested our FastAi Model on external Data to see how it performs on unseen data. We used the University of Twente Dataset for this. (Notebook: "test-twente-our-data.ipynb")


## Streets
We trained one model for streets.
1. We first tried to use the OSMX Street Masks for our Modell. But we found out that the masks were not accurate enough and the performance was awful (Notebook: "street-modell-osmx.ipynb").
We then trained a new model on our own masks and used a FastAi Model that was trained on our Dataset for 40 Epochs. 20 Epochs Frozen and 20 unfrozen.
We used a Resnet 34 as a backbone. Which lead to a Dice Score of 0,87. We played around with different Hyperparameters here but ultimately 
decided to use the same as for the car model, cause of performance and time reasons. (Notebook: "street_-modell.ipynb")
