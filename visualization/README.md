# Visualization 

This part contains all components used for visualizing our project. It consists 
of the separate pages for the different components as well as scirpts for utility
functions and a factory, where the different models are loaded under one common interface. 
Since importing fastai is a rather complicated process, we decided to create a factory, which incorporates all 
code necessary for loading the model (e.g. dataloader). 

To avoid incompatibility problems, we decided to put the app into a 
docker container. The dockerfile is located in this directory of the project.
Installing requirements takes quite some time.
You can build the docker image by running the following command in the terminal:

```bash
docker build -t tds .
```
and run it with 
```bash
docker run -p 7860:7860 tds
```
The app is, after the models are loaded, accessible under port 7860 on localhost.
You can check if the app is ready by viewing the logs in the container. When all three Models (FastAI Car, Transformer Car, FastAI Street) are loaded,
there will be a statement informing you about that the app is ready and running on the port 7860.


## Requirements
To run the app, you have to do two things:
- Put the models into models/ directory 
- Put the dataset in the correct structure (Sample_Data/ with subfolders Google_Maps,ITC_VD and Swisstopo) into images/ directory. 

Both directories are included in the zip archive we provided.


