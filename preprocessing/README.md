# Preprocessing

In this folder, we provide the code for preprocessing the data. The preprocessing steps are as follows:
## Cars:
1. at first we downloaded the images from swwisstopo via a Download script
2. then we created the masks manually with labelme and labelbox. We therefore marked every car in the images in a polygon.
3. The labelprocess led us to JSON files that held information about the masks and the image. Thats why we then created the dataset with these JSONS. 
Therefore we splitted the images and masks into 1000x1000 images. After that we created a train and test and validation split. (This happened in the Notebooks:)
4. Because we also wanted to see how our model performs on external data we used a external Dataset from the University of Twente. Since the information about the masks
was stored in a different way we had to preprocess the data differently. We therefore used the notebook: "Twente_cars.ipynb" to create the masks in our format. 
After that the masks and images were also splitted into 1000x1000 images in the Nootebook: "splitting_twente.ipynb".

## Street:
1. We first trued to use the street masks from OSM for our Modell (Notebook: "Street_masks.ipynb" loads the masks from osmx into our dataset). 
But we found out that the masks were not accurate enough. Therefore we decided to create our own masks.
2. We then created the masks manually with labelme. We therefore marked every street in the images in a polygon.
3. Similar to 3. in Cars we then created the dataset with these JSONS.

Because we finally had two different datasets one for streets and one for cars we had to merge them. We therefore used the notebook: "Merge_Data.ipynb" to merge the datasets.
