# Preprocessing

Within this folder, we offer the code for data preprocessing, which encompasses the following steps:
## Cars:
1. Initially, we downloaded the images from swisstopo using a download script. 
2. Subsequently, we manually created masks using labelme and labelbox, marking each car in the images within a polygon. 
We then utilized the notebook "integration-tds.ipynb" to incorporate the labelbox masks into our dataset. 
3. The labeling process generated JSON files containing information about the masks and the images. 
Consequently, we constructed the dataset using these JSON files, dividing the images and masks into 1000x1000 segments. 
We further performed a split into train, test, and validation sets, a process documented in the notebooks "split_in_1000x1000.ipynb" and "train_test_split.py". 
4. In addition to our internal dataset, we evaluated the performance of our model on external data obtained from the University of Twente. 
Due to differences in mask storage, we processed the Twente dataset differently, utilizing the notebook "Twente_cars.ipynb" to convert the masks into our desired format. 
Following this conversion, we segmented the masks and images into 1000x1000 segments using the notebook "splitting_twente.ipynb".

## Street:
1. Initially, we attempted to utilize street masks from OSM for our model, as depicted in the notebook "Street_masks.ipynb", which loads the masks from OSMX into our dataset. 
However, we discovered that these masks lacked the necessary accuracy, prompting us to create our own masks. 
2. Subsequently, we manually created masks using labelme, marking each street in the images within a polygon. 
3. Similar to the process for cars, we generated the dataset using JSON files.

Having obtained separate datasets for streets and cars, we needed to merge them. This task was accomplished using the notebook "Merge_Data.ipynb".