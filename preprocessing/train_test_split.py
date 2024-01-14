#split train test set
import os
import random
import shutil
from pathlib import Path


def move_folders_randomly(source_folder, destination_folder, ratios=(0.7, 0.2, 0.1)):
    # Get the list of subfolders
    subfolders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]

#Calculate the number of folders for each split
    total_folders = len(subfolders)
    train_count = int(ratios[0] * total_folders)
    test_count = int(ratios[1] * total_folders)
    validation_count = total_folders - train_count - test_count

#Randomly shuffle the subfolders
    random.shuffle(subfolders)

#Create destination folders if they don't exist
    for split, count in zip(["train", "test", "validation"], [train_count, test_count, validation_count]):
        split_path = os.path.join(destination_folder, split)
        Path(split_path).mkdir(parents=True, exist_ok=True)

#Move folders to the destination
        for folder in subfolders[:count]:
            source_path = os.path.join(source_folder, folder)
            destination_path = os.path.join(split_path, folder)
            shutil.move(source_path, destination_path)

#Remove moved folders from the list
        subfolders = subfolders[count:]


if __name__ == '__main__':
    move_folders_randomly("labelled_images_splitted", "train_test_split")