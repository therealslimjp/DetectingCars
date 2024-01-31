import PIL.Image
import numpy as np
from fastai.vision.all import *
from abc import ABC, abstractmethod
import torch
from torch import nn
from transformers import TrainingArguments, Trainer , SegformerForSemanticSegmentation, AutoImageProcessor
import albumentations as A

image_folder_path = "./images"

# define helper methods:
def find_files_by_pattern(folder, pattern):
    matching_files = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(pattern):
                matching_files.append(root + "/" + file)

    return matching_files


def get_images_all(name):
    global image_folder_path
    image_files = []
    for file in find_files_by_pattern(image_folder_path, "img.png"):
        image_files.append(Path(file))
    return image_files


# define a function to get the numpy mask for the given path
def get_mask(path):
    #remove the file name and extension from the path
    path = path.parent
    # add the name mask.npy to the path
    path = path.joinpath("label.npy")
    return np.load(path)


# Define an abstract base class for image segmentation models
class SegmentationModel(ABC):

    def __init__(self, name, metrics, color = (255, 255, 255, 255)):
        self.name = name
        self.metrics = metrics
        self.color = color

    @abstractmethod
    def segment_image(self, image: PIL.Image):
        """
        Interface method to perform image segmentation.
        """
        pass


# Implementation of the fastai unet learner
class FastaiUnet(SegmentationModel):

    def __init__(self, name, metrics, path, color = (255, 255, 255, 255)):
        super().__init__(name, metrics, color)

        block = DataBlock(blocks=(ImageBlock, MaskBlock(codes=["nothing", "something"])),
                                         splitter=GrandparentSplitter(train_name='train', valid_name='test'),
                                         get_items=get_images_all,
                                         get_y=get_mask,
                                         batch_tfms=aug_transforms(size=500, max_lighting=0.3))

        dataloader = block.dataloaders("./", bs=1)
        self.model = unet_learner(dataloader, resnet34, metrics=Dice)

        # load the model
        self.model = self.model.load(path)

    def segment_image(self, image):
        prediction_array = self.model.predict(image)[0].numpy()

        combined_image, prediction_mask = combine_image_and_mask(image, prediction_array, self.color)

        return combined_image, prediction_mask, prediction_array


class TransformerModel(SegmentationModel):
    def __init__(self, name, metrics, path, color = (255, 255, 255, 255)):
        super().__init__(name, metrics, color)
        # create Image processor
        self.input_transform = A.Compose([
            A.Resize(height=500, width=500)
        ])
        # load the model
        self.model = SegformerForSemanticSegmentation.from_pretrained(path)

    def segment_image(self, image):
        image_array = np.array(image)
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available, otherwise use a CPU
        encoding = self.input_transform(image=image_array)

        pixel_values = (torch.tensor(encoding["image"], dtype=torch.float32).permute(2, 0, 1) / 255).unsqueeze(0)

        outputs = self.model(pixel_values=pixel_values)

        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        prediction_array = upsampled_logits.argmax(dim=1)[0].numpy()

        combined_image, prediction_mask = combine_image_and_mask(image, prediction_array, self.color)

        return combined_image, prediction_mask, prediction_array


# helper function to combine prediction array and image
def combine_image_and_mask(image, prediction_array, color = (255, 255, 255, 255)):
    prediction_mask = PIL.Image.fromarray(prediction_array.astype(np.uint8) * 255).convert("RGBA").resize((500, 500))

    # change the color from the white pixels to purple
    prediction_mask.putdata(
        [color if pixel == (255, 255, 255, 255) else pixel for pixel in prediction_mask.getdata()])

    # Resize the image to size of mask
    image_resized = image.resize((500, 500)).convert("RGBA")

    # Create a new image by blending the resized original image and the mask
    combined_image = PIL.Image.blend(image_resized, prediction_mask, alpha=0.4)

    return combined_image, prediction_mask


def get_all_models():
    models = []
    print("Loading FastAI Model")
    models.append(FastaiUnet("FastAi Car Detection", {'Dice': 0.77}, "best_model_car", color= (200, 0, 200, 100)))
    print("Loading Transformer Model")
    models.append(TransformerModel("Transformer Car Detection", {'Dice': 0.76}, "./models/TransformerCarModel", color=(200, 0, 200, 100)))
    print("Loading FastAI Street Model")
    models.append(FastaiUnet("FastAi Street Detection", {'Dice': 0.9}, "best_model_street", color= (0, 200, 200, 100)))
    return models




