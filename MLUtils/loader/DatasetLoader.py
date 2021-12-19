import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np

from MLUtils.loader.ImageProcessor import ImageProcessor


class DatasetLoader:
    def __init__(self, resize_height, resize_width):
        # store the image preprocessor
        self.image_processor = ImageProcessor(resize_height, resize_width)

    # Only for datasets that fit in ram
    def load_fit_in_ram_dataset(self, image_paths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, image_path) in enumerate(image_paths):
            # load the image and extract the class label assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-2]

            # Process the image
            image = self.image_processor.process_image(image)

            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)

            # show an update every `verbose` images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(image_paths)))

        # return a tuple of the data and labels
        return np.array(data), np.array(labels)

    def train_test_split(self, data, labels, test_size, random_state):
        return train_test_split(data, labels, test_size=test_size, random_state=random_state)

    def convert_labels_to_integer(self, labels):
        label_encoder = LabelEncoder()
        return label_encoder.fit_transform(labels), label_encoder.classes_
