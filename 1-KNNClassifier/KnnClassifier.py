import argparse

from sklearn.preprocessing import LabelEncoder
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

# Ask informations about the algo and the dataset
from MLUtils.loader.DatasetLoader import DatasetLoader

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1, help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# Store args info in variables
dataset_path = args["dataset"]
neighbors_number = args["neighbors"]
jobs_number = args["jobs"]

# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset_path))

# initialize the dataset loader, load the dataset from disk, and reshape the data matrix
dataset_loader = DatasetLoader(32, 32)
(data, labels) = dataset_loader.load_fit_in_ram_dataset(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1024.0)))

# encode the labels as integers
labels, classes = dataset_loader.convert_labels_to_integer(labels)

# partition the data into training and testing splits using 75%
# of the data for training and the remaining 25% for testing
testing_dataset_size = 0.25
random_factor = 42
(trainX, testX, trainY, testY) = dataset_loader.train_test_split(data, labels, testing_dataset_size, random_factor)

# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=neighbors_number, n_jobs=jobs_number)
model.fit(trainX, trainY)


print(classification_report(testY, model.predict(testX), target_names=classes))
