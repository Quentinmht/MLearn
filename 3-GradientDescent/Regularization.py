import argparse
from imutils import paths
from sklearn.linear_model import SGDClassifier
from MLUtils.loader.DatasetLoader import DatasetLoader

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of images paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
dataset_loader = DatasetLoader(32, 32)
(data, labels) = dataset_loader.load_fit_in_ram_dataset(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1024.0)))

# encode the labels as integers
labels, classes = dataset_loader.convert_labels_to_integer(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
testing_dataset_size = 0.25
random_factor = 42
(trainX, testX, trainY, testY) = dataset_loader.train_test_split(data, labels, testing_dataset_size, random_factor)

# loop over our set of regularizers
for r in (None, "l1", "l2"):
    # train a SGD classifier using a softmax loss function and the
    # specified regularization function for 10 epochs
    print("[INFO] training model with `{}` penalty".format(r))
    model = SGDClassifier(loss="log", penalty=r, max_iter=100, learning_rate="constant", tol=1e-3, eta0=0.01,
                          random_state=42)
    model.fit(trainX, trainY)

    # evaluate the classifier
    acc = model.score(testX, testY)
    print("[INFO] `{}` penalty accuracy: {:.2f}%".format(r, acc * 100))
