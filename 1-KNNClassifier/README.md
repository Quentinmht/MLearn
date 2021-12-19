# K-Nearest Neighbor classifier
Tell me who your neighbors are, and I’ll tell you who you are

## Principle
It relies on the distance between feature vectors/images to make a classification.

## Hyper Parameters
There are two hyperparameters that we are concerned with when running the k-NN algorithm :

- The value of k (number of neighbors)
- The distance metric

## Algos
I will use the most popular distance metric: The Euclidean distance.

In reality, you can use whichever distance metric/similarity function that most suits your data
(and gives you the best classification results).

## Running
python KnnClassifier.py --dataset "path to dataset"