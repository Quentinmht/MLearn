# Parameterized Learning - LinearFunction
The purpose of this example is not to demonstrate how we train a model from start to finish but to simply show how we would initialize a weight matrix W ,
bias vector b, and then use these parameters to classify an image via a simple dot product.

## Principle
A learning model that summarizes data with a set of parameters of fixed size
(independent of the number of training examples) is called a parametric model. No
matter how much data you throw at the parametric model, it won’t change its mind
about how many parameters it needs.

parameterized learning is the cornerstone of modern day machine learning and deep learning algorithms.

In the task of machine learning, parameterization involves defining a problem in
terms of four key components: data, a scoring function, a loss function, and weights and biases.

data : This component is our input data that we are going to learn from. This data includes both the data
points (i.e., raw pixel intensities from images, extracted features, etc.) and their associated class
labels. Typically we denote our data in terms of a multi-dimensional design matrix

Scoring Function : The scoring function accepts our data as an input and maps the data to class labels. For instance,
given our set of input images, the scoring function takes these data points, applies some function f (our scoring function), 
and then returns the predicted class labels

Loss Function : A loss function quantifies how well our predicted class labels agree with our ground-truth labels.
The higher level of agreement between these two sets of labels, the lower our loss (and higher our
classification accuracy, at least on the training set).
Our goal when training a machine learning model is to minimize the loss function, thereby
increasing our classification accuracy.

Weights and Biases : The weight matrix, typically denoted as W and the bias vector b are called the weights or
parameters of our classifier that we’ll actually be optimizing. Based on the output of our scoring
function and loss function, we’ll be tweaking and fiddling with the values of the weights and biases
to increase classification accuracy.

## Running
python LinearClassifier.py 