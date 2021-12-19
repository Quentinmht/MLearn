# Parameterized Learning - LinearFunction
The purpose of this example is not to demonstrate how we train a model from start to finish but to simply show how we would initialize a weight matrix W ,
bias vector b, and then use these parameters to classify an image via a simple dot product.

## Principle
It's the most common algorithm used to train neural networks and deep learning models.

The gradient descent algorithm has two primary flavors:
1. The standard “vanilla” implementation : only performs a weight update once for every epoch.
2. The optimized “stochastic” version that is more commonly used : performs a weight update for every batch of training
data, implying there are multiple weight updates per epoch.

Stochastic Gradient Descent (SGD) is arguably the most important algorithm when it comes to training deep neural networks.

There are two primary extensions that you’ll encounter to SGD in practice. The first is momentum, 
a method used to accelerate SGD, enabling it to learn faster by focusing on dimensions whose
gradient point in the same direction. The second method is Nesterov acceleration, an extension
to standard momentum.

Nesterov Acceleration : If we build up too much momentum, we may
overshoot a local minimum and keep on rolling. Therefore, it would be advantageous to have a
smarter roll, one that knows when to slow down, which is where Nesterov accelerated gradient [92]
comes in. As for Nesterov acceleration, I tend to use it on smaller datasets, but for larger datasets (such as
ImageNet), I almost always avoid it.

Regularization : Regularization helps us control our model capacity, ensuring that our models are better at
making (correct) classifications on data points that they were not trained on, which we call the
ability to generalize.

## Running
python <Type>GradientDescent.py 