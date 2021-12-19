# Perceptron
First introduced by Rosenblatt in 1958, The Perceptron: A Probabilistic Model for Information
Storage and Organization in the Brain is arguably the oldest and most simple of the ANN
algorithms.

Rosenblatt defined a Perceptron as a system that learns using labeled examples (i.e., supervised
learning) of feature vectors (or raw pixel intensities), mapping these inputs to their corresponding
output class labels.

The Perceptron is still a very important algorithm to understand as it sets the stage for more advanced multi-layer networks.

This example demonstrates the limitation of Perceptron with a non Linear problem like XOR problem

No matter how many times you run this experiment with varying learning rates or different
weight initialization schemes, you will never be able to correctly model the XOR function with a
single layer Perceptron. Instead, what we need is more layers with nonlinear activation functions –
and with that, comes the start of deep learning.

## Principle
1. Initialize our weight vector w with small random values
2. Until Perceptron converges:
   (a) Loop over each feature vector xj and true class label di in our training set D
   (b) Take x and pass it through the network, calculating the output value: yj = f (w(t) . xj)
   (c) Update the weights w: wi(t +1) = wi(t)+a(dj - yj)xj;i for all features 0 <= i <= n

## Running
python file.py 

# BackPropagation
Backpropagation is arguably the most important algorithm in neural network history – without
(efficient) backpropagation, it would be impossible to train deep learning networks to the depths
that we see today. Backpropagation can be considered the cornerstone of modern neural networks
and deep learning.

## Principle
The backpropagation algorithm consists of two phases:

1. The forward pass where our inputs are passed through the network and output predictions
   obtained (also known as the propagation phase).
2. The backward pass where we compute the gradient of the loss function at the final layer (i.e.,
   predictions layer) of the network and use this gradient to recursively apply the chain rule to
   update the weights in our network (also known as the weight update phase).

## Running
python file.py --output "dest output"