
Training a Neural Network
=========================

In this example, we'll be training a neural network using particle swarm
optimization. For this we'll be using the standard global-best PSO
``pyswarms.single.GBestPSO`` for optimizing the network's weights and
biases. This aims to demonstrate how the API is capable of handling
custom-defined functions.

For this example, we'll try to classify the three iris species in the
Iris Dataset.

.. code-block:: python

    # Import modules
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    
    
    # Import PySwarms
    import pyswarms as ps
    
    # Some more magic so that the notebook will reload external python modules;
    # see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
    %load_ext autoreload
    %autoreload 2

First, we'll load the dataset from ``scikit-learn``. The Iris Dataset
contains 3 classes for each of the iris species (*iris setosa*, *iris
virginica*, and *iris versicolor*). It has 50 samples per class with 150
samples in total, making it a very balanced dataset. Each sample is
characterized by four features (or dimensions): sepal length, sepal
width, petal length, petal width.

.. code-block:: python

    # Load the iris dataset
    data = load_iris()
    
    # Store the features as X and the labels as y
    X = data.data
    y = data.target

Constructing a custom objective function
----------------------------------------

Recall that neural networks can simply be seen as a mapping function
from one space to another. For now, we'll build a simple neural network
with the following characteristics: 

* Input layer size: 4 

* Hidden layer size: 20 (activation: :math:`\tanh(x)`)

* Output layer size: 3 (activation: :math:`softmax(x)`)

Things we'll do: 

1. Create a ``forward_prop`` method that will do forward propagation for one particle.

2. Create an overhead objective function ``f()`` that will compute ``forward_prop()`` for the whole swarm.

What we'll be doing then is to create a swarm with a number of
dimensions equal to the weights and biases. We will **unroll** these
parameters into an n-dimensional array, and have each particle take on
different values. Thus, each particle represents a candidate neural
network with its own weights and bias. When feeding back to the network,
we will reconstruct the learned weights and biases.

When rolling-back the parameters into weights and biases, it is useful
to recall the shape and bias matrices: 

* Shape of input-to-hidden weight matrix: (4, 20)

* Shape of input-to-hidden bias array: (20, )

* Shape of hidden-to-output weight matrix: (20, 3)

* Shape of hidden-to-output bias array: (3, )

By unrolling them together, we have
:math:`(4 * 20) + (20 * 3) + 20 + 3 = 163` parameters, or 163 dimensions
for each particle in the swarm.

The negative log-likelihood will be used to compute for the error
between the ground-truth values and the predictions. Also, because PSO
doesn't rely on the gradients, we'll not be performing backpropagation
(this may be a good thing or bad thing under some circumstances).

Now, let's write the forward propagation procedure as our objective
function. Let :math:`X` be the input, :math:`z_l` the pre-activation at
layer :math:`l`, and :math:`a_l` the activation for layer :math:`l`:

.. code-block:: python

    # Forward propagation
    def forward_prop(params):
        """Forward propagation as objective function
        
        This computes for the forward propagation of the neural network, as
        well as the loss. It receives a set of parameters that must be 
        rolled-back into the corresponding weights and biases.
        
        Inputs
        ------
        params: np.ndarray
            The dimensions should include an unrolled version of the 
            weights and biases.
            
        Returns
        -------
        float
            The computed negative log-likelihood loss given the parameters
        """
        # Neural network architecture
        n_inputs = 4
        n_hidden = 20
        n_classes = 3
        
        # Roll-back the weights and biases
        W1 = params[0:80].reshape((n_inputs,n_hidden))
        b1 = params[80:100].reshape((n_hidden,))
        W2 = params[100:160].reshape((n_hidden,n_classes))
        b2 = params[160:163].reshape((n_classes,))
        
        # Perform forward propagation
        z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
        a1 = np.tanh(z1)     # Activation in Layer 1
        z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
        logits = z2          # Logits for Layer 2
        
        # Compute for the softmax of the logits
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
        
        # Compute for the negative log likelihood
        N = 150 # Number of samples
        corect_logprobs = -np.log(probs[range(N), y])
        loss = np.sum(corect_logprobs) / N
        
        return loss
    

Now that we have a method to do forward propagation for one particle (or
for one set of dimensions), we can then create a higher-level method to
compute ``forward_prop()`` to the whole swarm:

.. code-block:: python

    def f(x):
        """Higher-level method to do forward_prop in the 
        whole swarm.
        
        Inputs
        ------
        x: numpy.ndarray of shape (n_particles, dims)
            The swarm that will perform the search
            
        Returns
        -------
        numpy.ndarray of shape (n_particles, )
            The computed loss for each particle
        """
        n_particles = x.shape[0]
        j = [forward_prop(x[i]) for i in range(n_particles)]
        return np.array(j)
        

Performing PSO on the custom-function
-------------------------------------

Now that everything has been set-up, we just call our global-best PSO
and run the optimizer as usual. For now, we'll just set the PSO
parameters arbitrarily.

.. code-block:: python

    # Initialize swarm
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    
    # Call instance of PSO with bounds argument
    dims = (4 * 20) + (20 * 3) + 20 + 3 
    optimizer = ps.single.GBestPSO(n_particles=100, dims=dims, **options)
    
    # Perform optimization
    cost, pos = optimizer.optimize(f, print_step=100, iters=1000, verbose=3)


.. parsed-literal::

    Iteration 1/1000, cost: 1.11338932053
    Iteration 101/1000, cost: 0.0541135752532
    Iteration 201/1000, cost: 0.0468046270747
    Iteration 301/1000, cost: 0.0434828849533
    Iteration 401/1000, cost: 0.0358833340106
    Iteration 501/1000, cost: 0.0312474981647
    Iteration 601/1000, cost: 0.0150869267541
    Iteration 701/1000, cost: 0.01267166403
    Iteration 801/1000, cost: 0.00632312205821
    Iteration 901/1000, cost: 0.00194080306565
    ================================
    Optimization finished!
    Final cost: 0.0015
    Best value: -0.356506 0.441392 -0.605476 0.620517 -0.156904 0.206396 ...
    
    

Checking the accuracy
---------------------

We can then check the accuracy by performing forward propagation once
again to create a set of predictions. Then it's only a simple matter of
matching which one's correct or not. For the ``logits``, we take the
``argmax``. Recall that the softmax function returns probabilities where
the whole vector sums to 1. We just take the one with the highest
probability then treat it as the network's prediction.

Moreover, we let the best position vector found by the swarm be the
weight and bias parameters of the network.

.. code-block:: python

    def predict(X, pos):
        """
        Use the trained weights to perform class predictions.
        
        Inputs
        ------
        X: numpy.ndarray
            Input Iris dataset
        pos: numpy.ndarray
            Position matrix found by the swarm. Will be rolled
            into weights and biases.
        """
        # Neural network architecture
        n_inputs = 4
        n_hidden = 20
        n_classes = 3
        
        # Roll-back the weights and biases
        W1 = pos[0:80].reshape((n_inputs,n_hidden))
        b1 = pos[80:100].reshape((n_hidden,))
        W2 = pos[100:160].reshape((n_hidden,n_classes))
        b2 = pos[160:163].reshape((n_classes,))
        
        # Perform forward propagation
        z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
        a1 = np.tanh(z1)     # Activation in Layer 1
        z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
        logits = z2          # Logits for Layer 2
        
        y_pred = np.argmax(logits, axis=1)
        return y_pred

And from this we can just compute for the accuracy. We perform
predictions, compare an equivalence to the ground-truth value ``y``, and
get the mean.

.. code-block:: python

    (predict(X, pos) == y).mean()


.. parsed-literal::

    1.0


