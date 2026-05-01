import numpy as np
import matplotlib.pyplot as plt

def generate_spiral(N, classes, noise):
    t = np.linspace(0, 1, N)   
    r = t                          
    angle = t * 4 * np.pi         

    x = r * np.cos(angle)
    y = r * np.sin(angle)

    angle2 = t * 4 * np.pi + np.pi
    x2 = r * np.cos(angle2)
    y2 = r * np.sin(angle2)

    plt.scatter(x, y, color='blue')
    plt.scatter(x2, y2, color='red')
    

    x = x + np.random.randn(N) * noise
    y  = y  + np.random.randn(N) * noise
    x2 = x2 + np.random.randn(N) * noise
    y2 = y2 + np.random.randn(N) * noise
    plt.show()

    
    X = np.vstack([
        np.column_stack([x, y]),
        np.column_stack([x2, y2])
    ])

    
    Y = np.vstack([
        np.tile([1, 0], (N, 1)),   
        np.tile([0, 1], (N, 1))   
    ])

    return X,Y

##Task 2.1 — Initialization
##Initialize weights using He initialization and biases to zero. Store all parameters in a dictionary `params` with keys `W1, b1, W2, b2, W3, b3`.

def initialize_params():
    params = {}
    params['W1'] = np.random.randn(2, 64) * np.sqrt(2 / 2)
    params['b1'] = np.zeros((1, 64))
    params['W2'] = np.random.randn(64, 64) * np.sqrt(2/64)
    params['b2'] = np.zeros((1, 64))
    params['W3'] = np.random.randn(64, 2) * np.sqrt(2/64)
    params['b3'] = np.zeros((1, 2))

    return params

    
## Task 2.2 — Activation functions
#Implement `relu(Z)`, `relu_backward(dA, Z)`, and `softmax(Z)` as standalone functions. Do not use any library's built-in activation. For numerical stability, subtract the row-wise maximum before exponentiating in Softmax.

def relu(Z):
    return np.maximum(Z, 0)

def relu_backward(da,Z):
    dZ = da * (Z > 0)
    return dZ

def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)

##Task 3.1
##Implement `forward(X, params)`. It must return both the final output `A3` (probabilities) and a cache dictionary containing every intermediate value needed for backprop: `Z1, A1, Z2, A2, Z3`.
def forward(X, params):
    Z1 = np.dot(X, params['W1']) + params['b1']
    A1 = relu(Z1)

    Z2 = np.dot(A1, params['W2']) + params['b2']
    A2 = relu(Z2)

    Z3 = np.dot(A2, params['W3']) + params['b3']
    A3 = softmax(Z3)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3}

    return A3, cache

## Task 4.1
##Implement `compute_loss(A3, Y)` returning the mean categorical cross-entropy loss over the batch. Clip probabilities to `[1e-15, 1]` before taking the log.
##`L = -1/m · Σ Σ Y_ij · log(A3_ij)`

def compute_loss(A3, Y):
    m = A3.shape[0]
    A3 = np.clip(A3, 1e-15, 1)
    loss = -1/m * np.sum(Y * np.log(A3))
    return loss

##Task 5.1 — Output layer gradient

## Show (in a comment block) that when Softmax is paired with cross-entropy loss, the gradient simplifies to:

##`dZ3 = A3 - Y`

## Then compute `dW3`, `db3`, and `dA2`.

## When Softmax and cross-entropy are combined, the math simplifies beautifully. You don't need to derive it fully right now — just know that the gradient of the loss with respect to Z3 is simply:
## dZ3 = A3 - Y

"""
Task 5.2 — Hidden layer gradients
Propagate the gradient through each ReLU layer. The ReLU backward pass gates the gradient: it passes through only where the pre-activation `Z` was positive. Compute `dW2, db2, dA1`, then `dW1, db1`. All six gradients must be stored in a dictionary `grads`.

"""
def backward(X, Y, params, cache, A3):
    m = X.shape[0]

    A2 = cache['A2']
    Z1 = cache['Z1']
    A1 = cache['A1']
    Z2 = cache['Z2']

    dZ3 = A3 - Y
    dW3 = (1/m) * np.dot(A2.T, dZ3)
    db3 = (1/m) * np.sum(dZ3, axis=0, keepdims=True)
    dA2 = np.dot(dZ3, params['W3'].T)

    dZ2 = relu_backward(dA2, Z2)
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
    dA1 = np.dot(dZ2, params['W2'].T)
    
    dZ1 = relu_backward(dA1, Z1)
    dW1 = (1/m) * np.dot(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

    grads = {
    'dW1': dW1, 'db1': db1,
    'dW2': dW2, 'db2': db2,
    'dW3': dW3, 'db3': db3
    }

    return grads
"""
Task 5.3 — Gradient check (sanity check)

For at least one weight element in `W1`, verify your analytic gradient numerically using the two-sided finite difference approximation with `ε = 1e-5`. Report the relative error. A correct implementation should give a relative error below `1e-5`.

`grad_numerical ≈ (L(θ + ε) − L(θ − ε)) / (2ε)`
"""

def gradient_check(X, Y, params, grads):
    ε = 1e-5
    
    # nudge W1[0,0] up → forward → loss
    params['W1'][0,0] += ε
    A3, _ = forward(X, params)
    L_plus = compute_loss(A3, Y)
    params['W1'][0,0] -= ε
    # nudge W1[0,0] down → forward → loss
    params['W1'][0,0] -= ε
    A3, _ = forward(X, params)
    L_minus = compute_loss(A3, Y)
    params['W1'][0,0] += ε
    # compute numerical gradient
    numerical = (L_plus - L_minus) / (2 * ε)
    # get analytic gradient from grads
    analytic = grads['dW1'][0,0]
    # compute relative error
    relative_error = np.abs(numerical - analytic) / (np.abs(numerical) + np.abs(analytic))
    # print the result
    print("Numerical gradient:", numerical)
    print("Analytic gradient:", analytic)
    print("Relative error:", relative_error)

"""
Task 6.1 — Gradient descent
Implement `update_params(params, grads, lr)` using vanilla gradient descent. Then write a training loop that runs for **10,000 iterations** with a learning rate of **0.01** and prints the loss every 1,000 steps.
"""

def update_params(params, grads, lr):
    params['W1'] = params['W1'] - lr * grads['dW1']
    params['b1'] = params['b1'] - lr * grads['db1']
    params['W2'] = params['W2'] - lr * grads['dW2']
    params['b2'] = params['b2'] - lr * grads['db2']
    params['W3'] = params['W3'] - lr * grads['dW3']
    params['b3'] = params['b3'] - lr * grads['db3']

    return params

X, Y = generate_spiral(100, 2, 0.2)
params = initialize_params()
lr = 0.1
losses = []

for i in range(0, 50000):
    A3, cache = forward(X, params)
    loss = compute_loss(A3, Y)
    grads = backward(X, Y, params, cache, A3)
    params = update_params(params, grads, lr)
    losses.append(loss)
    
    if i % 1000 == 0:
        print(f"Step {i}, Loss: {loss}")

plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Learning curve')
plt.show()

"""
7. Decision boundary

Task 7.1

After training, generate a dense grid of points over the input space and run a forward pass on each. Use `plt.contourf` to visualize the decision boundary, overlaid with the original data points. Report the final training accuracy. A correct implementation should exceed **95% accuracy** on the training set.

"""
xx, yy = np.meshgrid(
    np.linspace(-1.5, 1.5, 200),
    np.linspace(-1.5, 1.5, 200)
)

grid = np.column_stack([xx.ravel(), yy.ravel()])

A3_grid, _ = forward(grid, params)
predictions = np.argmax(A3_grid, axis=1)
predictions = predictions.reshape(xx.shape)
plt.contourf(xx, yy, predictions, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=np.argmax(Y, axis=1))
plt.show()

train_preds = np.argmax(forward(X, params)[0], axis=1)
train_labels = np.argmax(Y, axis=1)
accuracy = np.mean(train_preds == train_labels)
print(f"Training accuracy: {accuracy * 100:.2f}%")








