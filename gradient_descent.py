# Gradient descent for a simple quadratic function y = x^2
# Define the function and its derivative
def f(x):
    return x**2
def df(x):
    return 2*x
# Implement gradient descent
x = 10.0 # Initial guess
learning_rate = 0.1
iterations = 20
history = []
for i in range(iterations):
x = x - learning_rate * df(x)
history.append((i, x, f(x)))
# Visualize the convergence
history = np.array(history)
plt.plot(history[:, 0], history[:, 2], 'b.-')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Convergence of Gradient Descent')
plt.show()