import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
np.set_printoptions(formatter={'float': lambda x: "{0:0.9f}".format(x)})

def save_theta(theta, x, y):
	a = (y[0] - y[1]) / (x[0] - x[1])
	b = a * x[0] * -1 + y[0]
	theta = [b, a]
	np.savetxt("theta.csv", theta, delimiter = ',');

def visualize(data, x, y):
	plt.plot(data['km'], data['price'], 'o')
	plt.plot(x, y)
	plt.ylabel("Price")
	plt.xlabel("Km")
	plt.show()

def normalise(x):
	return (x - np.mean(x)) / np.std(x)

def unnormalise(x, x_ref):
	return (x * np.std(x_ref) + np.mean(x_ref))

def predict(theta_0, theta_1, x):
	return (theta_0 + theta_1 * x)

def gradientDescent(x, y, m, theta, alpha, iterations):
	for i in range(0, iterations):
		tmp_theta = np.zeros((1, 2))
		for j in range(0, m):
			tmp_theta[0, 0] += (predict(theta[0, 0], theta[0, 1], x[j]) - y[j])
			tmp_theta[0, 1] += ((predict(theta[0, 0], theta[0, 1], x[j]) - y[j]) * x[j])
		theta -= (tmp_theta * alpha) / m
	return theta

def cost(X, y, theta):
	loss = ((predict(theta[0, 0], theta[0, 1], x)) - y)
	cost = (1 / (2 * X.size)) * np.dot(loss, loss.T)
	return (cost)

def fit_with_cost(X, y, theta, alpha, iterations):
    m = len(X)
    sigma = [0.0,0.0]
    J_history = []
    for i in range(iterations):
        sigma[0] = 0.0
        sigma[1] = 0.0
        for k in range(m):
            sigma[0] += predict(theta[0, 0], theta[0, 1], x[k]) - y[k]
            sigma[1] += (predict(theta[0, 0], theta[0, 1], x[k]) - y[k]) * X[k]
        theta[0, 0] = theta[0, 0] - alpha * sigma[0] / m
        theta[0, 1] = theta[0, 1] - alpha * sigma[1] / m
        J_history.append(cost(X, y, theta)) 
    return theta, J_history

data = pd.read_csv("data.csv")
data.plot.scatter('km', 'price')
if (len(data) < 2):
	exit()
X = np.array(data['km'])
Y = np.array(data['price'])
x = normalise(X)
y = normalise(Y)
m = len(data)
theta = np.zeros((1, 2))
theta = gradientDescent(x, y, m, theta, 0.01, 1000)
y = predict(theta[0, 0], theta[0, 1], x)
X = unnormalise(x, data['km'])
Y = unnormalise(y, data['price'])
save_theta(theta, X, Y)
visualize(data, X, Y)
theta = np.zeros((1, 2))
theta, J_history = fit_with_cost(x, y, theta, 0.01, 1000)
fit = plt.figure()
ax = plt.axes()
ax.plot(J_history)
plt.show()