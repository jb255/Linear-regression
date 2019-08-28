import numpy as np

def predict(theta_0, theta_1, x):
	return theta_0 + theta_1 * x

try:
	theta = np.loadtxt("theta.csv", dtype = np.longdouble, delimiter = ',')
except:
	print ("theta.csv not found")
	exit()

try:
	x = np.longdouble(input("Enter km value of vehicule: "))
except:
	print ("Error")
	exit()
print (predict(theta[0], theta[1], x))