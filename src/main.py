from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Initial data
f = pd.read_csv('btc.csv')
print(f)

plt.plot(f['Close'])
plt.show()

#Accumulation function
F = []
F.append(f['Close'][0])
for i in range(len(f) - 1):
	F.append(F[i] + f['Close'][i + 1])

plt.plot(F, linewidth = 1)
plt.show()

#Linear regression
x = np.array(range(15, len(f))).reshape(-1, 1)
y = np.log(f['Close'][15:] / F[15:])
model = LinearRegression().fit(x, y)

plt.plot(x, model.predict(x))
plt.plot(x, y, 'rx')
plt.show()

#Inflection point
x_inf = (np.log(abs(model.coef_[0])) - model.intercept_) / model.coef_[0]
print(x_inf)
