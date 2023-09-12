import math
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
import pandas as pd

# Sample time-to-failure data
datattf = np.array([10700, 6700, 9500, 8700, 10600, 13400, 19700, 6600, 13500, 9800,12400,14600,7800,9500,10500])
p = 25
f = 90


data = pd.read_csv('dataset4.csv',sep=',')
datasets = {}
for col in data.columns[1:]:
    dataset_name = col.strip()  # Remove any leading/trailing spaces from column names
    dataset = data[['t', col]].copy()
    dataset.columns = ['t', 'sensor_value']  # Rename the columns

    # Find the index where the threshold limit is first reached
    first_threshold_index = dataset[dataset['sensor_value'] > f].index.min()

    if not pd.isnull(first_threshold_index):
        # If the threshold is reached, remove all values after the threshold index
        dataset = dataset.loc[:first_threshold_index - 1]

        # Calculate the Remaining Useful Life (RUL) for each dataset
        last_recorded_time = dataset['t'].max()
        dataset['RUL'] = last_recorded_time - dataset['t']

        datasets[dataset_name] = dataset

print(datasets["dataset 3"])


# Estimate beta and eta using MLE
params = weibull_min.fit(datattf, floc=0)

# Unpack the estimated parameters
beta, eta = params[0], params[2]

def remaininglife(vc, t0):
    tp = t0 - 100
    def rul(eta, beta, t0):
        reliability = math.e ** -((t0 /eta) ** beta)
        print(reliability)
        t = (eta * (-math.log(reliability*0.9)) ** (1/beta)) - t0
        return t


    if (vc < p):
        rulp = rul(eta, beta, tp)
        rulc = rul(eta, beta, t0)
    else:
        m = (f - vc)/(f-p)
        etac = eta *m
        rulc = rul(etac, beta, t0)
        rulp = rul(etac, beta, tp)

    # if rulc is less than rulp then take rulc else rulp

    if (rulc < rulp):
        return (rulc)
    else:
        return (rulp)

print(remaininglife(30,200))

# Generate t_values and time_values for the graph
time_values = list(datasets["dataset 5"]['t'][10:80])
sensor_values = list(datasets["dataset 5"]['sensor_value'][10:80])
print(time_values)
print(sensor_values)
# Store the predicted RUL values for different sensor values and time points
predicted_rul_values = []
for i,j in enumerate(time_values):
    print(i,j)
    prediction =remaininglife(sensor_values[i], j)
    predicted_rul_values.append((j, sensor_values[i], prediction))

# Convert the results into a DataFrame for easier manipulation
predicted_rul_df = pd.DataFrame(predicted_rul_values, columns=['t', 'sensor_value', 'predicted_rul'])

# Step 12: Create a 3D plot to visualize the predicted RUL values
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract data for the 3D plot
x = predicted_rul_df['t']
y = predicted_rul_df['sensor_value']
z = predicted_rul_df['predicted_rul']

a = datasets["dataset 5"]['t']
b = datasets["dataset 5"]['sensor_value']
c = datasets["dataset 5"]['RUL']

# Plot the data points
ax.scatter(x, y, z, c='r', marker='o')
ax.scatter(a, b, c, c='b', marker='x')

# Set labels for the axes
ax.set_xlabel('Time')
ax.set_ylabel('Sensor Value')
ax.set_zlabel('Predicted RUL')

# Show the plot
plt.show()


