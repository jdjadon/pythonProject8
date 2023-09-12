import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

threshold = float(input("threshold limit : "))
data = pd.read_csv('dataset4.csv',sep=',')
datasets = {}
for col in data.columns[1:]:
    dataset_name = col.strip()  # Remove any leading/trailing spaces from column names
    dataset = data[['t', col]].copy()
    dataset.columns = ['t', 'sensor_value']  # Rename the columns

    # Filter the dataset to consider only values up to the threshold limit
    dataset = dataset[dataset['sensor_value'] <= threshold]

    # Find the index of the last recorded value in the dataset
    last_recorded_index = dataset['sensor_value'].last_valid_index()
    # Filter the dataset to consider only the rows up to the last recorded value
    dataset = dataset.loc[:last_recorded_index]

    # Calculate the Remaining Useful Life (RUL) for each dataset
    last_recorded_time = dataset['t'].max()
    dataset['RUL'] = last_recorded_time - dataset['t']


    datasets[dataset_name] = dataset

# Step 4: Concatenate the DataFrames into a single DataFrame
merged_data = pd.concat(datasets.values())

# Optionally, sort the DataFrame by time (t) if it's not already sorted
merged_data.sort_values(by='t', inplace=True)
print(merged_data)

# Step 7: Split the data into features (X) and target variable (y)
feature_value = merged_data[['t','sensor_value']]
RUL = merged_data['RUL']


# Step 9: Create the Random Forest Regression model
regressor = RandomForestRegressor(n_estimators=100, random_state=0)

# Step 10: Train the model on the training data
regressor.fit(feature_value.values, RUL.values)

# Step 11: Make predictions on the testing data

# sen_value = float(input("sensor data : "))
# time_value = float(input("time : "))
# y_pred = regressor.predict([[time_value,sen_value]])
#
#
# print(y_pred[0])

# Generate t_values and time_values for the graph
sensor_values = np.linspace(25, 80, 10)
time_values = np.linspace(0, 20000, 100)

# Store the predicted RUL values for different sensor values and time points
predicted_rul_values = []
for time in time_values:
    for sensor_value in sensor_values:
        prediction = regressor.predict([[time, sensor_value]])
        predicted_rul_values.append((time, sensor_value, prediction[0]))

# Convert the results into a DataFrame for easier manipulation
predicted_rul_df = pd.DataFrame(predicted_rul_values, columns=['t', 'sensor_value', 'predicted_rul'])

# Step 12: Create a 3D plot to visualize the predicted RUL values (with interpolation for smoothness)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for the sensor values and time values
time_mesh, sensor_mesh = np.meshgrid(time_values, sensor_values)
# Extract data for the 3D plot
x = predicted_rul_df['t']
y = predicted_rul_df['sensor_value']
z = predicted_rul_df['predicted_rul']

# Perform 2D interpolation to obtain smooth RUL values over the meshgrid
smooth_rul_values = griddata((x, y), z, (time_mesh, sensor_mesh), method='cubic')

# Plot the interpolated surface
ax.plot_surface(time_mesh, sensor_mesh, smooth_rul_values, cmap='viridis')

# Set labels for the axes
ax.set_xlabel('Time')
ax.set_ylabel('Sensor Value')
ax.set_zlabel('Predicted RUL')

# Show the plot
plt.show()