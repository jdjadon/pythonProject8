import pandas as pd
from sklearn.ensemble import RandomForestRegressor

threshold = float(input("threshold limit : "))
data = pd.read_csv('dataset2.csv',sep=',')
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


merged_data = pd.concat(datasets.values())


merged_data.sort_values(by='t', inplace=True)
print(merged_data)
feature_value = merged_data[['t','sensor_value']]
RUL = merged_data['RUL']



regressor = RandomForestRegressor(n_estimators=100, random_state=0)

regressor.fit(feature_value.values, RUL.values)


while True:
    sen_value = float(input("sensor data : "))
    time_value = float(input("time : "))
    y_pred = regressor.predict([[time_value,sen_value]])
    print(y_pred[0])

#