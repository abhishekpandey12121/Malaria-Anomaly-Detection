import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import geopandas as gpd

# Load the dataset
file_path = "C:/Users/abhishek pandey/Downloads/malaria_2015_19.csv"
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print("Error: The dataset file was not found. Please check the file path.")
    exit()

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Ensure 'LONGITUDE' and 'LATITUDE' columns exist
if 'LONGITUDE' not in data.columns or 'LATITUDE' not in data.columns:
    print("Error: Dataset must contain 'LONGITUDE' and 'LATITUDE' columns.")
    exit()

# Extract features (longitude and latitude)
X = data[['LONGITUDE', 'LATITUDE']].values

# Normalize the data for SVM
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Define a custom scoring function for GridSearchCV
def custom_score(estimator, X):
    """Custom scoring function for OneClassSVM."""
    predictions = estimator.predict(X)
    # In OneClassSVM, 1 means normal, -1 means anomaly
    normal_count = np.sum(predictions == 1)  # Count normal points
    return normal_count / len(predictions)  # Proportion of normal points

# Wrap the scoring function with `make_scorer`
scorer = make_scorer(custom_score, greater_is_better=True, needs_proba=False)

# Create and tune the SVM model using GridSearchCV
param_grid = {'nu': [0.1, 0.2, 0.3, 0.4, 0.5], 'kernel': ['rbf', 'linear']}
grid = GridSearchCV(svm.OneClassSVM(), param_grid, scoring=scorer, cv=5)
grid.fit(X_train)

# Display the best parameters
best_params = grid.best_params_
print(f"Best Parameters: {best_params}")

# Train the model with the best parameters
model = svm.OneClassSVM(kernel=best_params['kernel'], nu=best_params['nu'])
model.fit(X_train)

# Predict anomalies on the test set
y_pred = model.predict(X_test)

# Interpret predictions (1 = normal, -1 = anomaly)
anomalies = np.where(y_pred == -1)[0]  # Indices of detected anomalies

# (Optional) Evaluation if ground truth labels are available
# Uncomment the following lines if you have `y_test` labels
# y_test = ...  # Provide your ground truth labels here
# print("Classification Report:")
# print(classification_report(y_test, y_pred))
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], color='blue', label='Normal Cases')
if anomalies.size > 0:  # Plot anomalies only if they exist
    plt.scatter(
        X_test[anomalies, 0],
        X_test[anomalies, 1],
        color='red',
        label='Anomalies',
        edgecolor='k',
        s=100
    )
plt.title('Anomaly Detection in Malaria Cases')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid()
plt.show()

# Step 1: Load the world map shapefile
shapefile_path = "C:/Users/abhishek pandey/Downloads/ne_110m_admin_0_countries.shp"  # Update the path
world = gpd.read_file(shapefile_path)

# Step 3: Convert the dataset into a GeoDataFrame
# Assuming the malaria dataset has 'latitude' and 'longitude' columns
gdf_malaria = gpd.GeoDataFrame(
    data, 
    geometry=gpd.points_from_xy(data.LONGITUDE,data.LATITUDE)
)

# Step 4: Plot the world map and overlay malaria data points
fig, ax = plt.subplots(figsize=(15, 10))
world.plot(ax=ax, color="lightgrey", edgecolor="black")  # Base world map
gdf_malaria.plot(ax=ax, color="red", markersize=10, label="Malaria Cases")  # Malaria data points

# Add title and legend
plt.title("Malaria Cases on World Map (2015-2019)", fontsize=16)
plt.legend()
plt.show()