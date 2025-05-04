🦟 Malaria Case Anomaly Detection (Nigeria Dataset)
This project visualizes and detects anomalies in malaria case locations across Nigerian villages from 2015 to 2019 using One-Class SVM. The focus is on spotting unusual geographical patterns that might suggest data errors, outbreaks, or reporting inconsistencies. This real-time data is provided by one of my friend Mvuyandiani Lyson.

📁 Files Included
Malaria.py – Main Python script for:

Loading malaria dataset

Performing anomaly detection using One-Class SVM

Visualizing results with matplotlib and geopandas

malaria_2015_19.csv – CSV file (not included here) with real malaria case data containing LATITUDE and LONGITUDE.

Scatter plotting of Malaria Dataset.png – Output visualization showing detected anomalies in red.

World shapefile – Required for geographical plotting (path used in code, not included in repo).

🚀 How It Works
Data Preprocessing:

Extracts geographic coordinates (LATITUDE, LONGITUDE)

Scales features for SVM processing

Model Training & Tuning:

Trains a One-Class SVM with grid search for best nu and kernel

Detects anomalies as data points that deviate significantly from normal clusters

Visualization:

Scatter plot highlighting normal vs. anomalous points

World map showing malaria cases overlaid geographically

🧪 Requirements
pip install pandas numpy scikit-learn matplotlib geopandas
📊 Sample Output

🔵 Blue Dots: Normal cases

🔴 Red Dots: Anomalies (Unusual geolocation patterns)

🗺 Notes
Uses a shapefile for world boundaries. Download from Natural Earth and update the path in the script:


shapefile_path = "path_to_shapefile/ne_110m_admin_0_countries.shp"
📌 Applications
Public health anomaly detection

Monitoring disease spread & data quality

Early warning systems for outbreaks

👤 Author
Abhishek Pandey and Mvuyandiani Lyson
Malaria Mapping & AI Research – 2025

