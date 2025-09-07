🛍️ Customer Segmentation using K-Means Clustering

This project applies K-Means clustering to segment customers based on their Annual Income and Spending Score. Customer segmentation is a common unsupervised learning technique used in marketing to understand customer groups and target them effectively.

.

📌 How It Works

1. Import Libraries
*numpy, pandas → data manipulation and analysis.
*matplotlib, seaborn → visualization.
*scikit-learn → machine learning (KMeans clustering).

2. Load Dataset
*Reads Mall_Customers.csv into a Pandas DataFrame.
*Checks shape, info, and missing values.

3. Feature Selection
*Selects two columns for clustering: Annual Income & Spending Score

4. Determine Optimal Clusters (Elbow Method)
*Calculates WCSS (Within-Cluster Sum of Squares) for 1 to 10 clusters.
*Plots the Elbow Graph to identify the optimal number of clusters (chosen as 5).

5. Train the K-Means Model
*K-Means is trained with 5 clusters (n_clusters=5).
*Each customer is assigned a cluster label.

6. Visualization
*Plots the customer groups in different colors.
*Marks the cluster centroids.
*Graph helps to visually understand the customer segments.

🛠️ Requirements

Install the required libraries: pip install -r requirements.txt

▶️ Usage
- Place Mall_Customers.csv in the same folder as the script.
- Run the script: python Customer_Segmentation.py
- The program will:
. Perform clustering on the dataset.
. Show the Elbow Graph.
. Display the Customer Group visualization with 5 clusters.

📊 Example Output

Elbow Method Graph to choose optimal clusters
Customer Segmentation Graph with 5 clusters
