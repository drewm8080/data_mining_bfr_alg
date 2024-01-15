# BFR Algorithm Implementation

### Overview of the Assignment
I will implement the Bradley-Fayyad-Reina (BFR) algorithm on a synthetic dataset. The goal is to familiarize myself with the clustering process and various distance measurements using synthetic datasets.

### Dataset
Synthetic datasets were generated to meet the strong assumption of the BFR algorithm that clusters are normally distributed with independent dimensions. The dataset includes random centroids, data points, and outliers for algorithm evaluation. (not included because it is too large for Github)

### Task 
I will implement the Bradley-Fayyad-Reina (BFR) algorithm to cluster the data in hw6_clustering.txt. The algorithm involves three sets of points: Discard set (DS), Compression set (CS), and Retained set (RS). The steps of the BFR algorithm include loading data, running K-Means, and generating DS and CS clusters.

   - **Step 1**: Load 20% of the data randomly.
   - **Step 2**: Run K-Means with a large K on the data using the Euclidean distance as the similarity measurement.
   - **Step 3**: Move single-point clusters to RS (outliers).
   - **Step 4**: Run K-Means again to cluster the remaining data points.
   - **Step 5**: Generate DS clusters and statistics from the K-Means result.
   - **Step 6**: Run K-Means on the points in RS to generate CS and RS clusters.
   - **Step 7**: Update the DS and CS clusters based on the new points in the dataset.
   - **Step 8**: Repeat steps 1-7 until all data points are processed.
   - **Step 9**: Merge the DS and CS clusters to form the final set of clusters.
   - **Step 10**: Assign each data point to the nearest cluster.
   - **Step 11**: Output the cluster assignments for each data point.
   - **Step 12**: Evaluate the clustering performance using appropriate metrics.

The BFR algorithm aims to efficiently cluster the dataset and handle outliers.

### Output
The output file is a text file, containing the following information:

a. The intermediate results (the line is named as “The intermediate results”):
I will start each line with “Round { }” and output the numbers in the order of “the number of the discard points”, “the number of the clusters in the compression set”, “the number of the compression points”, and “the number of the points in the retained set”. I will leave one line in the middle before writing out the cluster results.

b. The clustering results (the line is named as “The clustering results”):
I will include the data points index and their clustering results after the BFR algorithm. The clustering results should be in [0, the number of clusters). The cluster of outliers should be represented as -1.


- **Accuracy**: 99.98%
- **Grade**: 100%

