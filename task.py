from sklearn.cluster import KMeans
import numpy as np
import os
import sys
import time
import random
import math
from itertools import combinations

   
def write_intermediate_results(round_num, ds_sum_dict, cs_sum_dict, rs, output_path):
    with open(output_path, 'a') as f:
        if round_num == 1:
            f.write('The intermediate results:\n')
        
        # Calculate the total number of points in ds
        num_discard_points = sum(entry['count'] for entry in ds_sum_dict.values())
        
        # Calculate the total number of clusters in cs
        num_clusters_in_cs = len(cs_sum_dict)
        
        # Calculate the total number of points in cs
        num_compression_points = sum(entry['count'] for entry in cs_sum_dict.values())
        
        # Calculate the length of rs
        num_retained_points = len(rs)
        
        # Write the result to the file
        result_str = 'Round {}: {},{},{},{}\n'.format(round_num, num_discard_points, num_clusters_in_cs, num_compression_points, num_retained_points)
        f.write(result_str)


def compute_merge_distance( center1, center2, std_dev1, std_dev2):
    centroid_diff = np.subtract(center1, center2)
    mean_std_dev = np.divide(np.add(std_dev1, std_dev2), 2)
    normalized_diff = np.divide(centroid_diff, mean_std_dev)
    squared_diff = np.square(normalized_diff)
    distance = np.sqrt(np.sum(squared_diff))
    return distance


def compute_mahalanobis_metric(data_point, cluster_center, standard_deviation):
    deviation = data_point - cluster_center
    normalized_deviation = (deviation / standard_deviation) ** 2
    mahalanobis_metric = np.sqrt(np.sum(normalized_deviation))
    return mahalanobis_metric

if __name__ == '__main__':
    time_start = time.time()
    input_datapath = '/Users/andrewmoore/Desktop/DSCI 553/DSCI 553 HW 6/hw6_clustering.txt'
    input_clusters = 10
    output_path= '/Users/andrewmoore/Desktop/DSCI 553/DSCI 553 HW 6/intermediate_results.txt'
    ds_sum_dict = {}
    ds_indexes_dict = {}
    ds_centroid_dict = {}
    ds_deviation_dict = {}
    full_clusters = 5*input_clusters
    

    ##### STEP 1 Randomly shuffle the data 

    input_data = np.array([np.array(line.strip().split(','), dtype=np.float64) for line in open(input_datapath)])

    
    np.random.shuffle(input_data)
    npdata_partitions = np.array_split(input_data, 5)
    first_partition = npdata_partitions[0]
    threshold = 2 * np.sqrt(first_partition.shape[1])
    data_npdata = input_data[:, 2:]
    # Extract the first row from each partition
    first_rows = [partition[0] for partition in npdata_partitions]

    # Concatenate the first rows into a single array
    all_first_rows = np.vstack(first_rows)






    ##### STEP 2 Randomly shuffle the data 
    inital_kmeans = KMeans(n_clusters=full_clusters, random_state =42).fit(first_partition[:,2:])

    labels = inital_kmeans.labels_

    ##### STEP 3
    unique_clusters = np.where(np.bincount(labels) == 1)[0]

    # Find indices of points belonging to clusters with only one member
    rs = [idx for idx, label in enumerate(labels) if label in unique_clusters]
    ds = np.delete(first_partition,rs,axis= 0)
    rs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]


    ##### step 4 

    new_kmeans = KMeans(n_clusters=input_clusters, random_state =42).fit(ds[:,2:])

    ### step 5

    # Step 5: Generate DS clusters
    labels = new_kmeans.labels_
    cluster_indices = {i: np.where(labels == i)[0] for i in range(new_kmeans.n_clusters)}

    for cluster, indices in cluster_indices.items():
        points = ds[indices,2:]
        ds_sum_dict[cluster] = {
            'count': len(points),
            'sum': np.sum(points, axis=0),
            'sum_sq': np.sum(points ** 2, axis=0)
        }
        indices_list = []
        for x in ds[indices, 0]:
            indices_list.append(int(x))

        ds_indexes_dict[cluster] = indices_list
        ds_centroid_dict[cluster] = ds_sum_dict[cluster]['sum'] / ds_sum_dict[cluster]['count']
        ds_deviation_dict[cluster] = np.sqrt(
            ds_sum_dict[cluster]['sum_sq'] / ds_sum_dict[cluster]['count'] - ds_centroid_dict[cluster] ** 2
        )


    # step 6 
    cs_sum_dict = {}
    cs_indexes_dict = {}
    cs_centroid_dict = {}
    cs_deviation_dict = {}
    five_times_k_clusters = 5 * input_clusters

    # Run K-Means on the points in the RS only if its length is >= 5 * input_clusters
    if len(rs)>= full_clusters:
        kmeans_rs = KMeans(n_clusters=five_times_k_clusters, random_state=42).fit(first_partition[rs, 2:])
        
        # Generate CS clusters
        labels = kmeans_rs.labels_
        cluster_indices = {i: np.where(labels == i)[0] for i in range(kmeans_rs.n_clusters)}
        

        new_rs = []
        
        for cluster, indices in cluster_indices.items():
            points = first_partition[rs][indices, 2:]
            new_data = first_partition[rs]

            if len(points) == 1:
                new_rs.append(indices)
            else:
                cs_sum_dict[cluster] = {
                    'count': len(points),
                    'sum': np.sum(points, axis=0),
                    'sum_sq': np.sum(points ** 2, axis=0)
                }
                indices_list = []
                for x in new_data[indices, 0]:
                    print(x)
                    indices_list.append(int(x))
                cs_indexes_dict[cluster] = indices_list
                cs_centroid_dict[cluster] = cs_sum_dict[cluster]['sum'] / cs_sum_dict[cluster]['count']
                cs_deviation_dict[cluster] = np.sqrt(
                    cs_sum_dict[cluster]['sum_sq'] / cs_sum_dict[cluster]['count'] - cs_centroid_dict[cluster] ** 2
                )

        rs = new_rs
    write_intermediate_results(1,ds_sum_dict, cs_sum_dict, rs, output_path)



    # rs_data =[]
    rs_data = [np.array(i) for i in first_partition[rs,2:]]

    round_count = 1
    for partition in range(1, len(npdata_partitions)):
        new_subset = npdata_partitions[partition]
        round_count =round_count+1

        for npdata_index, coordinates in enumerate(new_subset):
            # Initialize variables
            min_distance_ds, min_cluster_ds = None, None
            min_distance_cs, min_cluster_cs = None, None

            # Step 8: Assign new points to DS clusters using Mahalanobis Distance
            for ds_cluster, centroid in ds_centroid_dict.items():
                deviation = ds_deviation_dict[ds_cluster]
                mahalanobis_distance_ds = compute_mahalanobis_metric(coordinates[2:], centroid, deviation)

                if min_distance_ds is None or mahalanobis_distance_ds < min_distance_ds:
                    min_distance_ds = mahalanobis_distance_ds
                    min_cluster_ds = ds_cluster

            # Step 9: Assign remaining new points to CS clusters using Mahalanobis Distance
            for cs_cluster, centroid in cs_centroid_dict.items():
                deviation = cs_deviation_dict[cs_cluster]
                mahalanobis_distance_cs = compute_mahalanobis_metric(coordinates[2:], centroid, deviation)

                if min_distance_cs is None or mahalanobis_distance_cs < min_distance_cs:
                    min_distance_cs = mahalanobis_distance_cs
                    min_cluster_cs = cs_cluster

            # Assign coordinates to the nearest cluster or RS
            if min_distance_ds is not None and min_distance_ds < threshold:
                # Update DS cluster stats
                ds_sum_dict[min_cluster_ds]['count'] = ds_sum_dict[min_cluster_ds]['count'] + 1
                ds_sum_dict[min_cluster_ds]['sum'] = ds_sum_dict[min_cluster_ds]['sum'] + coordinates[2:]
                ds_sum_dict[min_cluster_ds]['sum_sq'] = ds_sum_dict[min_cluster_ds]['sum_sq'] + (coordinates[2:] ** 2)
                ds_centroid_dict[min_cluster_ds] = ds_sum_dict[min_cluster_ds]['sum'] / ds_sum_dict[min_cluster_ds]['count']
                ds_deviation_dict[min_cluster_ds] = np.sqrt(
                    ds_sum_dict[min_cluster_ds]['sum_sq'] / ds_sum_dict[min_cluster_ds]['count'] -
                    ds_centroid_dict[min_cluster_ds] ** 2
                )
                ds_indexes_dict[min_cluster_ds].append(int(new_subset[npdata_index][0]))
            elif min_distance_cs is not None and min_distance_cs < threshold:
                # Update CS cluster stats
                cs_sum_dict[min_cluster_cs]['sum'] = cs_sum_dict[min_cluster_cs]['sum'] + coordinates[2:]
                cs_sum_dict[min_cluster_cs]['sum_sq'] = cs_sum_dict[min_cluster_cs]['sum_sq'] + (coordinates[2:] ** 2)
                cs_centroid_dict[min_cluster_cs] = cs_sum_dict[min_cluster_cs]['sum'] / len(cs_indexes_dict[min_cluster_cs])
                cs_deviation_dict[min_cluster_cs] = np.sqrt(
                    cs_sum_dict[min_cluster_cs]['sum_sq'] / len(cs_indexes_dict[min_cluster_cs]) -
                    cs_centroid_dict[min_cluster_cs] ** 2
                )
                cs_indexes_dict[min_cluster_cs].append(int(new_subset[npdata_index][0]))
            else:
                # Assign coordinates to RS
                rs.append(int(new_subset[npdata_index][0]))
                rs_data.append(new_subset[npdata_index][2:])


        if len(rs_data) >= input_clusters * 5:
            kmeans_rs = KMeans(n_clusters=input_clusters * 5, random_state=42).fit(np.array(rs_data))
            rs_labels = kmeans_rs.labels_.tolist()

            new_rs = []
            new_rs_data = []

            rs_label_counts = []
            for i in range(input_clusters * 5):
                rs_label_counts.append(rs_labels.count(i))

            for cluster_id in range(input_clusters * 5):
                cluster_indices = []
                for i, x in enumerate(rs_labels):
                    if x == cluster_id:
                        cluster_indices.append(i)

                cluster_points = []
                for i in cluster_indices:
                    cluster_points.append(rs_data[i])

                original_indices = []
                for i in cluster_indices:
                    original_indices.append(rs[i])

                if rs_label_counts[cluster_id] > 1:  # Cluster with more than one point forms CS
                    size = len(cluster_points)

                    sum_ = []
                    for x in zip(*cluster_points):
                        sum_.append(sum(x))

                    sum_sq = []
                    for col in zip(*cluster_points):
                        sum_sq.append(sum(x**2 for x in col))

                    cs_sum_dict[cluster_id] = {
                        'count': size,
                        'sum': sum_,
                        'sum_sq': sum_sq
                    }
                    cs_indexes_dict[cluster_id] = original_indices

                    cs_centroid_dict[cluster_id] = []
                    for s in sum_:
                        cs_centroid_dict[cluster_id].append(s/size)

                    cs_deviation_dict[cluster_id] = []
                    for s, c in zip(sum_sq, cs_centroid_dict[cluster_id]):
                        cs_deviation_dict[cluster_id].append(((s/size) - (c ** 2)) ** 0.5)

                else:
                    # updating rs 
                    new_rs.extend(original_indices)
                    new_rs_data.extend(cluster_points)

            # updating rs and rs_clsuters
            rs = new_rs
            rs_data = new_rs_data



        cs_clusters = list(cs_centroid_dict.keys())  # Create a copy of the keys

        for cluster_1, cluster_2 in combinations(cs_clusters, 2):
            if cluster_1 in cs_centroid_dict and cluster_2 in cs_centroid_dict:
                # Calculate Mahalanobis Distance
                mahalanobis_dist = compute_merge_distance(
                    cs_centroid_dict[cluster_1], cs_centroid_dict[cluster_2],
                    cs_deviation_dict[cluster_1], cs_deviation_dict[cluster_2]
                )

                # Merge clusters if Mahalanobis Distance < threshold
                if mahalanobis_dist < threshold:
                    # Update CS statistics
                    combined_size = cs_sum_dict[cluster_1]['count'] + cs_sum_dict[cluster_2]['count']
                    combined_sum = [s1 + s2 for s1, s2 in zip(cs_sum_dict[cluster_1]['sum'], cs_sum_dict[cluster_2]['sum'])]
                    combined_sum_sq = [sq1 + sq2 for sq1, sq2 in zip(cs_sum_dict[cluster_1]['sum_sq'], cs_sum_dict[cluster_2]['sum_sq'])]

                    cs_sum_dict[cluster_1] = {
                        'count': combined_size,
                        'sum': combined_sum,
                        'sum_sq': combined_sum_sq
                    }
                    cs_indexes_dict[cluster_1].extend(cs_indexes_dict[cluster_2])

                    # Recalculate centroid and deviation
                    cs_centroid_dict[cluster_1] = [s / combined_size for s in combined_sum]
                    cs_deviation_dict[cluster_1] = [np.sqrt(sq / combined_size - c ** 2) for sq, c in zip(combined_sum_sq, cs_centroid_dict[cluster_1])]

                    # Delete merged cluster
                    del cs_sum_dict[cluster_2]
                    del cs_indexes_dict[cluster_2]
                    del cs_centroid_dict[cluster_2]
                    del cs_deviation_dict[cluster_2]

        
        # If this is the last run (after the last chunk of data), merge CS clusters with DS clusters
        if partition == len(npdata_partitions) - 1:
            for cs_cluster in cs_centroid_dict:
                min_distance = 1e1000000
                min_cluster_ds = -1

                # Calculate Mahalanobis Distance between CS cluster and each DS cluster
                for ds_cluster in ds_centroid_dict.keys():
                    mahalanobis_distance = compute_merge_distance(cs_centroid_dict[cs_cluster], ds_centroid_dict[ds_cluster],cs_deviation_dict[cs_cluster], ds_deviation_dict[ds_cluster])

                    if mahalanobis_distance < min_distance:
                        min_distance = mahalanobis_distance
                        min_cluster_ds = ds_cluster

                # Merge CS cluster with nearest DS cluster if Mahalanobis Distance < threshold
                if min_cluster_ds != -1 and min_distance < threshold:
                    # Update DS statistics
                    ds_sum_dict[min_cluster_ds]['count'] = ds_sum_dict[min_cluster_ds]['count']+cs_sum_dict[cs_cluster]['count']
                    ds_sum_dict[min_cluster_ds]['sum'] = ds_sum_dict[min_cluster_ds]['sum']+cs_sum_dict[cs_cluster]['sum']
                    ds_sum_dict[min_cluster_ds]['sum_sq'] = ds_sum_dict[min_cluster_ds]['sum_sq']+cs_sum_dict[cs_cluster]['sum_sq']

                    # Recalculate centroid and deviation for DS cluster
                    ds_centroid_dict[min_cluster_ds] = [s / ds_sum_dict[min_cluster_ds]['count'] for s in ds_sum_dict[min_cluster_ds]['sum']]
                    ds_deviation_dict[min_cluster_ds] = [np.sqrt(sq / ds_sum_dict[min_cluster_ds]['count'] - c ** 2) for sq, c in zip(ds_sum_dict[min_cluster_ds]['sum_sq'], ds_centroid_dict[min_cluster_ds])]

                    # Delete merged CS cluster
                    del cs_sum_dict[cs_cluster]
                    del cs_indexes_dict[cs_cluster]
                    del cs_centroid_dict[cs_cluster]
                    del cs_deviation_dict[cs_cluster]



        write_intermediate_results(round_count, ds_sum_dict, cs_sum_dict, rs, output_path)
    # The clustering results
    clustering_results = {}

    # Assign DS clusters to the results
    for cluster_id, indices in ds_indexes_dict.items():
        for index in indices:
            clustering_results[index] = cluster_id

    # Assign CS clusters to the results
    for cluster_id, indices in cs_indexes_dict.items():
        for index in indices:
            clustering_results[index] = cluster_id

    # Assign RS points to the results
    for index in rs:
        clustering_results[index] = -1

    # Sort the results by data point index
    sorted_results = sorted(clustering_results.items())

    # Append sorted clustering results to the existing file
    with open(output_path, "a") as file:
        file.write("\nThe clustering results:\n")
        for index, result in sorted_results:
            file.write(f"{index},{result}\n")    


    time_end = time.time()
    duration= time_end-time_start
    final_time = print('Duration:',duration)




