import pandas as pd
import numpy as np
import torch
from loss_transfer import TransferLoss
from pathlib import Path
import time
from sklearn.model_selection import train_test_split
import torch.nn as nn


def load_csv_data(csv_path):
    """
    Load data from a CSV file into a pandas DataFrame.
    """
    df = pd.read_csv(csv_path)
    return df

def segment_distance(v1, v2, loss_type='cosine', input_dim=512):
    # Truncate vectors to the same length
    min_length = min(len(v1), len(v2))
    v1 = v1[:min_length]
    v2 = v2[:min_length]
    
    # Convert to numpy arrays for convenience
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    
    # Normalize the vectors to sum to 1
    v1 /= np.sum(v1) if np.sum(v1) != 0 else 1
    v2 /= np.sum(v2) if np.sum(v2) != 0 else 1
    
    # Compute the Euclidean distance
    euclidean_distance = np.sqrt(np.sum((v1 - v2) ** 2))
    
    return euclidean_distance

def ReconstructSegments(data, end, num_segments, splits):
    segments = []
    segment_indices = []
    current_end = end
    for _ in range(num_segments):
        seg_len = splits[current_end][num_segments]
        segment = data[current_end - seg_len: current_end]
        segments.append(segment)
        # Store the start and end indices of the segment
        segment_indices.append((current_end - seg_len, current_end))
        current_end -= seg_len
        num_segments -= 1
    segments.reverse()
    segment_indices.reverse()
    return segments, segment_indices

def TimeSeriesSegmentation(data, K, lower_bound, upper_bound):
    n = len(data)
    DP = np.full((n + 1, K + 1), float('-inf'))
    splits = np.zeros((n + 1, K + 1), dtype=int)    
    for i in range(0, n+1):
        DP[i][0] = 0
    for i in range(1, n + 1):
        if i % 50 == 0:
            print(f'I: {i}')
        for seg_len in range(lower_bound, min(upper_bound, i) + 1):
            if i - seg_len >= 0:
                curr_segment = data[i - seg_len: i]
                for j in range(1, K + 1):
                    if j == 1:
                        if i == seg_len:
                            DP[i][j] = 0
                            splits[i][j] = seg_len
                        continue
                    segments, segment_indices = ReconstructSegments(data, i - seg_len, j - 1, splits)

                    average_distance = sum(segment_distance(seg, curr_segment) for seg in segments) / len(segments) if len(segments) != 0 else 0
                    
                    if DP[i][j] < DP[i - seg_len][j - 1] + average_distance:
                        DP[i][j] = DP[i - seg_len][j - 1] + average_distance
                        splits[i][j] = seg_len

    optimal_value = DP[n][K]
    optimal_segments, optimal_segments_indices = ReconstructSegments(data, n, K, splits)
    return optimal_value, optimal_segments, optimal_segments_indices


def main(csv_path, num_segments):
    # Load the data
    df = load_csv_data(csv_path)
    df = df.drop(columns=['DateTime'])
    data_array = df.to_numpy()

    
    # Apply TDC to get segments
    _, segments_indices = TimeSeriesSegmentation(data_array, num_segments, 10, 365)
    
def split_data(data, test_size=0.2, val_size=0.1):
    """
    Split data into training, validation, and test sets.

    Parameters:
    - data: Numpy array of the dataset.
    - test_size: Proportion of the dataset to include in the test split.
    - val_size: Proportion of the training dataset to include in the validation split.

    Returns:
    - A tuple containing training, validation, and test data as Numpy arrays.
    """
    # First, split data into training + validation and test sets
    train_val_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
    
    # Calculate adjusted validation size for splitting the training data
    val_size_adjusted = val_size / (1 - test_size)
    
    # Then, split training + validation set into training and validation sets
    train_data, val_data = train_test_split(train_val_data, test_size=val_size_adjusted, shuffle=False)
    
    return train_data, val_data, test_data

def find_and_save_best_segmentation(csv_paths, K_range, output_dir, test_size=0.2, val_size=0.1):
    """
    Find the best segmentation for a range of K values for multiple datasets, split the data into
    training, validation, and test sets, and save the results.

    Parameters:
    - csv_paths: List of paths to CSV files containing the datasets.
    - K_range: Range of K values to try for segmentation.
    - output_dir: Directory to store the segmentation results files.
    - test_size: Proportion of the dataset to include in the test split.
    - val_size: Proportion of the training dataset to include in the validation split.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    for csv_path in csv_paths:
        print(f'Processing file: {csv_path}')
        dataset_name = Path(csv_path).stem
        df = pd.read_csv(csv_path)
        data_array = df.to_numpy()
        
        # Split data into training, validation, and test sets
        train_data, val_data, test_data = split_data(data_array, test_size=test_size, val_size=val_size)
        
        # Use training data for segmentation
        data_for_segmentation = train_data
        
        # Calculate bounds based on training data length
        dataset_len = len(data_for_segmentation)
        lower_bound = int(dataset_len * 0.90) // max(K_range)  # 25% less than average segment length
        upper_bound = int(dataset_len * 1.1) // min(K_range)  # 25% more than average segment length
        
        results_file_path = output_dir_path / f"{dataset_name}_segmentation_results.txt"
        
        with open(results_file_path, 'w') as f:
            f.write("K, Optimal Value, Segments, Time (seconds), Lower Bound, Upper Bound, Test Size, Validation Size\n")
            
            for K in K_range:
                print(f'K = {K}')
                start_time = time.time()
                optimal_value, _, segments_indices = TimeSeriesSegmentation(data_for_segmentation, K, lower_bound, upper_bound)
                end_time = time.time()
                computation_time = end_time - start_time
                
                f.write(f"{K}, {optimal_value}, {segments_indices}, {computation_time:.2f}, {lower_bound}, {upper_bound}, {test_size}, {val_size}\n")


if __name__ == "__main__":
    csv_paths = ['processed_dingling.csv']  # Add your CSV file paths here
    #csv_paths = ['./nq2023_processed.csv']
    K_range = [5,7,10]
    output_dir = './segmentation_results_dingling_pair'
    find_and_save_best_segmentation(csv_paths, K_range, output_dir)

