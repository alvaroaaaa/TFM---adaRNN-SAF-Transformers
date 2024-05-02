import numpy as np
import pandas as pd
from random import random, randint
from math import exp
from itertools import combinations
from sklearn.model_selection import train_test_split

# Placeholder for your custom distance function
def custom_distance(v1, v2):
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

def compute_dissimilarity(data, segments):
    # Calculate distances between all unique pairs of segments
    distances = []
    for (start1, end1), (start2, end2) in combinations(segments, 2):
        segment1 = data[start1:end1]  # add 1 because end index is inclusive
        segment2 = data[start2:end2]  # add 1 because end index is inclusive
        dist = custom_distance(segment1, segment2)
        distances.append(dist)

    # Compute the average distance
    K = len(segments)
    average_distance = sum(distances) / K

    return average_distance
import numpy as np

def generate_initial_solution(K, data_length, min_length, max_length):
    segments_indices = []
    start = 0

    # Generate K - 1 segments ensuring each one adheres to the min and max length constraints
    for _ in range(K - 1):
        if start + min_length >= data_length:
            break  # Break if adding another minimum length segment exceeds data length
        # Ensure the end of the segment doesn't exceed data_length and respects the max_length
        end = min(start + max_length, data_length - 1)
        # Random end point for this segment
        if end > start + min_length:
            end = np.random.randint(start + min_length, end + 1)
        segments_indices.append([start, end])
        start = end  # Start next segment where this one ended
    
    # Ensure the last segment captures all remaining data
    if start < data_length:
        segments_indices.append([start, data_length])

    return segments_indices


def generate_neighbor(current_solution, min_length, max_length, data_length):
    if len(current_solution) < 2:
        return current_solution

    neighbor = current_solution.copy()
    # Select a random segment to adjust, not including the last segment
    segment_to_adjust = randint(0, len(neighbor) - 2)

    # Current start and end of the selected segment
    current_start = neighbor[segment_to_adjust][0]
    current_end = neighbor[segment_to_adjust][1]

    # Determine the boundaries for the new end of the selected segment
    if segment_to_adjust == 0:
        # For the first segment, we can adjust the end only
        max_end = min(data_length - 1, current_start + max_length)
        min_end = max(current_start + min_length, 1)
    else:
        # For middle segments, adjust keeping neighbors in mind
        max_end = min(neighbor[segment_to_adjust + 1][1] - min_length, current_start + max_length)
        min_end = max(neighbor[segment_to_adjust - 1][1] + min_length, current_start + min_length)

    if min_end < max_end:
        new_end = randint(min_end, max_end)
        neighbor[segment_to_adjust][1] = new_end
        if segment_to_adjust < len(neighbor) - 1:
            neighbor[segment_to_adjust + 1][0] = new_end

    return neighbor

def simulated_annealing(data, K, min_length, max_length):
    best_solution = None
    best_score = -np.inf
    initial_temperature = 1.0
    cooling_rate = 0.95
    num_iterations = 1000

    for trial in range(10):
        print(f'Trial: {trial}')
        initial_solution = generate_initial_solution(K, len(data), min_length, max_length)
        if len(initial_solution) < 3:
            continue

        current_solution = initial_solution
        current_score = compute_dissimilarity(data, current_solution)

        for i in range(num_iterations):
            temperature = max(initial_temperature * (cooling_rate ** i), 0.01)
            neighbor = generate_neighbor(current_solution, min_length, max_length, len(data))
            neighbor_score = compute_dissimilarity(data, neighbor)

            if neighbor_score > current_score or random() < exp((neighbor_score - current_score) / temperature):
                current_solution = neighbor
                current_score = neighbor_score

            if current_score > best_score:
                best_score = current_score
                best_solution = current_solution

    return best_score, best_solution


# Example usage:
data = pd.read_csv('processed_tsla.csv')
data_array = data.to_numpy()
train_data, val_data, test_data = split_data(data_array, test_size=0.2, val_size=0.1)
K_max = 10  # Maximum number of segments
min_length = int(len(train_data) * 0.90) // K_max  # 25% less than average segment length
max_length = int(len(train_data) * 1.1) // K_max  # Maximum length of a segment
print(min_length)
print(max_length)
best_score, best_solution = simulated_annealing(train_data, K_max, min_length, max_length)

print("Best Score:", best_score)
print("Best Solution:", best_solution)

