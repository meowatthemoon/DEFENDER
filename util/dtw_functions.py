from fastdtw import fastdtw
import numpy as np

MAX_LENGTH = 20

def min_same(trajectory, demonstrations) -> float:
    trajectory = trajectory[-MAX_LENGTH:]
    length = trajectory.shape[0]

    dtws = [fastdtw(demonstration[-length:], trajectory)[0] for demonstration in demonstrations]
    return np.min(dtws)

def min_trajectory_window5(trajectory, demonstrations) -> float:
    trajectory = trajectory[-MAX_LENGTH:]

    dtws = [fastdtw(demonstration, trajectory[-5:])[0] for demonstration in demonstrations]
    return np.min(dtws)

def min_trajectory_window10(trajectory, demonstrations) -> float:
    trajectory = trajectory[-MAX_LENGTH:]

    dtws = [fastdtw(demonstration, trajectory[-10:])[0] for demonstration in demonstrations]
    return np.min(dtws)

def mean_both_window5(trajectory, demonstrations) -> float:
    trajectory = trajectory[-MAX_LENGTH:]

    dtws = [fastdtw(demonstration[-5:], trajectory[-5:])[0] for demonstration in demonstrations]
    return np.mean(dtws)

def mean_both_window10(trajectory, demonstrations) -> float:
    trajectory = trajectory[-MAX_LENGTH:]

    dtws = [fastdtw(demonstration[-10:], trajectory[-10:])[0] for demonstration in demonstrations]
    return np.mean(dtws)

def min_demonstration_window5(trajectory, demonstrations) -> float:
    trajectory = trajectory[-MAX_LENGTH:]

    dtws = [fastdtw(demonstration[-5:], trajectory)[0] for demonstration in demonstrations]
    return np.min(dtws)

def mean_demonstration_window5(trajectory, demonstrations) -> float:
    trajectory = trajectory[-MAX_LENGTH:]

    dtws = [fastdtw(demonstration[-5:], trajectory)[0] for demonstration in demonstrations]
    return np.mean(dtws)

def min_demonstration_window10(trajectory, demonstrations) -> float:
    trajectory = trajectory[-MAX_LENGTH:]

    dtws = [fastdtw(demonstration[-10:], trajectory)[0] for demonstration in demonstrations]
    return np.min(dtws)

def mean_demonstration_window10(trajectory, demonstrations) -> float:
    trajectory = trajectory[-MAX_LENGTH:]

    dtws = [fastdtw(demonstration[-10:], trajectory)[0] for demonstration in demonstrations]
    return np.mean(dtws)

def max_trajectory_window10(trajectory, demonstrations) -> float:
    trajectory = trajectory[-MAX_LENGTH:]

    dtws = [fastdtw(demonstration, trajectory[-10:])[0] for demonstration in demonstrations]
    return np.max(dtws)

def get_function_by_key(key : str):
    if key == 'min_same':
        return min_same
    
    if key == 'min_trajectory_window5':
        return min_trajectory_window5

    if key == 'min_trajectory_window10':
        return min_trajectory_window10

    if key == 'mean_both_window5':
        return mean_both_window5
    
    if key == 'mean_both_window10':
        return mean_both_window10
    
    if key == 'min_demonstration_window5':
        return min_demonstration_window5
    
    if key == 'mean_demonstration_window5':
        return mean_demonstration_window5
    
    if key == 'min_demonstration_window10':
        return min_demonstration_window10

    if key == 'mean_demonstration_window10':
        return mean_demonstration_window10
    
    if key == 'max_trajectory_window10':
        return max_trajectory_window10
    
    raise ValueError