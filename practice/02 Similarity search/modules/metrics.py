import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two time series

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: Euclidean distance between ts1 and ts2
    """

    # Ensure both series are the same length
    if len(ts1) != len(ts2):
        raise ValueError("Time series must be of the same length to compute Euclidean distance.")
    
    # Compute the Euclidean distance
    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))
    
    return ed_dist



def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance between two time series

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2
    """
    
    # Ensure both series are the same length
    if len(ts1) != len(ts2):
        raise ValueError("Time series must be of the same length to compute normalized Euclidean distance.")
    
    # Z-normalize both series
    ts1_norm = (ts1 - np.mean(ts1)) / np.std(ts1)
    ts2_norm = (ts2 - np.mean(ts2)) / np.std(ts2)

    # Compute Euclidean distance for normalized series
    norm_ed_dist = np.sqrt(np.sum((ts1_norm - ts2_norm) ** 2))
    
    return norm_ed_dist



def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size (should be <= min(len(ts1), len(ts2)))
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """
    
    n, m = len(ts1), len(ts2)
    
    # Check if the warping window is valid
    r = max(1, min(r, max(n, m)))  # Ensure r is in a valid range
    
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(max(1, i-r), min(m+1, i+r+1)):  # Ограничиваемся вокруг диагонали
            cost = (ts1[i-1] - ts2[j-1]) ** 2
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # Вниз
                dtw_matrix[i, j-1],    # Влево
                dtw_matrix[i-1, j-1]   # По диагонали
            )

    return np.sqrt(dtw_matrix[n, m])  # Корень из итогового значения


