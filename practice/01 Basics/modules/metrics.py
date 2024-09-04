import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """

    # Сначала убедимся, что длины временных рядов совпадают
    if len(ts1) != len(ts2):
        raise ValueError("The two time series must have the same length")
    
    # Рассчитаем евклидово расстояние
    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))

    return ed_dist

def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2
    """

    # Сначала убедимся, что длины временных рядов совпадают
    if len(ts1) != len(ts2):
        raise ValueError("The two time series must have the same length")
    
    n = len(ts1)  # Длина временных рядов

    # Вычисляем средние значения и стандартные отклонения
    mu_T1, mu_T2 = np.mean(ts1), np.mean(ts2)
    sigma_T1, sigma_T2 = np.std(ts1), np.std(ts2)

    # Проверим на нулевые стандартные отклонения
    if sigma_T1 == 0 or sigma_T2 == 0:
        raise ValueError("Standard deviation of one of the time series is zero")

    # Вычисляем скалярное произведение
    dot_product = np.dot(ts1, ts2)

    # Формула для нормализованного евклидова расстояния
    norm_ed_dist = np.sqrt(abs(2 * n * (1 - (dot_product - n * mu_T1 * mu_T2) / (n * sigma_T1 * sigma_T2))))

    return norm_ed_dist


import numpy as np

def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: np.ndarray
        First time series
    ts2: np.ndarray
        Second time series
    r: float
        Warping window size (a fraction of the time series length)
    
    Returns
    -------
    dtw_dist: float
        DTW distance between ts1 and ts2
    """
    
    n, m = len(ts1), len(ts2)
    r = max(int(r * max(n, m)), 1)  # Convert window size to an integer
    
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(max(1, i - r), min(m, i + r) + 1):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    # вставка
                                          dtw_matrix[i, j - 1],    # удаление
                                          dtw_matrix[i - 1, j - 1]) # соответствие
    
    dtw_dist = dtw_matrix[n, m]
    return dtw_dist

