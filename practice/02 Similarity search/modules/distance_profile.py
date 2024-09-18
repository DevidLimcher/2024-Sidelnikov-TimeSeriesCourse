import numpy as np

from modules.utils import z_normalize
from modules.metrics import ED_distance, norm_ED_distance


def brute_force(ts: np.ndarray, query: np.ndarray, is_normalize: bool = True) -> np.ndarray:
    """
    Calculate the distance profile using the brute force algorithm

    Parameters
    ----------
    ts: time series
    query: query, shorter than time series
    is_normalize: normalize or not time series and query

    Returns
    -------
    dist_profile: distance profile between query and time series
    """

    n = len(ts)  # Длина временного ряда
    m = len(query)  # Длина запроса
    dist_profile = np.zeros(n - m + 1)  # Инициализация массива для профиля расстояний
    
    if is_normalize:  # Если требуется нормализация
        query = z_normalize(query)  # Нормализовать запрос
    
    for i in range(n - m + 1):  # Проход по всем подпоследовательностям временного ряда
        subseq = ts[i:i + m]  # Выделение подпоследовательности временного ряда
        if is_normalize:  # Если требуется нормализация
            subseq = z_normalize(subseq)  # Нормализовать подпоследовательность
        
        dist_profile[i] = np.linalg.norm(query - subseq)  # Вычисление евклидова расстояния между запросом и подпоследовательностью
    
    return dist_profile  # Возврат профиля расстояний
