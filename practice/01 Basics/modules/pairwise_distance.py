import numpy as np
from modules.metrics import ED_distance, DTW_distance, norm_ED_distance
from modules.utils import z_normalize

class PairwiseDistance:
    """
    Distance matrix between time series 

    Parameters
    ----------
    metric: distance metric between two time series
            Options: {euclidean, dtw}
    is_normalize: normalize or not time series
    """

    def __init__(self, metric: str = 'euclidean', is_normalize: bool = False) -> None:
        self.metric: str = metric
        self.is_normalize: bool = is_normalize

    @property
    def distance_metric(self) -> str:
        """Return the distance metric"""
        norm_str = "normalized " if self.is_normalize else "non-normalized "
        return norm_str + self.metric + " distance"

    def _choose_distance(self):
        """ Choose distance function for calculation of matrix """
        if self.metric == 'euclidean':
            if self.is_normalize:
                return norm_ED_distance
            return ED_distance
        elif self.metric == 'dtw':
            return DTW_distance
        else:
            raise ValueError("Unsupported metric provided: {}".format(self.metric))

    def calculate(self, input_data: np.ndarray) -> np.ndarray:
        """ Calculate distance matrix """
        K = input_data.shape[0]
        matrix_values = np.zeros((K, K))
        dist_func = self._choose_distance()

        for i in range(K):
            for j in range(i, K):
                if self.is_normalize:
                    series_i = z_normalize(input_data[i])
                    series_j = z_normalize(input_data[j])
                else:
                    series_i = input_data[i]
                    series_j = input_data[j]

                dist = dist_func(series_i, series_j)
                matrix_values[i, j] = dist
                matrix_values[j, i] = dist  # Симметрия матрицы

        return matrix_values
