import numpy as np
import math

from modules.bestmatch import UCR_DTW, topK_match
import mass_ts as mts
from scipy.stats import zscore 


default_match_alg_params = {
    'UCR-DTW': {
        'topK': 3,
        'r': 0.05,
        'excl_zone_frac': 1,
        'is_normalize': True
    },
    'MASS': {
        'topK': 3,
        'excl_zone_frac': 1
    }
}


class BestMatchPredictor:
    """
    Predictor based on best match algorithm (UCR-DTW or MASS)
    """
    
    def __init__(self, h: int = 1, match_alg: str = 'UCR-DTW', match_alg_params: dict | None = None, aggr_func: str = 'average') -> None:
        """ 
        Constructor of class BestMatchPredictor

        Parameters
        ----------
        h: prediction horizon
        match_algorithm: name of the best match algorithm ('UCR-DTW' or 'MASS')
        match_algorithm_params: input parameters for the best match algorithm
        aggr_func: aggregate function (average, median)
        """
        self.h = h
        self.match_alg = match_alg
        self.match_alg_params = match_alg_params or {}
        self.agg_func = aggr_func

    def _calculate_predict_values(self, topK_subs_predict_values: np.ndarray) -> np.ndarray:
        """
        Calculate the future values of the time series using the aggregate function
        Parameters
        ----------
        topK_subs_predict_values: values of time series after topK subsequences
        Returns
        -------
        predict_values: prediction values
        """
        if self.agg_func == 'average':
            predict_values = np.mean(topK_subs_predict_values, axis=0).round()
        elif self.agg_func == 'median':
            predict_values = np.median(topK_subs_predict_values, axis=0).round()
        else:
            raise NotImplementedError(f"Agg function {self.agg_func} not implemented")
        
        return predict_values

    def _ucr_dtw(self, ts: np.ndarray, query: np.ndarray) -> np.ndarray:
        """
        Apply UCR-DTW algorithm to find topK subsequences similar to query
        Parameters
        ----------
        ts: time series for search
        query: query subsequence
        Returns
        -------
        topK_indices: indices of topK similar subsequences
        """
        # Apply z-normalization
        if self.match_alg_params.get('is_normalize', True):
            ts = zscore(ts)
            query = zscore(query)
        
        r = self.match_alg_params.get('r', 0.1)
        topK = self.match_alg_params.get('topK', 5)
        excl_zone_frac = self.match_alg_params.get('excl_zone_frac', 1)

        # Apply UCR-DTW search (replace with actual implementation)
        # Here would be UCR-DTW computation and similarity scoring
        
        topK_indices = np.random.choice(len(ts) - len(query), topK)  # Placeholder for actual UCR-DTW indices
        return topK_indices

    def _mass(self, ts: np.ndarray, query: np.ndarray) -> np.ndarray:
        """
        Apply MASS algorithm to find topK subsequences similar to query
        Parameters
        ----------
        ts: time series for search
        query: query subsequence
        Returns
        -------
        topK_indices: indices of topK similar subsequences
        """
        # Apply z-normalization if needed
        if self.match_alg_params.get('is_normalize', True):
            ts = zscore(ts)
            query = zscore(query)
        
        topK = self.match_alg_params.get('topK', 5)
        excl_zone_frac = self.match_alg_params.get('excl_zone_frac', 1)

        # Apply MASS search (replace with actual implementation)
        # Here would be MASS computation and similarity scoring
        
        topK_indices = np.random.choice(len(ts) - len(query), topK)  # Placeholder for actual MASS indices
        return topK_indices

    def predict(self, ts: np.ndarray, query: np.ndarray) -> np.ndarray:
        """
        Predict time series at future horizon
        Parameters
        ----------
        ts: time series
        query: query, shorter than time series
        Returns
        -------
        predict_values: prediction values
        """
        if self.match_alg == 'UCR-DTW':
            topK_indices = self._ucr_dtw(ts, query)
        elif self.match_alg == 'MASS':
            topK_indices = self._mass(ts, query)
        else:
            raise NotImplementedError(f"Algorithm {self.match_alg} not implemented")
        
        # Extract the values after the found subsequences for prediction
        topK_subs_predict_values = np.array([ts[idx + len(query): idx + len(query) + self.h] for idx in topK_indices])

        # Calculate predicted values using the aggregate function
        predict_values = self._calculate_predict_values(topK_subs_predict_values)
        
        return predict_values