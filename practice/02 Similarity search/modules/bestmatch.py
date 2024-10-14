import numpy as np
import math
import copy

from modules.my_utils import sliding_window, z_normalize
from modules.metrics import DTW_distance


def apply_exclusion_zone(array: np.ndarray, idx: int, excl_zone: int) -> np.ndarray:
    """
    Apply an exclusion zone to an array (inplace)
    
    Parameters
    ----------
    array: the array to apply the exclusion zone to
    idx: the index around which the window should be centered
    excl_zone: size of the exclusion zone
    
    Returns
    -------
    array: the array which is applied the exclusion zone
    """

    zone_start = max(0, idx - excl_zone)
    zone_stop = min(array.shape[-1], idx + excl_zone)
    array[zone_start : zone_stop + 1] = np.inf

    return array


def topK_match(dist_profile: np.ndarray, excl_zone: int, topK: int = 3, max_distance: float = np.inf) -> dict:
    """
    Search the topK match subsequences based on distance profile
    
    Parameters
    ----------
    dist_profile: distances between query and subsequences of time series
    excl_zone: size of the exclusion zone
    topK: count of the best match subsequences
    max_distance: maximum distance between query and a subsequence `S` for `S` to be considered a match
    
    Returns
    -------
    topK_match_results: dictionary containing results of algorithm
    """

    candidates = np.argsort(dist_profile)  # Получить индексы, сортированные по возрастанию расстояния
    results = {'indices': [], 'distances': []}  # Инициализация словаря для результатов

    for idx in candidates:  # Перебор кандидатов
        if len(results['indices']) >= topK:  # Если найдено достаточно подпоследовательностей, прервать цикл
            break
        # Проверка, что текущий индекс не в зоне исключения относительно уже найденных индексов
        if all(abs(idx - x) > excl_zone for x in results['indices']):  
            results['indices'].append(idx)  # Добавить индекс в результаты
            results['distances'].append(dist_profile[idx])  # Добавить соответствующее расстояние в результаты

    return results  # Вернуть результаты


class BestMatchFinder:
    """
    Base Best Match Finder
    
    Parameters
    ----------
    excl_zone_frac: exclusion zone fraction
    topK: number of the best match subsequences
    is_normalize: z-normalize or not subsequences before computing distances
    r: warping window size
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05) -> None:
        """ 
        Constructor of class BestMatchFinder
        """

        self.excl_zone_frac: float = excl_zone_frac
        self.topK: int = topK
        self.is_normalize: bool = is_normalize
        self.r: float = r


    def _calculate_excl_zone(self, m: int) -> int:
        """
        Calculate the exclusion zone
        
        Parameters
        ----------
        m: length of subsequence
        
        Returns
        -------
        excl_zone: exclusion zone
        """

        excl_zone = math.ceil(m * self.excl_zone_frac)

        return excl_zone


    def perform(self):

        raise NotImplementedError


class NaiveBestMatchFinder(BestMatchFinder): 
    """ 
    Naive Best Match Finder 
    """ 
 
    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05): 
        super().__init__(excl_zone_frac, topK, is_normalize, r) 
        """  
        Constructor of class NaiveBestMatchFinder 
        """ 
 
 
    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict: 
        """ 
        Search subsequences in a time series that most closely match the query using the naive algorithm 
         
        Parameters 
        ---------- 
        ts_data: time series 
        query: query, shorter than time series 
 
        Returns 
        ------- 
        best_match: dictionary containing results of the naive algorithm 
        """ 
 
        query = copy.deepcopy(query) 
        if (len(ts_data.shape) != 2): # time series set 
            ts_data = sliding_window(ts_data, len(query)) 
 
        N, m = ts_data.shape 
        excl_zone = self._calculate_excl_zone(m) 
 
        dist_profile = np.ones((N,))*np.inf 
        bsf = np.inf 
 
        bestmatch = { 
            'indices' : [], 
            'distance' : [] 
        } 
         
        # Normalize query if needed 
        if self.is_normalize: 
            query = z_normalize(query) 
 
        # Compute distance profile for each subsequence 
        for i in range(N): 
            subsequence = ts_data[i] 
            if self.is_normalize: 
                subsequence = z_normalize(subsequence) 
            distance = DTW_distance(subsequence, query, r=self.r) 
            dist_profile[i] = distance 
 
        # Find topK matches 
        topK_results = topK_match(dist_profile, excl_zone, self.topK, max_distance=bsf) 
 
        bestmatch['indices'] = topK_results['indices'] 
        bestmatch['distance'] = topK_results['distances'] 
 
 
 
        return bestmatch



class UCR_DTW(BestMatchFinder):
    """
    UCR-DTW Match Finder
    
    Additional parameters
    ----------
    not_pruned_num: number of non-pruned subsequences
    lb_Kim_num: number of subsequences that pruned by LB_Kim bounding
    lb_KeoghQC_num: number of subsequences that pruned by LB_KeoghQC bounding
    lb_KeoghCQ_num: number of subsequences that pruned by LB_KeoghCQ bounding
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        # Передаем параметры в родительский класс
        super().__init__(excl_zone_frac, topK, is_normalize, r)
        """ 
        Constructor of class UCR_DTW
        """        

        # Инициализируем дополнительные атрибуты
        self.not_pruned_num = 0
        self.lb_Kim_num = 0
        self.lb_KeoghQC_num = 0
        self.lb_KeoghCQ_num = 0


    def _LB_Kim(self, subs1: np.ndarray, subs2: np.ndarray) -> float:
        """
        Compute LB_Kim lower bound between two subsequences.
        
        Parameters
        ----------
        subs1: the first subsequence
        subs2: the second subsequence
        
        Returns
        -------
        lb_Kim: LB_Kim lower bound
        """

        lb_Kim = 0
        length = len(subs1)
        
        # Проверяем первую и последнюю точку
        lb_Kim += (subs1[0] - subs2[0])**2
        lb_Kim += (subs1[-1] - subs2[-1])**2
        
        # Проверка центральных точек на 10%, 50%, 90% длины
        for frac in [0.1, 0.5, 0.9]:
            idx = int(length * frac)
            lb_Kim += (subs1[idx] - subs2[idx])**2

        return np.sqrt(lb_Kim)






    def _LB_Keogh(self, subs1: np.ndarray, subs2: np.ndarray, r: float, n: int) -> float:
        """
        Compute LB_Keogh lower bound between two subsequences.
        
        Parameters
        ----------
        subs1: the first subsequence
        subs2: the second subsequence
        r: warping window size
        n: length of the second subsequence (ts_data)

        Returns
        -------
        lb_Keogh: LB_Keogh lower bound
        """

        lb_Keogh = 0
        m = len(subs1)
        
        # Верхняя и нижняя границы для warping window, преобразуем генераторы в списки
        upper_bound = [np.max(subs2[max(0, i - int(r)): min(n, i + int(r) + 1)]) for i in range(m)]
        lower_bound = [np.min(subs2[max(0, i - int(r)): min(n, i + int(r) + 1)]) for i in range(m)]

        for i in range(m):
            if subs1[i] > upper_bound[i]:
                lb_Keogh += (subs1[i] - upper_bound[i]) ** 2
            elif subs1[i] < lower_bound[i]:
                lb_Keogh += (subs1[i] - lower_bound[i]) ** 2

        
        return np.sqrt(lb_Keogh)



    def _LB_Keogh_EC(self, query: np.ndarray, subsequence: np.ndarray, r: float, n: int) -> float:
        """ Compute LB_KeoghEC as the reverse version of LB_KeoghEQ """
        return self._LB_Keogh(subsequence, query, r, n)  # Передаем n

    

    def get_statistics(self) -> dict:
        """
        Return statistics on the number of pruned and non-pruned subsequences of a time series   
        
        Returns
        -------
            dictionary containing statistics
        """

        statistics = {
            'not_pruned_num': self.not_pruned_num,
            'lb_Kim_num': self.lb_Kim_num,
            'lb_KeoghCQ_num': self.lb_KeoghCQ_num,
            'lb_KeoghQC_num': self.lb_KeoghQC_num
        }

        return statistics


    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        """
        Поиск подпоследовательностей временного ряда, наиболее похожих на запрос, с использованием UCR-DTW.
        
        Параметры
        ----------
        ts_data: np.ndarray
            Временной ряд.
        query: np.ndarray
            Запрос, короче чем временной ряд.

        Возвращает
        -------
        dict
            Словарь, содержащий результаты выполнения алгоритма UCR-DTW.
        """
        query = copy.deepcopy(query)
        
        if self.is_normalize:
            query = z_normalize(query)

        n = len(ts_data)
        m = len(query)
        
        excl_zone = self._calculate_excl_zone(m)
        
        dist_profile = np.full(n - m + 1, np.inf)
        bsf = np.inf  # best-so-far distance
        
        bestmatch = {
            'indices': [], 
            'distances': []
        }

        for i in range(n - m + 1):
            subsequence = ts_data[i:i + m]

            if self.is_normalize:
                subsequence = z_normalize(subsequence)

            # Применение LB_Kim
            lb_Kim_dist = self._LB_Kim(query, subsequence)
            if lb_Kim_dist >= bsf:
                self.lb_Kim_num += 1
                continue

            # Применение LB_Keogh
            lb_Keogh_dist = self._LB_Keogh(query, subsequence, self.r, n)
            if lb_Keogh_dist >= bsf:
                self.lb_KeoghQC_num += 1
                continue

            # Применение LB_Keogh_EC
            lb_Keogh_EC_dist = self._LB_Keogh_EC(query, subsequence, self.r, n)
            if lb_Keogh_EC_dist >= bsf:
                self.lb_KeoghCQ_num += 1
                continue

            # Рассчитываем полное DTW расстояние
            dist = DTW_distance(query, subsequence)

            if np.isinf(dist):
                print(f"Distance calculation resulted in inf for subsequence starting at index {i}. Ignoring this subsequence.")
                continue

            if dist < bsf:
                bestmatch['indices'].append(i)
                bestmatch['distances'].append(dist)

                if len(bestmatch['indices']) >= self.topK:
                    bsf = max(bestmatch['distances'])
                    sorted_indices = np.argsort(bestmatch['distances'])[:self.topK]
                    bestmatch['indices'] = [bestmatch['indices'][idx] for idx in sorted_indices]
                    bestmatch['distances'] = [bestmatch['distances'][idx] for idx in sorted_indices]

        self.not_pruned_num = n - m + 1 - (self.lb_Kim_num + self.lb_KeoghQC_num + self.lb_KeoghCQ_num)

        return bestmatch

