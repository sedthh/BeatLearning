import numpy as np

from typing import Union, List

    
def tokenize(prediction: Union[list, np.ndarray]) -> Union[List[int], int]:
    """Tokenizes on a single list or batch of lists (see self._tokenize(...) for more info)"""
    if isinstance(prediction, (list, tuple)):
        prediction = np.array(prediction)
    if len(prediction.shape) == 2:
        return np.array([_tokenize(pred) for pred in prediction])
    else:
        return _tokenize(prediction)
    
def _tokenize(prediction: np.ndarray) -> int:
    """Returns a single integer from list of booleans: [1., 0., 1.] --> 5"""
    return int("".join([str(int(val)) for val in prediction]), 2)
    