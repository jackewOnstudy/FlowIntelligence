import numpy as np
from fastdtw import fastdtw
import Levenshtein

def logic_and_distance(x, y):
    # y = np.concatenate((np.zeros(15), y))
    # if len(x) <= len(y):
    #     x = np.pad(x, (0, len(y) - len(x)), 'constant')
    # else:
    #     y = np.pad(y, (0, len(x) - len(y)), 'constant')
    assert len(x) == len(y), "两个序列长度应相同"
    if np.sum(x) == 0 and np.sum(y) == 0:
        return 0        # 因为这里是针对一个小的segment
    return 1 - np.sum(np.logical_and(x, y)) / (np.sum(x) + np.sum(y)) * 2
    

def logic_xonr_distance(x, y):
    # y = np.concatenate((np.zeros(15), y))
    # if len(x) <= len(y):
    #     x = np.pad(x, (0, len(y) - len(x)), 'constant')
    # else:
    #     y = np.pad(y, (0, len(x) - len(y)), 'constant')
    assert len(x) == len(y), "两个序列长度应相同"
    # 最后返回的是：len(x) - (len(x) - np.sum(np.bitwise_xor(x, y)))
    return np.sum(np.bitwise_xor(x, y))