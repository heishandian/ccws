import pandas as pd
import numpy as np


def topsis(data, weight=[0, 0.5, 0.5]):
    data = data / np.sqrt((data ** 2).sum())
    Z = pd.DataFrame([data.min(), data.max()], index=['N', 'P'])
    weight = entropyWeight(data) if weight is None else np.array(weight)
    Result = data.copy()
    Result['P'] = np.sqrt(((data - Z.loc['P']) ** 2 * weight).sum(axis=1))
    Result['N'] = np.sqrt(((data - Z.loc['N']) ** 2 * weight).sum(axis=1))
    Result['P'] = Result['N'] / (Result['N'] + Result['P'])
    Result['S'] = Result.rank(ascending=False)['P']
    return Result, Z, weight


def entropyWeight(data):
    data = np.array(data)
    P = data / data.sum(axis=0)
    E = np.nansum(-P * np.log(P) / np.log(len(data)), axis=0)
    return (1 - E) / (1 - E).sum()
