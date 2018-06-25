"""Evaluation metrics, taken from the file ML_benchmarks.py provided by the competition organisers
and available in the Github repo
https://github.com/M4Competition/M4-methods
I changed the factor of 2 in SMAPE to 200 to be consistent with the R code.
"""
import numpy as np

def smape(a, b):
    """
    Calculates sMAPE

    :param a: actual values
    :param b: predicted values
    :return: sMAPE
    """
    a = np.reshape(a, (-1,))
    b = np.reshape(b, (-1,))
    return np.mean(200.0 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()   # BT changed 2 to 200


def mase(insample, y_test, y_hat_test, freq):
    """
    Calculates MAsE

    :param insample: insample data
    :param y_test: out of sample target values
    :param y_hat_test: predicted values
    :param freq: data frequency
    :return:
    """
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])

    masep = np.mean(abs(insample[freq:] - y_hat_naive))

    return np.mean(abs(y_test - y_hat_test)) / masep
