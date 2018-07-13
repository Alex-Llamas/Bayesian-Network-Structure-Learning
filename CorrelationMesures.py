__author__ = 'AlexLlamas'
import numpy as np
import math
from typing import Dict
from minepy import MINE

# -------------------------------------- Mutual information -----------------------------------------------------------
def mutual_info(x, y, num_bin_x, num_bin_y):
    n = x.size
    h, x_edges, y_edges = np.histogram2d(x, y, bins=[num_bin_x, num_bin_y])
    h = np.array(h)  # type: Dict[Tuple[int, int]]
    sum_x = np.sum(h, axis=0)  # type: list
    sum_y = np.sum(h, axis=1)  # type: list
    mi = 0.0
    for i in range(0, num_bin_x):
        for j in range(0, num_bin_y):
            if h[i, j] != 0:
                mi += (h[i, j] /
                       float(n)) * math.log10((h[i, j] / float(n)) / ((sum_y[i] / float(n)) * (sum_x[j] / float(n))))
    return mi


# ---------------------------------------------------------------------------------------------------------------------


# --------------------------------------------- Entropy ---------------------------------------------------------------
def entropy(x, num_bin_x):
    n = x.size
    h, edges = np.histogram(x, num_bin_x)  # type: list
    e = 0.0
    for i in range(0, num_bin_x):
        if h[i] != 0:
            e -= (h[i] / float(n)) * math.log10(h[i] / float(n))
    return e


# ---------------------------------------------------------------------------------------------------------------------


# --------------------------------------- Normalized mutual information -----------------------------------------------
def norm_mi(x, y, num_bin_x, num_bin_y):
    mi = mutual_info(x, y, num_bin_x, num_bin_y)
    ex = entropy(x, num_bin_x)
    ey = entropy(y, num_bin_y)
    r = mi / float(ex + ey)
    r_max = min(ex, ey) / float(ex + ey)
    return r / r_max


# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------- Pearson correlation --------------------------------------------------
def pearson_corr(x, y):
    """
    x :type np.ndarray
    y :type np.ndarray

    :return rxy :type float
    """
    n = x.size
    xm = np.mean(x)
    ym = np.mean(y)
    numerator = 0.0
    for i in range(0, n):
        numerator += (x[i] - xm) * (y[i] - ym)
    sx = 0.0
    for i in range(0, n):
        exp = float(x[i] - xm)
        sx += math.pow(exp, 2)
    sy = 0.0
    for i in range(0, n):
        exp = float(y[i] - ym)
        sy += math.pow(exp, 2)
    rxy = numerator / math.sqrt(sx * sy)
    return rxy


# ---------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------ Entropy X|Y --------------------------------------------------------
def entropy_x_y(x, y, num_bin_x, num_bin_y):
    n = x.size
    h, x_edges, y_edges = np.histogram2d(x, y, bins=[num_bin_x, num_bin_y])
    h = np.array(h)  # type: Dict[Tuple[int, int]]
    sum_x = np.sum(h, axis=0)  # type: list
    exy = 0.0
    for i in range(0, num_bin_x):
        for j in range(0, num_bin_y):
            if h[i, j] != 0:
                exy += (h[i, j] / float(n)) * math.log10((sum_x[j] / float(n)) / (h[i, j] / float(n)))
    return exy


# ---------------------------------------------------------------------------------------------------------------------


# -------------------------------- Mutual information based on Entropy ------------------------------------------------
def mi_entropy(x, y, num_bin_x, num_bin_y):
    mie = entropy(x, num_bin_x) - entropy_x_y(x, y, num_bin_x, num_bin_y)
    return mie


# ---------------------------------------------------------------------------------------------------------------------


# ----------------------------------- Uniform Dependency --------------------------------------------------------------
# -------------------- Information in Y of X : dependency of X over Y -------------------------------------------------
def ud_ixy(x, y, num_bin_x, num_bin_y):
    h, x_edges, y_edges = np.histogram2d(x, y, bins=[num_bin_x, num_bin_y])
    h = np.array(h)  # type: Dict[Tuple[int, int]]
    sum_x = np.sum(h, axis=0)  # type: list
    norm = float(num_bin_x) / ((num_bin_x - 1) * 2 * num_bin_y)
    ixy = 0.0
    for i in range(0, num_bin_x):
        for j in range(0, num_bin_y):
            if sum_x[j] != 0:
                ixy += math.fabs((1 / float(num_bin_x)) - (h[i, j] / float(sum_x[j])))
    return norm * ixy


# ---------------------------------------------------------------------------------------------------------------------


# --------------------------------------- Uniform mutual dependency ---------------------------------------------------
def umd(x, y, num_bin_x, num_bin_y):
    ixy = ud_ixy(x, y, num_bin_x, num_bin_y)
    iyx = ud_ixy(y, x, num_bin_y, num_bin_x)
    return (ixy + iyx) / 2


# ---------------------------------------------------------------------------------------------------------------------


# ------------------------------------------ Distance Covariance ------------------------------------------------------
def d_cov(x, y):
    n = x.size

    # --- first for a  -----------------
    a_kp = np.zeros(n)  # type: list
    for k in range(0, n):
        for l in range(0, n):
            a_kp[k] += math.fabs(x[k] - x[l])
        a_kp[k] *= (1 / float(n))

    a_pl = np.zeros(n)  # type: list
    for l in range(0, n):
        for k in range(0, n):
            a_pl[l] += math.fabs(x[k] - x[l])
        a_pl[l] *= 1 / float(n)

    a_pp = 0.0
    for k in range(0, n):
        for l in range(0, n):
            a_pp += math.fabs(x[k] - x[l])
    a_pp *= 1 / math.pow(n, 2)

    a = np.zeros((n, n))  # type: Dict[Tuple[int, int]]
    for k in range(0, n):
        for l in range(0, n):
            a[k, l] = math.fabs(x[k] - x[l]) - a_kp[k] - a_pl[l] + a_pp

            # --- now with b  -----------------
    b_kp = np.zeros(n)  # type: list
    for k in range(0, n):
        for l in range(0, n):
            b_kp[k] += math.fabs(y[k] - y[l])
        b_kp[k] *= (1 / float(n))

    b_pl = np.zeros(n)  # type: list
    for l in range(0, n):
        for k in range(0, n):
            b_pl[l] += math.fabs(y[k] - y[l])
        b_pl[l] *= 1 / float(n)

    b_pp = 0.0
    for k in range(0, n):
        for l in range(0, n):
            b_pp += math.fabs(y[k] - y[l])
    b_pp *= 1 / math.pow(n, 2)

    b = np.zeros((n, n))  # type: Dict[Tuple[int, int]]
    for k in range(0, n):
        for l in range(0, n):
            b[k, l] = math.fabs(y[k] - y[l]) - b_kp[k] - b_pl[l] + b_pp

    # --- Distance  ---------------------
    v_xy = 0.0
    for k in range(0, n):
        for l in range(0, n):
            v_xy += a[k, l] * b[k, l]
    v_xy *= 1 / math.pow(n, 2)

    return v_xy


# ---------------------------------------------------------------------------------------------------------------------


# ----------------------------------------- Distance correlation ------------------------------------------------------
def d_corr(x, y):
    var_x = d_cov(x, x)
    var_y = d_cov(y, y)
    cond = var_x * var_y
    r_xy = 0.0
    if cond != 0:
        r_xy = d_cov(x, y) / math.sqrt(cond)
    return r_xy


# ---------------------------------------------------------------------------------------------------------------------

#"""
# ----------------------------------------------------- MIC -----------------------------------------------------------
def MIC(x, y):
    mine = MINE(alpha =0.6, c=5)
    mine.compute_score(x, y)
    return mine.mic()
# ---------------------------------------------------------------------------------------------------------------------
#"""


# ----------------------------------------- Comparative Dependency ----------------------------------------------------
# ---------------------------- Propose 2 distance between p(X) and P(X|Y) ---------------------------------------------
def cd_ixy(x, y, num_bin_x, num_bin_y):
    n = x.size
    h, x_edges, y_edges = np.histogram2d(x, y, bins=[num_bin_x, num_bin_y])
    h = np.array(h)  # type: Dict[Tuple[int, int]]
    sum_x = np.sum(h, axis=0)  # type: list
    sum_y = np.sum(h, axis=1)  # type: list
    norm = float(num_bin_x) / ((num_bin_x - 1) * 2 * num_bin_y)
    ixy = 0.0
    for i in range(0, num_bin_x):
        for j in range(0, num_bin_y):
            if sum_x[j] != 0:
                ixy += math.fabs((sum_y[i] / float(n)) - (h[i, j] / float(sum_x[j])))
    return norm * ixy


# ---------------------------------------------------------------------------------------------------------------------


# ----------------------------------------- Comparative Mutual Dependency ---------------------------------------------
def cmd(x, y, num_bin_x, num_bin_y):
    ixy = cd_ixy(x, y, num_bin_x, num_bin_y)
    iyx = cd_ixy(y, x, num_bin_y, num_bin_x)
    return (ixy + iyx) / 2

def ucmd(x, y, num_bin_x, num_bin_y):
    ixy1 = ud_ixy(x, y, num_bin_x, num_bin_y)
    ixy2 = cd_ixy(x, y, num_bin_x, num_bin_y)
    iyx1 = ud_ixy(y, x, num_bin_y, num_bin_x)
    iyx2 = cd_ixy(y, x, num_bin_y, num_bin_x)
    return (ixy1+ixy2+iyx1+iyx2)/4

# ------------------------------------------------ FS Dependencies ---------------------------------------------------

def umdv(x, y, num_bin_x, num_bin_y):
    ixy = ud_ixy(x, y, num_bin_x, num_bin_y)
    iyx = ud_ixy(y, x, num_bin_y, num_bin_x)
    #return (ixy+iyx)/2
    return ixy
    return iyx

def cmdv(x, y, num_bin_x, num_bin_y):
    ixy = cd_ixy(x, y, num_bin_x, num_bin_y)
    iyx = cd_ixy(y, x, num_bin_y, num_bin_x)
    #return (ixy+iyx)/2
    return ixy
    return iyx

def ucmdv(x, y, num_bin_x, num_bin_y):
    ixy1 = ud_ixy(x, y, num_bin_x, num_bin_y)
    ixy2 = cd_ixy(x, y, num_bin_x, num_bin_y)
    iyx1 = ud_ixy(y, x, num_bin_y, num_bin_x)
    iyx2 = cd_ixy(y, x, num_bin_y, num_bin_x)
    return (ixy1+ixy2+iyx1+iyx2)/4
    #return (ixy1 + ixy2) / 2
    #return (iyx1 + iyx2) / 2
        
# ---------------------------------------------------------------------------------------------------------------------

