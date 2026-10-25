"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    st = set()
    m , n = X.shape
    nv = np.zeros(n)
    nc = np.zeros(n)
    for j in range( n ) : 
        for i in range( m ) : 
            if np.isnan(X[i][j]) : 
                st.add((i,j))
                continue 
            else : 
                nv[j] += X[i][j]
                nc[j] += 1 
    nm = np.zeros(n)
    for j in range ( n ) : 
        if nc[j] == 0 : 
            continue 
        nm[j] = nv[j] / nc[j]
    for s in st : 
        X[s] = nm[s[1]]
    return X

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    q1 = np.percentile(X , 25 , axis = 0)
    q3 = np.percentile(X , 75 , axis = 0)
    iqr = q3 - q1 
    lower = q1 - k * iqr 
    upper = q3 + k * iqr 
    return lower , upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    NewX = X.copy()
    for idx in np.ndindex(NewX.shape):
        NewX[idx] = max(lower[idx[1]] , NewX[idx])
        NewX[idx] = min(upper[idx[1]] , NewX[idx])
    return NewX

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    m , = numerator.shape
    res = np.zeros(m)
    for i in range(m) : 
        res[i] = numerator[i] / ( denominator[i] + eps )
    return res

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    m , n  = X.shape 
    j = 0 
    nX = []
    for i in range( m ) : 
        s = X[i].tolist()
        s.append(col[j])
        ns = np.array(s)
        nX.append(ns)
        j+= 1 
    return np.array(nX)

# Step 6 - one_hot_encode
from collections import defaultdict
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    m , = labels.shape
    nl = labels.tolist()
    snl = sorted(nl)
    k = 0 
    mp = defaultdict(int)
    for i in range ( m ) : 
        if snl[i] not in mp : 
            mp[snl[i]] = k 
            k+= 1 
        else : 
            snl[i] = mp[snl[i]]
    for i in range ( m ) : 
        nl[i] = mp[nl[i]]
    # print(nl)
    st = sorted(set(nl))
    res = [ [ 0 ] * len(st) for _ in range (m) ]
    # print(res)
    for i in range ( m ) : 
        for j in range ( len(st) ) : 
            if int(nl[i]) == st[j] : 
                res[i][j] = 1.0 
    return np.array(res)

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    mn , std = (np.mean(X,axis=0) , np.std(X,axis=0))
    m , = std.shape 
    for i in range ( m ) : 
        if std[i] == np.float64(0.0) : 
            std[i] = np.float64(1.0)
    return mn , std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    N , F = X.shape 
    res = np.zeros_like(X , dtype = np.float64)
    for i in range ( N ) : 
        for j in range ( F ) : 
            res[i][j] = np.float64((X[i][j] - mean[j]) / std[j] ) 
    return res

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    m , n = X.shape
    col = np.full( m , 1.0 , dtype = np.float64 )
    return np.column_stack( ( col , X ) )

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    np.random.seed(seed)
    res = np.random.permutation(n_samples)
    return res

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    N , = indices.shape
    train = []
    val = []
    tst = []
    if train_ratio * N < 1 : 
        tst = indices.copy()
        return np.array(train) ,  np.array(val) , np.array(tst) 
    for i in range ( N ) : 
        if ( i < N * train_ratio) : 
            train.append(indices[i])
        elif ( i <= ( N * train_ratio + (N * val_ratio -1 )) ): 
            val.append(indices[i])
        else : 
            tst.append(indices[i])
    return np.array(train) ,  np.array(val) , np.array(tst)

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    xRes , yRes = [] , []
    nX = X.tolist()
    ny = y.tolist()
    nid = indices.tolist()
    for idx in nid : 
        xRes.append(nX[int(idx)])
        yRes.append(ny[int(idx)])
    return np.array(xRes , dtype = np.float64) , np.array(yRes , dtype = np.float64)

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return X @ theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    m , = y_true.shape 
    res = np.float64(0)
    for i in range ( m ) : 
        res += (abs(y_true[i] - y_pred[i]) )
    res /= (m )
    return res

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    m , = y_true.shape 
    res = np.float64(0)
    for i in range ( m ) : 
        res += (abs(y_true[i] - y_pred[i]) ) ** 2 
    res /= ( m )
    return res ** (1/2)

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    m , = y_true.shape
    mn = np.mean(y_true)
    SS_tot , SS_res = 0 , 0 
    for i in range ( m ) : 
        SS_tot += ( abs(y_true[i] - mn ) )  ** 2 
        SS_res += ( abs(y_true[i] -  y_pred[i]) ) ** 2 
    SS_tot /= ( m )
    SS_res /= ( m )
    if SS_tot == np.float64(0) : 
        return np.float64(0)
    return np.float64 ( 1 - (( SS_res ) / ( SS_tot )) )

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    r = y_true - y_pred
    return { 
        'mean' : np.mean(r)  , 
        'std' : np.std(r) , 
        'median_abs': np.median(np.abs(r))
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X = impute_nan_with_mean(X)
    lower , upper = compute_iqr_bounds(X , iqr_k)
    X = clip_columns(X , lower , upper )
    return X

# Step 20 - assemble_feature_matrix
import numpy as np 
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    num = []
    den = []
    m , n = X_num.shape 
    for i in range ( m ) : 
        num.append(X_num[i][ratio_num_idx])
        den.append(X_num[i][ratio_den_idx])
    col = make_ratio_feature(np.array(num) , np.array(den))
    X_num = append_column(X_num , col)
    if cat_labels is not None:
        labels = one_hot_encode(cat_labels)
        X_num = np.hstack((X_num, labels))
    return X_num

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...
    n_samples = X.shape[0]
    nr = make_shuffled_indices(n_samples , seed)
    i1 , i2 , i3 = partition_indices(
        nr , train_ratio , val_ratio
    )
    x1 , y1 = subset_xy(X,y,i1)
    x2 , y2 = subset_xy(X,y,i2)
    x3 , y3 = subset_xy(X,y,i3)
    return { 
         'X_train' : x1 , 'y_train' : y1  , 'X_val' : x2, 'y_val' : y2 , 'X_test' : x3, 'y_test' : y3
    }

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

