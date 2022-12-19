from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import zscore, norm, pearsonr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, permutation_test_score, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score


def split_data(X, Y, group, procedure):
    """
    Split the data according to the group parameters
    to ensure that the train and test sets are completely independent

    Parameters
    ----------
    X: numpy.ndarray
        predictive variable
    Y: numpy.ndarray
        predicted variable
    group: numpy.ndarray
        group labels used for splitting the dataset
    procedure: model_selection method 
        strategy to split the data

    Returns
    ----------
    X_train: list 
        train set containing the predictive variable
    X_test: list 
        test set containing the predictive variable
    y_train: list
        train set containing the predicted variable
    y_test: list    
        test set containing the predicted variable
    """
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for train_idx, test_idx in procedure.split(X, Y, group):
        X_train.append(X[train_idx])
        X_test.append(X[test_idx])
        y_train.append(Y[train_idx])
        y_test.append(Y[test_idx])
    
    return X_train, X_test, y_train, y_test


def verbose(splits, X_train, X_test, y_train, y_test, X_verbose=True, y_verbose=True):
    """
    Print the mean and the standard deviation of the train and test sets
   
    Parameters
    ----------
    splits: int
        number of splits used for the cross-validation
    X_train: list
        train set containing the predictive variable
    X_test: list
        test set containing the predictive variable
    y_train: list
        train set containing the predicted variable
    y_test: list
        test set containing the predicted variable
    X_verbose: boolean
        if X_verbose == True, print the descriptive stats for the X (train and test)
    y_verbose: boolean
        if y_verbose == True, print the descriptive stats for the y (train and test)
    """
    for i in range(splits):
        if X_verbose:
            print(i,'X_Train: \n   Mean +/- std = ', X_train[i][:][:].mean(),'+/-', X_train[i][:][:].std())
            print(i,'X_Test: \n   Mean +/- std = ', X_test[i][:][:].mean(),'+/-', X_test[i][:][:].std())
        if y_verbose:
            print(i,'y_Train: \n   Mean +/- std = ', y_train[i][:].mean(),'+/-', y_train[i][:].std(), '\n   Skew = ', stats.skew(y_train[i][:]), '\n   Kurt = ', stats.kurtosis(y_train[i][:]))
            print(i,'y_Test: \n   Mean +/- std = ', y_test[i][:].mean(),'+/-', y_test[i][:].std(), '\n   Skew = ', stats.skew(y_test[i][:]), '\n   Kurt = ', stats.kurtosis(y_test[i][:]))
        print('\n')


def compute_metrics(y_test, y_pred, df, fold, print_verbose=True): 
    """
    Compute different metrics and print them

    Parameters
    ----------
    y_test: numpy.ndarray
        ground truth
    y_pred: numpy.ndarray
        predicted values
    df: dataFrame
        dataFrame containing the result of the metrics
    fold: int
        cross-validation fold for which the metrics are computed
    
    Returns
    ----------
    df_metrics: dataFrame
        dataFrame containing the different metrics
    """  
    pearson_r = pearson(y_test, y_pred) 
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  
    df.loc[fold] = [pearson_r, r2, mae, mse, rmse]

    if print_verbose:
        print('------Metrics for fold {}------'.format(fold))
        print('Pearson-r value= {}'.format(pearson_r))
        print('R2 value = {}'.format(r2))
        print('MAE value = {}'.format(mae))
        print('MSE value = {}'.format(mse))
        print('RMSE value = {}'.format(rmse))
        print('\n')

    return df


def reg_PCA(n_component, reg=Lasso(), standard=False):
    """
    Parameters
    ----------
    n_component: int or float
        number of components (or percentage) to keep in the PCA

    Returns
    ----------
    pipe: Pipeline object
        pipeline to apply PCA and Lasso regression sequentially

    See also sklearn PCA documentation: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    See also sklearn Pipeline documentation: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    """
    pca = PCA(n_component)
    if standard:
        estimators = [('scaler',StandardScaler()),('reduce_dim', pca), ('clf', reg)] 
    else: 
        estimators = [('reduce_dim', pca), ('clf', reg)] 
    pipe = Pipeline(estimators)

    return pipe, pca


def pearson(y_true, y_pred):
    """
    Compute pearson correlation coefficient

    Parameters
    ----------
    y_true: numpy.ndarray
        ground truth
    y_pred: numpy.ndarray
        predicted values
    """
    return pearsonr(y_true, y_pred)[0]


def train_test_model(X, y, gr, reg=Lasso(), splits=5, split_procedure='GSS', test_size=0.3, n_components=0.80, random_seed=42, print_verbose=True, standard=False):
    """
    Build and evaluate a regression model
    First compute the PCA and then fit the regression technique specified on the PCs scores

    Parameters
    ----------
    X: numpy.ndarray
        predictive variable
    y: numpy.ndarray
        predicted variable
    gr: numpy.ndarray
        grouping variable
    reg: linear_model method
        regression technique to perform
    splits: int
        number of split for the cross-validation 
    test_size: float
        percentage of the data in the test set
    n_components: int or float
        number of components (or percentage) to keep for the PCA
    random_seed: int
        controls the randomness of the train/test splits
    print_verbose: bool
        either or not the verbose is printed

    Returns
    ----------
    X_train: list
        list containing the training sets of the predictive variable
    y_train: list
        list containing the training sets of the predictive variable
    X_test: list
        list containing the training sets of the predictive variable
    y_test: list
        list containing the training sets of the predictive variable
    y_pred: list
        list containing the predicted values for each fold
    model_voxel: list
        list of arrays containing the coefficients of the model in the voxel space 
    df_metrics: dataFrame
        dataFrame containing different metrics for each fold
    """ 
    #Initialize the variables
    y_pred = []
    model = []
    model_voxel = []
    df_metrics = pd.DataFrame(columns=["pearson_r", "r2", "mae", "mse", "rmse"])

    #Strategy to split the data
    if split_procedure=='GSS':
        split_method = GroupShuffleSplit(n_splits=splits, test_size=test_size, random_state=random_seed)
    elif split_procedure=='SS':
        split_method = ShuffleSplit(n_splits=splits, test_size=test_size, random_state=random_seed)
    elif split_procedure=='LOGO':
        split_method = LeaveOneGroupOut()
    #Split the data into train and test sets
    X_train, X_test, y_train, y_test = split_data(X, y, gr, procedure=split_method)
    if print_verbose:
        verbose(splits, X_train, X_test, y_train, y_test, X_verbose = True, y_verbose = True)

    #Build and test the model for each c-v fold
    for i in range(splits):
        ###Build and test the model###
        print("----------------------------")
        print("Training model")
        model_reg, _ = reg_PCA(n_components,reg=reg, standard=standard)
        model.append(model_reg.fit(X_train[i], y_train[i]))
        
        ###Scores###
        y_pred.append(model_reg.predict(X_test[i]))
        df_metrics = compute_metrics(y_test[i], y_pred[i], df_metrics, i, print_verbose)
        ###Model coefficients###
        if standard:
            model_voxel.append(model[i][1].inverse_transform(model[i][2].coef_))
        else:
            model_voxel.append(model[i][0].inverse_transform(model[i][1].coef_))

    return X_train, y_train, X_test, y_test, y_pred, model, model_voxel, df_metrics


def compute_permutation(X, y, gr, reg, splits=5, n_components=0.80, n_permutations=5000, scoring="r2", random_seed=42):
    """
    Compute the permutation test for a specified metric (r2 by default)
    Apply the PCA after the splitting procedure

    Parameters
    ----------
    X: numpy.ndarray
        predictive variable
    y: numpy.ndarray
        predicted variable
    gr: numpy.ndarray
        grouping variable
    n_components: int or float
        number of components (or percentage) to keep for the PCA
    n_permutations: int
        number of permuted iteration
    scoring: string
        scoring strategy
    random_seed: int
        controls the randomness

    Returns
    ----------
    score: float
        true score
    perm_scores: numpy.ndarray
        scores for each permuted samples
    pvalue: float
        probability that the true score can be obtained by chance

    See also scikit-learn permutation_test_score documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html
    """
    procedure = GroupShuffleSplit(n_splits = splits, test_size = 0.3, random_state = random_seed)
    pipe, _ = reg_PCA(n_components, reg=reg, standard=True)
    
    score, perm_scores, pvalue = permutation_test_score(estimator=pipe, X=X, y=y, groups= gr, scoring=scoring, cv=procedure, n_permutations=n_permutations, random_state=42, n_jobs=-1)
    
    return score, perm_scores, pvalue


def bootstrap_test(X,y,gr,reg,splits=5,test_size=0.30,n_components=0.80,n_resampling=1000,njobs=5):
    """
    Split the data according to the group parameters
    to ensure that the train and test sets are completely independent

    Parameters
    ----------
    X: numpy.ndarray
        predictive variable
    Y: numpy.ndarray
        predicted variable
    gr: numpy.ndarray
        group labels used for splitting the dataset
    reg: linear_model method
        regression strategy to use 
    splits: int
        number of split for the cross-validation 
    test_size: float
        percentage of the data in the test set
    n_components: int or float
        number of components (or percentage) to include in the PCA 
    n_resampling: int
        number of resampling subsets
    njobs: int
        number of jobs to run in parallel

    Returns
    ----------
    bootarray: numpy.ndarray
        2D array containing regression coefficients at voxel level for each resampling (array-like)
    """

    procedure = GroupShuffleSplit(n_splits=splits,test_size=test_size)

    bootstrap_coef = Parallel(n_jobs=njobs,verbose=1)(
        delayed(_bootstrap_test)(
            X=X,
            y=y,
            gr=gr,
            reg=reg,
            procedure=procedure,
            n_components=n_components,
        )
        for _ in range(n_resampling)
    )
    bootstrap_coef=np.stack(bootstrap_coef)
    bootarray = bootstrap_coef.reshape(-1, bootstrap_coef.shape[-1])
    
    return bootarray, bootstrap_coef


def _bootstrap_test(X,y,gr,reg,procedure,n_components):
    """
    Split the data according to the group parameters
    to ensure that the train and test sets are completely independent

    Parameters
    ----------
    X: numpy.ndarray
        predictive variable
    y: numpy.ndarray
        predicted variable
    gr: numpy.ndarray
        group labels used for splitting the dataset
    reg: linear_model method
        regression strategy to use
    procedure: model_selection method
        strategy to split the data
    n_components: int or float
        number of components (or percentage) to include in the PCA

    Returns
    ----------
    coefs_voxel: list
        regression coefficients for each voxel
    """
    coefs_voxel = []
    #Random sample
    idx = list(range(0,len(y)))
    random_idx = np.random.choice(idx,
                                  size=len(idx),
                                  replace=True)
    X_sample = X[random_idx]
    y_sample = y[random_idx]
    gr_sample = gr[random_idx]
    
    #Train the model and save the regression coefficients
    for train_idx, test_idx in procedure.split(X_sample, y_sample, gr_sample):
        X_train, y_train = X_sample[train_idx], y_sample[train_idx]
        model, _ = reg_PCA(n_components,reg=reg)
        model.fit(X_train, y_train)
        coefs_voxel.append(model[0].inverse_transform(model[1].coef_))
        
    return coefs_voxel
