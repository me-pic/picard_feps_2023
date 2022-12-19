#!/usr/bin/env python

import os
import json
import pickle
import prepping_data
import building_model
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.masking import unmask
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser

def main(path_dataset, path_fmri, path_output, seed, model, reg, confound, run_regression, run_permutations, run_bootstrap):
    """
    path
    ----
    Path to json file containing the data (independent variable, dependent variable, group variable if needed)

    seed
    ----
    Value to use to set the seed

    model parameter
    ---------------
    Areas of the brain that will be used to extract the data:

    'whole-brain': extract the activity from the whole-brain
    <PATH_TO_MASK>: model could take the path to a specified mask (niimg-like object)

    reg parameter
    -------------
    Algorithm to use on the data

    'lasso': Apply Lasso regression 
    'ridge': Apply Rigde regressions
    'svr': Apply a Support Vector Regression 
    'linear': Apply a linear regression 
    'svc': Apply a Support Vector Classifier
    'lda': Apply a Linear Discriminant Analysis classifier
    'rf': Apply a Random Forest classifier

    analysis parameter
    ------------------
    Specify which kind of analysis to run on the data between 3 choices:
    'regression': regression analysis
    'classification': classification analysis
    'sl': searchlight analysis

    folder
    ------
    Where to save the data
    """


    ########################################################################################
    #Loading the datasets
    ########################################################################################
    ##Open json file containing the dataset
    open_json = open(path_dataset, "r")
    data = json.loads(open_json.read())
    ##Predicted variable
    y = np.array(data["target"])
    ##Group variable: how the data is grouped (by subjects)
    gr = np.array(data["group"])
    ##Close json file
    open_json.close()
    ##Load confound file if specified
    if confound == None:
        confound = None
    else:
        confound = pd.read_csv(confound)

    ########################################################################################
    #Extract fmri signal
    ########################################################################################
    ##Convert fmri files to Nifti-like objects
    array_feps = prepping_data.hdr_to_Nifti(data["data"], path_fmri)
    if model == "whole-brain":
        masker, extract_X = prepping_data.extract_signal(array_feps, mask="template", standardize = True, confound=confound)
    else:
        masker = nib.load(model)
        extract_X = prepping_data.extract_signal_from_mask(array_feps, masker, affine=False)
    #Standardize the signal
    stand_X = StandardScaler().fit_transform(extract_X.T)
    X = stand_X.T
    
    ########################################################################################
    #Train and test the model
    ########################################################################################
    #Define the algorithm
    if reg == "lasso":
        algo = Lasso()
    elif reg == "ridge":
        algo = Ridge()
    elif reg == "svr":
        algo = SVR(kernel="linear")
    elif reg == 'linear':
        algo = LinearRegression()
    
    #Set model parameters
    splits=10
    split_procedure='GSS'
    test_size=0.3

    ########################################################################################
    #Regression
    ########################################################################################
    if run_regression:
        X_train, y_train, X_test, y_test, y_pred, model, model_voxel, df_metrics = building_model.train_test_model(X, y, gr, splits=splits, test_size=test_size,reg=algo,random_seed=42, standard=True)

        #Save the outputs
        if model == 'whole-brain':
            model_name = model
        else:
            model_name = os.path.basename(model).split('.')[0]

        #Save readme file with analysis info
        filename_txt = os.path.join(path_output, "readme.txt")
        with open(filename_txt, 'w') as f:
            f.write(f"cv procedure = {split_procedure} \nnumber of folds = {splits} \ntest size = {test_size} \nalgorithm = {reg} \nrandom seed = {seed} \nconfounds = {confound}")

        #Save the performance metrics
        df_metrics.to_csv(os.path.join(path_output, f'dataframe_metrics_{model_name}.csv'))

        #Save model's coefficients
        if model == "whole-brain" :
            for i, element in enumerate(model_voxel):
                (masker.inverse_transform(element)).to_filename(f"coefs_whole_brain_{i}.nii.gz")
            model_to_averaged = model_voxel.copy()
            model_averaged = sum(model_to_averaged)/len(model_to_averaged)
            (masker.inverse_transform(model_averaged)).to_filename("coefs_whole_brain_ave.nii.gz")
        else :
            array_model_voxel = []
            if model == "M1" :
                unmask_model = unmask(model_voxel, mask_M1)
            if model == "without M1": 
                unmask_model = unmask(model_voxel, mask_NoM1)

            for element in unmask_model:
                array_model_voxel.append(np.array(element.dataobj))

            model_ave = sum(array_model_voxel)/len(array_model_voxel)
            model_to_nifti = nib.nifti1.Nifti1Image(model_ave, affine = array_feps[0].affine)
            model_to_nifti.to_filename(f"coefs_{model}_ave.nii.gz")
        
        #Predict on the left out dataset
        print("Test accuray: ", building_model.predict_on_test(X_train=X[:len(y_0)], y_train=y_0, X_test=X[len(y_0):], y_test=y_1, reg=reg))
        
        for i in range(len(X_train)):
            filename = f"train_test_{i}.npz"
            np.savez(filename, X_train=X_train[i],y_train=y_train[i],X_test=X_test[i],y_test=y_test[i],y_pred=y_pred[i])

        #Saving the model
        filename_model = f"lasso_models_{model}.pickle" 
        pickle_out = open(filename_model,"wb")
        pickle.dump(model, pickle_out)
        pickle_out.close()

    ########################################################################################
    #Compute permutation tests
    ########################################################################################
    if run_permutations:
        score, perm_scores, pvalue = building_model.compute_permutation(X, y, gr, reg=algo, random_seed=seed)
        perm_dict = {'score': score, 'perm_scores': perm_scores.tolist(), 'pvalue': pvalue}
        filename_perm = f"permutation_output_{model}_{seed}.json"
        with open(filename_perm, 'w') as fp:
            json.dump(perm_dict, fp)

    ########################################################################################
    #Compute bootstrap tests
    ########################################################################################
    if run_bootstrap:
        True

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_dataset", type=str, default=None)
    parser.add_argument("--path_fmri", type=str, default=None)
    parser.add_argument("--path_output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="whole-brain")
    parser.add_argument("--reg", type=str, choices=['lasso','ridge','svr','linear'], default='lasso')
    parser.add_argument('--confound', type=str, default=None)
    parser.add_argument('--run_regression', action='store_true') 
    parser.add_argument('--run_permutations', action='store_true')
    parser.add_argument('--run_bootstrap', action='store_true')
    args = parser.parse_args()

    main(args.path_dataset, args.path_fmri, args.path_output, args.seed, args.model, args.reg, args.confound, )
