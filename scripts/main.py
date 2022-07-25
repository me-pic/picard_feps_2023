#!/usr/bin/env python

import os
import errno
import json
import pickle
import shutil
import sys
from datetime import datetime
import prepping_data
import building_model
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.masking import unmask
from sklearn.linear_model import Lasso, Ridge, HuberRegressor, LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser

def main():
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
    'svc': Apply a Support Vector Classifier
    'lda': Apply a Linear Discriminant Analysis classifier
    'rf': Apply a Random Forest classifier
    'huber': Apply a Robust Huber Regression
    'linear': Apply a linear regression 

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
    
    parser = ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="whole-brain")
    parser.add_argument("--reg", type=str, choices=['lasso','ridge','svr','svc','lda','rf','huber','linear'], default='lasso')
    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--confound', type=str, default=None)
    parser.add_argument('--analysis', type=str, choices=['regression','classification','sl'], default='regression')
    args = parser.parse_args()

    if args.folder == None:
        args.folder = datetime.now().strftime("%d_%m_%Y")
    
    try: 
        os.makedirs(args.folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    #Loading the datasets
    data = json.loads(open(args.path, "r").read())

    #Predicted variable
    y = np.array(data["target"])
    
    #Group variable: how the data is grouped (by subjects)
    gr = np.array(data["group"])

    #Convert fmri files to Nifti-like objects
    array_feps = prepping_data.hdr_to_Nifti(data["data"])
    
    #Load confound file
    if args.confound == None:
        confound = None
    else:
        confound = pd.read_csv(args.confound)

    #Extract signal from gray matter
    if args.model == "whole-brain":
        masker, extract_X = prepping_data.extract_signal(array_feps, mask="template", standardize = True, confound=confound)
    else:
        masker = nib.load(args.model)
        extract_X = prepping_data.extract_signal_from_mask(array_feps, masker, affine=False)
    
    #Standardize the signal
    stand_X = StandardScaler().fit_transform(extract_X.T)
    X = stand_X.T
    
    #Define the algorithm
    if args.reg == "lasso":
        reg = Lasso()
    elif args.reg == "ridge":
        reg = Ridge()
    elif args.reg == "svr":
        reg = SVR(kernel="linear")
    elif args.reg == "svc":
        reg = SVC(kernel="linear", class_weight='balanced')
    elif args.reg == "lda":
        reg = LinearDiscriminantAnalysis()
    elif args.reg == "rf":
        reg = RandomForestClassifier(n_estimators=4, random_state=42)
    elif args.reg == 'huber':
        reg = HuberRegressor(alpha=1.0)
    elif args.reg == 'linear':
        reg = LinearRegression()
    
    #ANALYSIS
    if args.analysis=="regression":
        X_train, y_train, X_test, y_test, y_pred, model, model_voxel, df_metrics = building_model.train_test_model(X[:len(y)], y, gr, test_size=0.3,reg=reg,random_seed=42)
    if args.analysis=="classification":
        X_train, y_train, X_test, y_test, y_pred, model, df_metrics = building_model.train_test_classify(X, y, gr, clf=reg)
    if args.analysis=="sl":
        sl_scores = building_model.searchlight_analysis(X,y_0,reg=reg,mask_sl=mask_sl,mask_signal=masker.mask_img_,gr=gr,splits=5,test_size=0.3)
        sl_scores.to_filename(os.path.join(args.folder,f"_sl_scores.nii"))
        sys.exit()

    if args.model == "whole-brain" :
        for i, element in enumerate(model_voxel):
            (masker.inverse_transform(element)).to_filename(f"coefs_whole_brain_{i}.nii.gz")
        
        model_to_averaged = model_voxel.copy()
        model_averaged = sum(model_to_averaged)/len(model_to_averaged)
        (masker.inverse_transform(model_averaged)).to_filename("coefs_whole_brain_ave.nii.gz")

    else :
        array_model_voxel = []
        if args.model == "M1" :
            unmask_model = unmask(model_voxel, mask_M1)
        if args.model == "without M1": 
            unmask_model = unmask(model_voxel, mask_NoM1)

        for element in unmask_model:
            array_model_voxel.append(np.array(element.dataobj))

        model_ave = sum(array_model_voxel)/len(array_model_voxel)
        model_to_nifti = nib.nifti1.Nifti1Image(model_ave, affine = array_feps[0].affine)
        model_to_nifti.to_filename(f"coefs_{args.model}_ave.nii.gz")
    
    #Predict on the left out dataset
    print("Test accuray: ", building_model.predict_on_test(X_train=X[:len(y_0)], y_train=y_0, X_test=X[len(y_0):], y_test=y_1, reg=reg))
    
    for i in range(len(X_train)):
        filename = f"train_test_{i}.npz"
        np.savez(filename, X_train=X_train[i],y_train=y_train[i],X_test=X_test[i],y_test=y_test[i],y_pred=y_pred[i])

    #Saving the model
    filename_model = f"lasso_models_{args.model}.pickle" 
    pickle_out = open(filename_model,"wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()

    #Compute permutation tests
    score, perm_scores, pvalue = building_model.compute_permutation(X[:len(y_0)], y_0, gr_0, reg=reg, random_seed=args.seed)
    perm_dict = {'score': score, 'perm_scores': perm_scores.tolist(), 'pvalue': pvalue}
    filename_perm = f"permutation_output_{args.model}_{args.seed}.json"
    with open(filename_perm, 'w') as fp:
        json.dump(perm_dict, fp)
    

if __name__ == "__main__":
    main()
