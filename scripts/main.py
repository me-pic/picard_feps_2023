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

def main(path_dataset, path_fmri, path_output, seed, mask, reg, confound, run_regression, run_permutations, run_bootstrap):
    """
    path_dataset: string
        specifies the path to json file containing the dataset
    path_fmri: string
        specifies the path containing the fmri data (not in BIDS format)
    path_output: string
        specifies the path to output the results of the regression analysis
    seed: int
        specifies the integer to initialize a pseudorandom number generator. The default value is 42
    mask: string
        specifies the mask to use to extract the signal. This argument can take the path to a nii file containing a mask. The default value is 'whole-brain', meaning that the signal from the whole-brain will be used
    reg: string
        specifies the regression algorithm to use in the analysis. The default value is 'lasso', meaning that a LASSO regression will be performed
    confound: string
        specifies the path to the counfounds file if needed. The default value is None, meaning that no confounds will be taken into account for the signal extraction
    run_regression: boolean
        if True, the regression analysis is computed
    run_permutations: boolean
        if True, the permutation tests are computed
    run_bootstrap: boolean
        if True, the bootstrap tests are computed
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
    if confound is not None:
        try:
            confound = pd.read_csv(confound)
        except:
            print('Confound file not in csv format')

    ########################################################################################
    #Extract fmri signal
    ########################################################################################
    ##Convert fmri files to Nifti-like objects
    array_feps = prepping_data.hdr_to_Nifti(data["data"], path_fmri)
    if mask == "whole-brain":
        masker, extract_X = prepping_data.extract_signal(array_feps, mask="template", standardize = True, confound=confound)
    else:
        masker = nib.load(mask)
        extract_X = prepping_data.extract_signal_from_mask(array_feps, masker, affine=True)
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
    if mask == 'whole-brain':
        mask_name = mask
    else:
        mask_name = os.path.basename(mask).split('.')[0]

    ########################################################################################
    #Regression
    ########################################################################################
    if run_regression:
        X_train, y_train, X_test, y_test, y_pred, model, model_voxel, df_metrics = building_model.train_test_model(X, y, gr, splits=splits, test_size=test_size,reg=algo,random_seed=42, standard=True)

        #Save the outputs
        ##Save readme file with analysis info
        filename_txt = os.path.join(path_output, "readme.txt")
        with open(filename_txt, 'w') as f:
            f.write(f"cv procedure = {split_procedure} \nnumber of folds = {splits} \ntest size = {test_size} \nalgorithm = {reg} \nrandom seed = {seed} \nconfounds = {confound}")

        ##Save the performance metrics
        df_metrics.to_csv(os.path.join(path_output, f'dataframe_metrics_{mask_name}.csv'))

        ##Save model's coefficients
        if mask == "whole-brain" :
            for i, element in enumerate(model_voxel):
                (masker.inverse_transform(element)).to_filename(os.path.join(path_output,f"coefs_whole_brain_{i}.nii.gz"))
            model_to_averaged = model_voxel.copy()
            model_averaged = sum(model_to_averaged)/len(model_to_averaged)
            (masker.inverse_transform(model_averaged)).to_filename(os.path.join(path_output, "coefs_whole_brain_ave.nii.gz"))
        else :
            array_model_voxel = []
            unmask_model = unmask(model_voxel, masker)

            for i, element in enumerate(unmask_model):
                array_model_voxel.append(np.array(element.dataobj))
                element.to_filename(os.path.join(path_output, f"coefs_{mask_name}_{i}.nii.gz"))

            model_ave = sum(array_model_voxel)/len(array_model_voxel)
            model_to_nifti = nib.nifti1.Nifti1Image(model_ave, affine = array_feps[0].affine)
            model_to_nifti.to_filename(f"coefs_{mask_name}_ave.nii.gz")
        
        ##Save y_train, y_test and y_pred using pickle
        filename_y_train = os.path.join(path_output, f"y_train_{mask_name}.pickle")
        y_train_out = open(filename_y_train,"wb")
        pickle.dump(y_train, y_train_out)
        y_train_out.close()

        filename_y_test = os.path.join(path_output, f"y_test_{mask_name}.pickle")
        y_test_out = open(filename_y_test,"wb")
        pickle.dump(y_test, y_test_out)
        y_test_out.close()

        filename_y_pred = os.path.join(path_output, f"y_pred_{mask_name}.pickle")
        y_pred_out = open(filename_y_pred,"wb")
        pickle.dump(y_pred, y_pred_out)
        y_pred_out.close()

    ########################################################################################
    #Compute permutation tests
    ########################################################################################
    if run_permutations:
        ##Define permutations tests
        n_permutations = 5000

        ##Run the permutations tests
        score, perm_scores, pvalue = building_model.compute_permutation(X, y, gr, reg=algo, random_seed=seed, n_permutations=n_permutations)
        perm_dict = {'score': score, 'perm_scores': perm_scores.tolist(), 'pvalue': pvalue}
        ##Save outputs
        filename_perm = f"permutation_output_{mask_name}.json"
        with open(filename_perm, 'w') as fp:
            json.dump(perm_dict, fp)

    ########################################################################################
    #Compute bootstrap tests
    ########################################################################################
    if run_bootstrap:
        ##Define bootstrap parameters
        n_resampling = 5000
        n_jobs = -1
        standard = True

        ##Run the bootstrap tests
        resampling_array, resampling_coef = building_model.bootstrap_test(X, y, gr, reg=algo, njobs=n_jobs, n_resampling=n_resampling, standard=standard)
        z, pval, pval_bonf = building_model.bootstrap_scores(resampling_array)
        ##Save outputs
        np.savez(os.path.join(path_output, f"bootstrap_{reg}_sample_{n_resampling}_{mask_name}"), array = resampling_array, coef = resampling_coef, z = z, pval = pval, pval_bonf = pval_bonf)



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
