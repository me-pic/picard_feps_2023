# Code for Picard et al., Title

## Notes

The scripts were developed to analyze already preprocessed trial-by-trial fMRI contrast images (hdr/img files). The data used in 
this study come from Kunz et al. (2012) study. The data were not in BIDS format, thus the scripts might not directly work on BIDS 
organized data. Each script can be run separately (see the documentation below), but a main script 
([main.py](https://github.com/me-pic/picard_feps_2022/blob/main/scripts/main.py)) is provided to run the complete analysis pipeline.

The preprocessing of the data and the first level analysis were performed using SPM8. Details about those steps can be found in the Method section of the article.

The brain signatures used for the similarity analyis can be found below:
- NPS: on request from Tor Wager
- [SIIPS](https://github.com/canlab/Neuroimaging_Pattern_Masks/tree/master/Multivariate_signature_patterns/2017_Woo_SIIPS1) 
- [PVP](https://github.com/canlab/Neuroimaging_Pattern_Masks/tree/master/Multivariate_signature_patterns/2022_coll_pain_monetary_reward_decision_value)
- [MAThS](https://github.com/canlab/Neuroimaging_Pattern_Masks/tree/master/Multivariate_signature_patterns/2021_Ceko_MPA2_multiaversive) 

Required python packages can be found in the [requirements.txt](https://github.com/me-pic/picard_feps_2022/blob/main/scripts/requirements.txt) file. To install the packages in a [virtual environment](https://pypi.org/project/virtualenv/), use the following line: 
`pip install -r requirements.txt`

## LASSO-PCR analysis
The regression analysis is run using [main.py](https://github.com/me-pic/picard_feps_2022/blob/main/scripts/main.py).

<b>How to run the LASSO-PCR analysis:</b>
<br>`python ./main.py --path_dataset /path/to/dataset --path_fmri /path/to/fmri/data --path_output /path/to/output --seed 42 --model 'whole-brain' --reg 'lasso' --confound /path/to/confound/file --run_regression`
- --path_dataset: specifies the path to json file containing the dataset
- --path_fmri: specifies the path containing the fmri data (not in BIDS format)
- --path_output: specifies the path to output the results of the regression analysis
- --seed (optional): specifies the integer to initialize a pseudorandom number generator. The default value is 42
- --mask (optional): specifies the mask to use to extract the signal. This argument can take the path to a nii file containing a mask. The default value is 'whole-brain', meaning that the signal from the whole-brain will be used
- --reg (optional): specifies the regression algorithm to use in the analysis. The default value is 'lasso', meaning that a LASSO regression will be performed
- --confound (optional): specifies the path to the counfounds file if needed. The default value is None, meaning that no confounds will be taken into account for the signal extraction
- --run_regression: need to be specified in order to run the regression analysis. No value is needed for that argument

## Permutation tests
The permutations tests are run using [main.py](https://github.com/me-pic/picard_feps_2022/blob/main/scripts/main.py).

<b>How to run the permutation tests:</b>
<br>`python ./main.py --path_dataset /path/to/dataset --path_fmri /path/to/fmri/data --path_output /path/to/output --seed 42 --model 'whole-brain' --reg 'lasso' --confound /path/to/confound/file --run_permutations`
- --path_dataset: specifies the path to json file containing the dataset
- --path_fmri: specifies the path containing the fmri data (not in BIDS format)
- --path_output: specifies the path to output the results of the regression analysis
- --seed (optional): specifies the integer to initialize a pseudorandom number generator. The default value is 42
- --mask (optional): specifies the mask to use to extract the signal. This argument can take the path to a nii file containing a mask. The default value is 'whole-brain', meaning that the signal from the whole-brain will be used
- --reg (optional): specifies the regression algorithm to use in the analysis. The default value is 'lasso', meaning that a LASSO regression will be performed
- --confound (optional): specifies the path to the counfounds file if needed. The default value is None, meaning that no confounds will be taken into account for the signal extraction
- --run_permutations: need to be specified in order to run the permutation tests. No value is needed for that argument

## Bootstrap tests
The bootstrap tests are run using [main.py](https://github.com/me-pic/picard_feps_2022/blob/main/scripts/main.py).

<b>How to run the bootstrap tests:</b>
<br>`python ./main.py --path_dataset /path/to/dataset --path_fmri /path/to/fmri/data --path_output /path/to/output --seed 42 --model 'whole-brain' --reg 'lasso' --confound /path/to/confound/file --run_bootstrap`
- --path_dataset: specifies the path to json file containing the dataset
- --path_fmri: specifies the path containing the fmri data (not in BIDS format)
- --path_output: specifies the path to output the results of the regression analysis
- --seed (optional): specifies the integer to initialize a pseudorandom number generator. The default value is 42
- --mask (optional): specifies the mask to use to extract the signal. This argument can take the path to a nii file containing a mask. The default value is 'whole-brain', meaning that the signal from the whole-brain will be used
- --reg (optional): specifies the regression algorithm to use in the analysis. The default value is 'lasso', meaning that a LASSO regression will be performed
- --confound (optional): specifies the path to the counfounds file if needed. The default value is None, meaning that no confounds will be taken into account for the signal extraction
- --run_bootstrap: need to be specified in order to run the bootstrap tests. No value is needed for that argument

## Similarity analysis
The similarity analysis can be run independently of the other scripts with [similarity.py](https://github.com/me-pic/picard_feps_2022/blob/main/scripts/similarity.py).

<b>How to launch the similarity script:</b>
<br>`python ./similarity.py --path_signature '/path/to/signature' --path_feps '/path/to/feps/or/any/other/signature' --path_output 
'/path/to/output'`
- --path_signature: specifies the path to the signature Nifti file on which to compute the spatial similarity
- --path_feps: specifies the path to the feps Nifti file (or any other signature) on which to compute the spatial similarity
- --path_output: specifies the path to output the results

## Visualization
The visualization script can be run with [visualization.py](https://github.com/me-pic/picard_feps_2022/blob/main/scripts/visualization.py). The fonts and the disposition of the figures were edited using Adobe Illustrator.

<b>How to launch the visualization script:</b>
<br>`python ./visualization.py --path_output '/path/to/output' --path_feps '/path/to/feps' 
--path_behavioral '/path/to/behavioral/measures' --path_dot_product '/path/to/dot/product' 
--path_performance '/path/to/performance/metrics' --path_y_test 'path/to/y/test' --path_y_pred 
'/path/to/y/pred' --path_siips_similarity_networks '/path/siips/similarity' --path_pvp_similarity_networks '/path/pvp/similarity' --path_maths_similarity_networks '/path/maths/similarity' --path_similarity_matrix '/path/similarity/matrix'`
- --path_output: specifies the path to output the figures
- --path_feps (optional): specifies the path to the feps Nifti file. If not specified, the 
signature weights will not be plotted 
- --path_behavioral (optional): specified the path to the csv file containing the behavioral 
measures
- --path_dot_product (optional): specified the path to the npy file containing the dot product 
values between a signature weights and the FACS scores. If not specified, the correlation between 
the signature expression and the FACS scores will not be plotted
 - --path_performance (optional): specifies the path to the csv file containing the 
performance 
metrics of the model. If not specified, the violin plot of the model performance will not be 
plotted
- --path_y_test (optional): specifies the path to the pickle file containing the y_test values. If 
not specified, the regression plot of the model performance will not be plotted
- --path_y_pred (optional): specifies the path to the pickle file containing the y_pred values
- --path_siips_similarity_networks (optional): specifies the path to pickle file containing the spatial similarity metrics between the FEPS and the SIIPS-1. If not specified, the spatial similarity barplots across networks are not plotted
- --path_pvp_similarity_networks (optional): specifies the path to pickle file containing the spatial similarity metrics between the FEPS and the PVP
- --path_maths_similarity_networks (optional): specifies the path to pickle file containing the spatial similarity metrics between the FEPS and the MAThS
- --path_similarity_matrix (optional): specifies the path to numpy npy file containing the spatial similarity values between the FEPS and other pain-related signatures. If not specified, the similarity matrix is not plotted
