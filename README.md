> :warning: This section is under construction :construction_worker_woman:

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

## LASSO-PCR analysis

## Permutation tests

## Bootstrap analysis

## Similarity analysis
The similarity analysis can be run independently of the other scripts with [similarity.py](https://github.com/me-pic/picard_feps_2022/blob/main/scripts/similarity.py)

<b>How to launch the similarity script:</b>
<br>`python ./similarity.py --path_signature '/path/to/signature' --path_feps '/path/to/feps/or/any/other/signature' --path_output /path/to/output`


## Visualization
The visualization script can be run with [visualization.py]()

<b>How to launch the visualization script:</b>
<br>`python ./visualization.py`
