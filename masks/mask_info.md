The group mask [masker.nii.gz]() was computed using the extract_signal function in [prepping_data.py](https://github.com/me-pic/picard_feps_2022/blob/main/scripts/prepping_data.py) to the extract the signal within the whole-brain.

The analysis performed on the signal within the primary motor cortex used the nilearn Harvard-Oxford atlas (precentral region, bilaterally). For more details, see nilearn documentation: https://nilearn.github.io/dev/modules/generated/nilearn.datasets.fetch_atlas_harvard_oxford.html

For the spatial similarity analysis decomposed across cortical networks, the nilearn Yeo atlas was used. For more details, see nilearn documentation: https://nilearn.github.io/dev/modules/generated/nilearn.datasets.fetch_atlas_yeo_2011.html