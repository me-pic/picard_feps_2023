import os
import json
import numpy as np
import nibabel as nib
from nilearn.masking import apply_mask
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_img
from argparse import ArgumentParser

def hdr_to_Nifti(files, path_files=''):
    """
    Convert hdr files to Nifti-like objects

    Parameters
    ----------
    files: list 
        list containing the hdr filenames
    path_files: string
        path to the hdr files if not already contained in files

    Returns
    ----------
    array: numpy.ndarray
        array containing Nifti-like objects
    """
    array = []
    for element in files:
        array = np.append(array, nib.load(os.path.join(path_files,element)))

    print(f'array size: {array.shape}')

    return array


def extract_signal(data, mask="template", standardize = True):
    """
    Apply a mask to extract the signal from the data and save the mask
    in a html format

    Parameters
    ----------
    data: list 
        list of Niimg-like objects
    mask: string
        strategy to compute the mask. By default the gray matter is extracted based on the MNI152 brain mask
    standardize: bool
        strategy to standardize the signal. The signal is z-scored by default

    Returns
    ----------
    masker_all: 
        mask of the data
    masker_gm: numpy.ndarray
        array containing the extracted signal

    See also nilearn NifitMasker documentation: https://nilearn.github.io/dev/modules/generated/nilearn.maskers.NiftiMasker.html
    """
    masker_all = NiftiMasker(mask_strategy = mask,standardize=standardize, verbose = 1, reports = True)
    
    masker_gm = masker_all.fit_transform(data)
    print("mean: ", round(masker_gm.mean(),2), "\nstd: ", round(masker_gm.std(),2))
    print("Signal size: ", masker_gm.shape)

    report = masker_all.generate_report()
    report.save_as_html("masker_report.html")

    return masker_all, masker_gm

def extract_signal_from_mask(data, mask):
    """
    Apply a pre-computed mask to extract the signal from the data

    Parameters
    ----------
    data: Niimg-like object
        Niimg-like objects to resample
    mask: 
        mask to apply to the data

    Returns
    ----------
    signal: numpy.ndarray
        extracted signal from mask

    See also nilearn masking documentation: https://nilearn.github.io/dev/modules/masking.html
    """
    affine = data[0].affine
    resample_mask = resample_img(mask,affine)
    signal = apply_mask(data, resample_mask, ensure_finite=True)
    print(signal.shape, type(signal))

    return signal

parser = ArgumentParser()
#Path to json file
parser.add_argument('--path_dataset', type=str, default=None)
#parser.add_argument('--path_output', type=str, default=None)
#parser.add_argument('--filename_output', type=str, default=None)
args = parser.parse_args()

data = json.loads(open(args.path_dataset, 'r').read())
signal = hdr_to_Nifti(data['data'], '/Users/mepicard/Documents/master_analysis/picard_feps_2022_v1/data/data_MK_pain')

#python ./prepping_data.py --path_dataset '/Users/mepicard/Documents/master_analysis/picard_feps_2022_outputs/dataset.json'