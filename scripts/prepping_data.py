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


def extract_signal(data, mask="template", standardize = True, path_output='', save = False):
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
    path_output: string
        path for saving the output(s)
    save: bool
        if True, the generated mask is saved in a nii.gz file in path_output. Otherwise the mask is not saved

    Returns
    ----------
    masker_all: NiftiMasker object
        mask generated from the data
    masker_gm: numpy.ndarray
        array containing the extracted signal

    See also nilearn NifitMasker documentation: https://nilearn.github.io/dev/modules/generated/nilearn.maskers.NiftiMasker.html
    """
    masker_all = NiftiMasker(mask_strategy = mask,standardize=standardize, verbose = 1, reports = True)
    masker_gm = masker_all.fit_transform(data)
    print(f'mean: {round(masker_gm.mean(),2)}, \nstd: {round(masker_gm.std(),2)}')
    print(f'Signal size: {masker_gm.shape}')

    report = masker_all.generate_report()
    report.save_as_html(os.path.join(path_output, 'masker_report.html'))

    if save:
        nib.save(masker_all.mask_img_, os.path.join(path_output, 'masker.nii.gz'))

    return masker_all, masker_gm


def extract_signal_from_mask(data, path_mask):
    """
    Apply a pre-computed mask to extract the signal from the data

    Parameters
    ----------
    data: Niimg-like object
        Niimg-like objects to resample
    path_mask: string
        path of mask to apply to the data

    Returns
    ----------
    signal: numpy.ndarray
        extracted signal from mask

    See also nilearn masking documentation: https://nilearn.github.io/dev/modules/masking.html
    """
    mask = nib.load(path_mask)
    affine = data[0].affine
    resample_mask = resample_img(mask,affine)
    signal = apply_mask(data, resample_mask, ensure_finite=True)
    print(f'Signal size: {signal.shape}')

    return signal

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path_dataset', type=str, default=None) #Path to json file containing the dataset information
    parser.add_argument('--path_fmri', type=str, default=None)
    parser.add_argument('--path_mask', type=str, default=None)
    parser.add_argument('--path_output', type=str, default=None)
    args = parser.parse_args()

    #Define the parameters
    save_extracted_signal = False #if True, the extracted signal from mask will be save as a npz file

    data = json.loads(open(args.path_dataset, 'r').read())
    signal = hdr_to_Nifti(data['data'], args.path_fmri)

    if args.path_mask is not None:
        X = extract_signal_from_mask(signal, args.path_mask)
    else:
        masker, X = extract_signal(signal, mask="template", standardize = True, path_output=args.path_output, save = True)

    if save_extracted_signal:
        if args.path_mask is not None:
            filename = f"{os.path.basename(args.path_mask).split('.')[0]}_extracted_signal.npz"
        else:
            filename = 'full_brain_extracted_signal.npz'
        np.savez(os.path.join(args.path_output, filename), X)