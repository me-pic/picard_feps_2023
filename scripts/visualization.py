import os
import pickle
import numpy as np
from numpy.linalg import norm
from nilearn.image import math_img
import random
from nilearn import plotting
from nilearn import datasets
from nilearn.masking import unmask
from nilearn.maskers import NiftiMasker
from scipy.stats import ttest_ind, ttest_1samp, permutation_test
from scipy.spatial.distance import correlation
from nilearn.image import load_img, resample_to_img, new_img_like
from nilearn.masking import apply_mask, unmask
from nilearn.reporting import get_clusters_table
from scipy.stats import pearsonr, zscore, linregress
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import nibabel as nib
from itertools import combinations
from argparse import ArgumentParser

def load_pickle(path):
    file = open(path,'rb')
    object_file = pickle.load(file)
    file.close()
    return object_file

def plot_FACS_pattern(x, y, signature_dot_prod, FACS, path_output='', idx=0, palette=None):
    """
    Parameters
    ----------
    x: string
        name of the x axis
    y: string
        name of the y axis
    signature_dot_prod: numpy.ndarray
        array containing the pattern expression values
    FACS: pandas Series or numpy.ndarray
        FACS scores or any behavioral continuours variable
    path_output: string
        path for saving the output
    idx: int
        integer specifying the color to use in palette
    palette: list
        color palette to use for the graph
    """
    sns_plot=sns.lmplot(x=x,
                        y=y,
                        data=pd.DataFrame(np.array([signature_dot_prod, FACS]).T, columns=[x, y]),
                        line_kws={'color': palette[idx]},
                        scatter_kws={'color': palette[idx]})
    plt.savefig(os.path.join(path_output, f'lmplot_{x}_expression_FACS.svg'))

def plotting_signature_weights(path_signature, coords_to_plot, path_output):
    """
    Plot the signature weights given a set of coordinates
    
    Parameters
    ----------
    path_signature: string
        path of the signature (path to nii file) to visualize
    coors_to_plot: dictionary
        dictionary containing the coordinates for each axis (x,y,z)
    path_output: string
        path for saving the output(s)

    Code adapted from https://github.com/mpcoll/coll_painvalue_2021/tree/main/figures
    """
    labelfontsize = 7
    ticksfontsize = np.round(labelfontsize*0.8)
    filename = os.path.basename(path_signature).split('.')[0]

    for axis, coord in coords_to_plot.items():
        for c in coord:
            fig, ax = plt.subplots(figsize=(1.5, 1.5))
            disp = plotting.plot_stat_map(path_signature, cmap=plotting.cm.cold_hot, colorbar=False,
                            dim=-0.3,
                            black_bg=False,
                            display_mode=axis,
                            axes=ax,
                            vmax=6,
                            cut_coords=(c,),
                            alpha=1,
                            annotate=False)
            disp.annotate(size=ticksfontsize, left_right=False)
            
            fig.savefig(os.path.join(path_output, f'{filename}_{axis}_{str(c)}.svg'),
                        transparent=True, bbox_inches='tight', dpi=600)



parser = ArgumentParser()
parser.add_argument("--path_feps", type=str, default=None)
parser.add_argument("--path_output", type=str, default=None)
parser.add_argument("--path_behavioral", type=str, default=None)
parser.add_argument("--path_dot_product", type=str, default=None)
args = parser.parse_args()

#Define the general parameters
sns.set_context("talk")

params = {'legend.fontsize': 'large',
          'figure.figsize': (10,10),
          'font.size': 10,
          'figure.dpi': 300,
          'axes.labelsize': 'x-large',
          'axes.titlesize':'large',
          'xtick.labelsize':12,
          'ytick.labelsize':12,
          'axes.spines.right': False,
          'axes.spines.top': False}

plt.rcParams.update(params)

#Define color palettes
##The colors were chosen from: https://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=5
cold_palette = ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84']
cold_palette_10 = ['#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494','#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494']
hot_palette = ['#ffffe5', '#fff7bc', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#8c2d04']
hot_palette_10 = ['#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#993404','#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#993404']
green_palette = ['#66c2a4', '#41ae76', '#238b45', '#005824']


########################################################################################
#Plotting the correlation between the signature expression and the FACS scores
########################################################################################
if args.path_dot_product is not None:
    #Define the parameters
    behavioral_col = 'FACS'
    df_behavioral = pd.read_csv(args.path_behavioral)
    x = 'Pattern expression'
    y = 'FACS score'
    idx = 0 #Adjust to change the color

    #Plot the correlation 
    signature_dot_prod=np.load(args.path_dot_product)
    plot_FACS_pattern(x, y, signature_dot_prod, df_behavioral[behavioral_col], path_output=args.path_output, idx=idx, palette=green_palette)

########################################################################################
#Plotting the signature weights
########################################################################################
#Define the parameters
coords_to_plot = {'x':[46,12,4,-4,-42],
           'y':[-12],
           'z':[-12,-2,20,42,54,68]}

#Plot the signature weights
plotting_signature_weights(args.path_feps, coords_to_plot, args.path_output)






