import os
import pickle
import numpy as np
from numpy.linalg import norm
from nilearn.image import math_img
import random
import ptitprince as pt
from nilearn import plotting
import matplotlib.ticker as ticker
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


def violin_plot_performance(df, metric='pearson_r', color='#fe9929', path_output='', filename='violin_plot'):
    """
    Plot the violin plot associated to the model performance across all folds

    Parameters
    ----------
    df: dataFrame
        dataFrame containing the performance metrics for all folds
    metric: string
        performance metric to use as defined in the dataFrame
    color: string
        color to use for the plots
    path_output: string
        path for saving the output
    filename: string
        name of the output file
    
    Code adapted from https://github.com/mpcoll/coll_painvalue_2021/tree/main/figures
    """
    colp = color
    labelfontsize = 7
    ticksfontsize = np.round(labelfontsize*0.8)

    fig1, ax1 = plt.subplots(figsize=(0.6, 1.5))
    pt.half_violinplot(y=df[metric], inner=None,
                    width=0.6,
                        offset=0.17, cut=1, ax=ax1,
                        color=colp,
                        linewidth=1, alpha=1, zorder=19)
    sns.stripplot(y=df[metric],
                    jitter=0.08, ax=ax1,
                    color=colp,
                    linewidth=0.2, alpha=0.6, zorder=1)
    sns.boxplot(y=df[metric], whis=np.inf, linewidth=1, ax=ax1,
                width=0.1, boxprops={"zorder": 10, 'alpha': 0.5},
                whiskerprops={'zorder': 10, 'alpha': 1},
                color=colp,
                medianprops={'zorder': 11, 'alpha': 0.9})

    ax1.axhline(0, linestyle='--', color='k', linewidth=0.6)
    ax1.set_ylabel(metric, fontsize=labelfontsize, labelpad=0.7)
    ax1.tick_params(axis='y', labelsize=ticksfontsize)
    ax1.set_xticks([], [])
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(1)
    ax1.tick_params(width=1, direction='out', length=4)
    fig1.tight_layout()
    plt.savefig(os.path.join(path_output, f'{filename}.svg'),transparent=False, bbox_inches='tight', facecolor='white', dpi=600)


def reg_plot_performance(y_test, y_pred, path_output='', filename='regplot'):
    """
    Plot the regression plot associated to the model performance. One regression line will be plotted by fold

    Parameters
    ----------
    y_test: list
        list containing the values of y in the test set for each fold
    y_pred: list
        list containing the values of the predicted y for each fold
    path_output: string
        path for saving the output
    filename: string
        name of the output file
        
    Code adapted from https://github.com/mpcoll/coll_painvalue_2021/tree/main/figures
    """
    fig1, ax1 = plt.subplots(figsize=(4,4))
    ax1.set_xlim([0,42])
    ax1.set_ylim([0,23])
    for idx, elem in enumerate(y_test):
        df = pd.DataFrame(list(zip(elem, y_pred[idx])), columns=['Y_true','Y_pred'])
        sns.regplot(data=df, x='Y_true', y='Y_pred',
                    ci=None, scatter=False, color=hot_palette_10[idx],
                    ax=ax1, line_kws={'linewidth':1.4}, truncate=False)
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.xlabel('FACS score')
    plt.ylabel('Cross-validated prediction')
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2.6)
    ax1.tick_params(width=2.6, direction='out', length=10)
    plt.savefig(os.path.join(path_output, f'{filename}.svg'),transparent=False, bbox_inches='tight', facecolor='white', dpi=600)


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


def plot_similarity_from_network(similarity_feps, signature=None,row=0,col=0, ax=None, ylim=[-0.26,0.26], palette="crest", rotation=90, y='Cosine similarity',path_output=''):
    df = pd.DataFrame({'Network':[tup[0] for tup in similarity_feps], y:[tup[1] for tup in similarity_feps]})
    sns.barplot(data=df,x='Network', y=y, palette=palette, ax=ax[row, col]).set(title=f'{signature}')
    ax[row, col].tick_params(axis='x',rotation=rotation)
    if col > 0:
        ax[row, col].set(xlabel=None, ylim=(ylim[0],ylim[1]), yticklabels=[], ylabel=None)
    else:
        ax[row, col].set(xlabel=None, ylim=(ylim[0],ylim[1]))
    plt.axhline(0, color="black")
    


parser = ArgumentParser()
parser.add_argument("--path_output", type=str, default=None)
parser.add_argument("--path_feps", type=str, default=None)
parser.add_argument("--path_behavioral", type=str, default=None)
parser.add_argument("--path_dot_product", type=str, default=None)
parser.add_argument("--path_performance", type=str, default=None)
parser.add_argument("--path_y_test", type=str, default=None)
parser.add_argument("--path_y_pred", type=str, default=None)
parser.add_argument("--path_siips_similarity_networks", type=str, default=None)
parser.add_argument("--path_pvp_similarity_networks", type=str, default=None)
parser.add_argument("--path_maths_similarity_networks", type=str, default=None)
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
#Plotting the performance of the model
########################################################################################
if args.path_performance is not None:
    df_performance = pd.read_csv(args.path_performance)
    metric = 'pearson_r'
    color='#fe9929'
    filename='violin_plot'

    #Violin plot visualization
    violin_plot_performance(df_performance, metric=metric, color=color, path_output=args.path_output, filename=filename)

if args.path_y_test is not None:
    #Define the parameters
    y_test = load_pickle(args.path_y_test)
    y_pred = load_pickle(args.path_y_pred)
    filename='regplot'

    #Regression plot visualization
    reg_plot_performance(y_test, y_pred, path_output=args.path_output, filename=filename)

########################################################################################
#Plotting the signature weights
########################################################################################
if args.path_feps is not None:
    #Define the parameters
    coords_to_plot = {'x':[46,12,4,-4,-42],
            'y':[-12],
            'z':[-12,-2,20,42,54,68]}

    #Plot the signature weights
    plotting_signature_weights(args.path_feps, coords_to_plot, args.path_output)

########################################################################################
#Spatial similarity matrix
########################################################################################


########################################################################################
#Spatial similarity across networks
########################################################################################
if args.path_siips_similarity_networks is not None:
    #Define parameters
    ymin = -0.26
    ymax = 0.26
    y = 'Cosine similarity'
    ncols = 3

    similarity_feps_siips = load_pickle(args.path_siips_similarity_networks)[0]
    similarity_feps_pvp = load_pickle(args.path_pvp_similarity_networks)[0]
    similarity_feps_maths = load_pickle(args.path_maths_similarity_networks)[0]
    similarity_feps_signatures = [similarity_feps_siips,similarity_feps_pvp, similarity_feps_maths]
    signatures = ['SIIPS-1', 'PVP', 'MAThS']

    fig,ax = plt.subplots(nrows=2,ncols=ncols, sharex=False)
    for idx, elem in enumerate(similarity_feps_signatures):
        if idx < ncols:
            row = 0
            col = idx
        else:
            row = 1
            col = idx-ncols
        plot_similarity_from_network(elem, signatures[idx], row=row,col=col,ax=ax, ylim=[ymin,ymax], palette=cold_palette[1:], rotation=90, y=y)
        
    fig.delaxes(ax[1][0])
    fig.delaxes(ax[1][1])
    fig.delaxes(ax[1][2])
    plt.tight_layout()
    plt.savefig(os.path.join(args.path_output, f'similarity_feps_cortical_networks'),transparent=False, bbox_inches='tight', facecolor='white', dpi=600)