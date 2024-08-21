# NOTE: to use the violin_plot_performance function, you'll need to use an virtual 
# environment with seaborn version 0.11.0


import os
import pickle
import json
import numpy as np
import networkx as nx
from nilearn import plotting
import matplotlib.ticker as ticker
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from argparse import ArgumentParser
from joypy import joyplot

if sns.__version__ == '0.11.0': 
    import ptitprince as pt

def load_pickle(path):
    file = open(path,'rb')
    object_file = pickle.load(file)
    file.close()
    return object_file


def load_json(path):
    file = open(path)
    object_file = json.load()
    file.close()
    return object_file


def plot_FACS_pattern(x, y, signature_dot_prod, FACS, path_output='', idx=0, palette=None, extension='svg'):
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
    plt.savefig(os.path.join(path_output, f'lmplot_{x}_expression_FACS.{extension}'))


def violin_plot_performance(df, metric='pearson_r', figsize = (0.6, 1.5), color='#fe9929', linewidth_half=1, alpha_half=1, linewidth_strip=0.2, size_strip=5, linewidth_box=1, linewidth_axh=0.6, linewidth_spine=1, path_output='', filename='violin_plot', extension='svg'):
    """
    Plot the violin plot associated to the model performance across all folds

    Parameters
    ----------
    df: dataFrame
        dataFrame containing the performance metrics for all folds
    metric: string
        performance metric to use as defined in the dataFrame
    figsize: tuple
        figure size
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

    fig1, ax1 = plt.subplots(figsize=figsize)
    pt.half_violinplot(y=df[metric], inner=None,
                    width=0.6,
                        offset=0.17, cut=1, ax=ax1,
                        color=colp,
                        linewidth=linewidth_half, alpha=alpha_half, zorder=19)
    sns.stripplot(y=df[metric],
                    jitter=0.08, ax=ax1,
                    color=colp,
                    linewidth=linewidth_strip, alpha=0.6, zorder=1, size=size_strip)
    sns.boxplot(y=df[metric], whis=np.inf, linewidth=linewidth_box, ax=ax1,
                width=0.1, boxprops={"zorder": 10, 'alpha': 0.5},
                whiskerprops={'zorder': 10, 'alpha': 1},
                color=colp,
                medianprops={'zorder': 11, 'alpha': 0.9})

    ax1.axhline(0, linestyle='--', color='k', linewidth=linewidth_axh)
    ax1.set_ylabel(metric, fontsize=labelfontsize, labelpad=0.7)
    ax1.tick_params(axis='y', labelsize=ticksfontsize)
    ax1.set_xticks([], [])
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(linewidth_spine)
    ax1.tick_params(width=1, direction='out', length=4)
    fig1.tight_layout()
    plt.savefig(os.path.join(path_output, f'{filename}.{extension}'),transparent=False, bbox_inches='tight', facecolor='white', dpi=600)


def reg_plot_performance(y_test, y_pred, path_output='', filename='regplot', extension='svg'):
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
    plt.savefig(os.path.join(path_output, f'{filename}.{extension}'),transparent=False, bbox_inches='tight', facecolor='white', dpi=600)


def plotting_signature_weights(path_signature, coords_to_plot, path_output, extension='svg'):
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
            
            fig.savefig(os.path.join(path_output, f'{filename}_{axis}_{str(c)}.{extension}'),
                        transparent=True, bbox_inches='tight', dpi=600)


def plot_similarity_matrix(similarity_matrix, labels, path_output, extension='svg'):
    """
    Parameters
    ----------
    similarity_matrix: numpy.ndarray
        array containing the similarities values between pain-related signatures
    labels: list
        list containing the name of the signatures
    path_output: string
        path for saving the output(s)
    """
    # Create a mask
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))

    fig = plt.figure(figsize=(16, 12))
    sns.heatmap(similarity_matrix, mask=mask, center=0, annot=False,
                fmt='.2f', square=True, cmap=sns.diverging_palette(220, 20, as_cmap=True), vmin=-0.10, vmax=0.10)
    plt.xticks(np.arange(len(labels)) + .5,labels=labels)
    plt.yticks(np.arange(len(labels)) + .5,labels=labels)
    plt.show()
    fig.savefig(os.path.join(path_output, f'similarity_matrix_full_brain.{extension}'),transparent=True, bbox_inches='tight', facecolor='white', dpi=600)


def plot_network_diagram(dict_similarity, path_output, extension='svg'):
    """
    Parameters
    ----------
    dict_similarity: dict
        dictionary containing the significant similarity values
    path_output: string
        path for saving the output(s)
    
    See also
    --------
    https://networkx.org
    """
    G = nx.Graph()
    G.add_node("FEPS")
    G.add_node("SIIPS")
    G.add_node("NPS")
    G.add_node("TPAS")
    G.add_node("PVP")

    pos = nx.spring_layout(G, seed=9)
    G.add_edges_from([("FEPS", "SIIPS", {"weight": dict_similarity["feps_siips"]}),
                 ("FEPS", "PVP", {"weight": dict_similarity["feps_pvp"]}),
                 ("SIIPS", "NPS", {"weight": dict_similarity["siips_nps"]}),
                 ("SIIPS", "PVP", {"weight": dict_similarity["siips_pvp"]}),
                 ("SIIPS", "TPAS", {"weight": dict_similarity["siips_tpas"]})])
    options= {"node_color": "slategray",
              "edge_color": ["#FA8072", "#FA8072", "#FC5A50", "#ADD8E6", "#FC5A50"],
              "width": 2}
    plt.figure(figsize=(3,3))
    nx.draw(
        G, pos, **options
    )
    plt.savefig(os.path.join(path_output, f'graph_network_similarity.{extension}'), dpi=300, transparent=True)


def plot_similarity_from_network(dict_similarity, path_output, label=None, extension='svg'):
    """
    Parameters
    ----------
    dict_similarity: dict
        dictionary containing the similarity values and null distributions across networks
    path_output: string
        path for saving the output(s)
    labels: string
        name of the signature
    """
    hls = sns.color_palette("hls", 7)
    networks_name = []
    networks_similarity = []
    for k in dict_similarity['distr'].keys():
        networks_name.append([k]*len(dict_similarity['distr'][k]))
        networks_similarity.append(dict_similarity['distr'][k])

    df = pd.DataFrame({'Values': networks_similarity, 'Network': networks_name})

    joyplot(df, 
            by = 'Network', column = 'Values', 
            color = hls, alpha=0.5, fade = False, 
            overlap=0.1, x_range=[-0.3,0.3], figsize=(3,3))
    plt.axvline(x = dict_similarity['similarity']['dan'], ymin = 0.875, ymax = 0.985, color = hls[0], lw=3)
    plt.axvline(x = dict_similarity['similarity']['dmn'], ymin = 0.73, ymax = 0.84, color = hls[1], lw=3)
    plt.axvline(x = dict_similarity['similarity']['fpn'], ymin = 0.59, ymax = 0.70, color = hls[2], lw=3)
    plt.axvline(x = dict_similarity['similarity']['ln'], ymin = 0.45, ymax = 0.56, color = hls[3], lw=3)
    plt.axvline(x = dict_similarity['similarity']['smn'], ymin = 0.305, ymax = 0.415, color = hls[4], lw=3)
    plt.axvline(x = dict_similarity['similarity']['van'], ymin = 0.165, ymax = 0.275, color = hls[5], lw=3)
    plt.axvline(x = dict_similarity['similarity']['vn'], ymin = 0.022, ymax = 0.132, color = hls[6], lw=3)
    plt.xlabel("Null distributions")
    plt.ylabel("Cortical networks")
    plt.savefig(os.path.join(path_output, f"ridgeplot_{label}_cortical_networks.{extension}"), 
                dpi=300, transparent=True)
    

def plot_behav_across_trials(data, y, palette, path_output, extension="svg"):
    """
    Parameters
    ----------
    data : dataFrame
        dataframe containing the behavioral data to plot, with a trial column and 
        a run column
    y: string
        name of the y axis
    palette: list
        color palette to use
    path_output: string
        path for saving the output(s)
    """
    plt.figure(figsize=(12,4))
    sns.lineplot(data=data, x='trial', y=y, hue='run', palette=palette)
    plt.xticks(np.arange(0, 8, 1), ['1', '2', '3', '4', '5', '6', '7', '8'])
    if y == "FACS LOG":
        y_label = "log(FACS scores + 1)"
    elif y == "VAS INT": 
        y_label = "Intensity ratings"

    plt.ylabel(y_label)
    plt.xlabel("Trials")
    plt.savefig(os.path.join(path_output, f"{y}_across_trials.{extension}"), dpi=600, transparent=True)


def plot_feps_facs(data, x, y, path_output, extension='svg'):
    """
    Parameters
    ----------
    data: dataFrame
        dataFrame containing the log transformed facs scores and the feps expression scores
    x: string
        column of the dataFrame containing the feps expression scores
    y: string
        column of the dataFrame containing the log transformed FACS scores
    path_output: string
        path for saving the output(s)
    """
    fig1, ax1 = plt.subplots(figsize=(10, 10))

    sns.regplot(data, y=y, x=x, color='#66DCA0', scatter_kws={'edgecolor':'#36454F', 'alpha':0.5, 'zorder': 10})

    plt.xlabel("FEPS pattern expression scores")
    plt.ylabel("log(FACS scores + 1)")
    plt.savefig(os.path.join(path_output, f"feps_facs_regplot.{extension}"),transparent=False, bbox_inches='tight', facecolor='white', dpi=600)


def plot_feps_across_cond(data, x, y, path_output, extension='svg'):
    """
    Parameters
    ----------
    data: dataFrame
        dataFrame containing the feps expression scores and the experimental conditions
    x: string
        column of the dataFrame containing the feps expression scores
    y: string
        column of the dataFrame containing the experimental conditions
    path_output: string
        path for saving the output(s)   
    """
    hls = sns.color_palette("hls", 7)

    fig1, ax1 = plt.subplots(figsize=(10, 10))

    sns.stripplot(data=data, palette=[hls[2],hls[3],hls[4]],
                x=x, y=y, 
                linewidth=1.2, alpha=0.45, zorder=0, size=10, edgecolor='#36454F'
                )
    sns.boxplot(data=data, color='black', 
                x=x, y=y, linewidth = 2, 
                width=0.2, boxprops={"zorder": 10, 'alpha': 1},
                medianprops={'zorder': 11, 'alpha': 0.9}, showfliers=False, fill=False
            )

    plt.xlabel("FEPS pattern expression scores")

    plt.savefig(os.path.join(path_output, f"feps_pattern_expression_cond.{extension}"),transparent=False, bbox_inches='tight', facecolor='white', dpi=600)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_output", type=str, default=None)
    parser.add_argument("--path_feps", type=str, default=None)
    parser.add_argument("--path_behavioral", type=str, default=None)
    parser.add_argument("--path_feps_expression", type=str, default=None)
    parser.add_argument("--path_performance", type=str, default=None)
    parser.add_argument("--path_y_test", type=str, default=None)
    parser.add_argument("--path_y_pred", type=str, default=None)
    parser.add_argument("--path_similarity_networks", type=str, default=None)
    parser.add_argument("--path_similarity_matrix", type=str, default=None)
    parser.add_argument("--path_similarity_dict", type=str, default=None)
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

    if args.path_feps_expression is not None:
        #Define the parameters
        dot_prod_col = 'dot_prod_log'
        facs_col = 'facs_log'
        dot_prod_scores = pd.read_csv(args.path_feps_expression)

        #Plot facs x feps scores for painful trials 
        plot_feps_facs(dot_prod_scores[dot_prod_scores['condition'] != 'warm'], dot_prod_col, facs_col, args.path_output, extension='png')
        # Plot FEPS expression scores distribution across experimental conditions
        plot_feps_across_cond(dot_prod_scores, x=dot_prod_col, y="condition", path_output=args.path_output, extension='png')
        

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
    if args.path_similarity_matrix is not None:
        #Define parameters
        labels = ['FEPS', 'NPS', 'SIIPS-1', 'PVP', 'TPAS']

        similarity_matrix = np.load(args.path_similarity_matrix)
        similarity_dict = load_json(args.path_similarity_dict)
        plot_similarity_matrix(similarity_matrix, labels, args.path_output)
        plot_network_diagram(similarity_dict, args.path_output)

    ########################################################################################
    #Spatial similarity across networks
    ########################################################################################
    if args.path_similarity_networks is not None:
        label = args.path_similarity_networks.split('/')[-1].split('.')[0]
        similarity_feps_sign = load_json(args.path_similarity_networks)
        plot_similarity_from_network(similarity_feps_sign, args.path_output, label)

    ########################################################################################
    #Behavioral scores
    ########################################################################################
    
    if args.path_behavioral is not None:
        df_behavioral = pd.read_csv(args.path_behavioral)

        # LOG TRANSFORMED FACS
        y_facs = 'FACS LOG'
        palette_facs = ['#fec44f', '#ec7014']
        color_facs = '#fe9929'

        # Log transformed FACS across trials
        plot_behav_across_trials(df_behavioral, y_facs, palette_facs, args.path_output, extension='png')

        # Log transformed FACS distribution
        violin_plot_performance(df_behavioral, metric=y_facs, figsize = (3, 3.8), color=color_facs, linewidth_half=2, alpha_half=0.8, linewidth_strip=1.2, size_strip=8, linewidth_box=2, linewidth_axh=2, linewidth_spine=2, path_output=args.path_output, filename='dist_FACS_log', extension='png')

        # INTENSITY RATINGS
        y_int = 'VAS INT'
        palette_int =['#7fcdbb', '#1d91c0']
        color_int = '#41b6c4'

        # Intensity rating across trials
        plot_behav_across_trials(df_behavioral, y_int, palette_int, args.path_output, extension='png')

        # Intensity rating distribution
        # NOTE: Make sure you are in a virtual environment with seaborn 0.11.0
        violin_plot_performance(df_behavioral, metric=y_int, figsize = (3, 3.8), color=color_int, linewidth_half=2, alpha_half=0.8, linewidth_strip=1.2, size_strip=8, linewidth_box=2, linewidth_axh=2, linewidth_spine=2, path_output=args.path_output, filename='dist_int_ratings', extension='png')

        if args.path_dot_product is not None:
            #Plot the correlation
            x = 'Pattern expression'
            y = 'FACS score'
            idx = 0 #Adjust to change the color

            signature_dot_prod=np.load(args.path_dot_product)
            plot_FACS_pattern(x, y, signature_dot_prod, df_behavioral[y_facs], path_output=args.path_output, idx=idx, palette=green_palette, extension='png')

