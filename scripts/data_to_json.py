import json
import os
import pandas as pd
from argparse import ArgumentParser

def read_files(path_data):
    """
    Load all functional files into Nifti format
    
    Parameters
    ----------
    path_data: string
        path containing the functional images

    Returns
    ----------
    files_hdr: list 
        list containing the path of each functional images
    """
    path_files_hdr = []

    path, _, files = next(os.walk(path_data))

    for file in files:
        if file[-3:] == "hdr":
            path_files_hdr.append(path + "//" + file)
    
    path_files_hdr.sort(reverse=True)
    path_files_hdr.sort()

    return path_files_hdr


def save_to_json(dataframe, id, target, group, files, path_output, filename_output=None):
    """
    Create and save a json file containing the relative path of the hdr fmri files,
    the target variable and the group variable

    Parameters
    ----------
    dataframe: dataFrame
        dataframe containing the behavioral data
    id: string
        name of the column containing the participants' ID
    target: string
        name of the column in dataframe containing the target variable
    group: string
        name of the column in dataframe containing the group variable
    files: list
        list of hdr file paths
    path_output: string 
        path to save the json file
    filename_output: string
        filename to use for the json output file
    """
    filename = []
    for file in files:
        filename.append(os.path.splitext(file)[0].split('/')[-1])

    if filename == dataframe[id].tolist():
        idx = files[0].find("//")
        for i, file in enumerate(files):
            files[i] = file[idx+2:]

        if group == None:
            data = {"target": dataframe[target].to_list(), "group": group, "data": files}
        else:
            data = {"target": dataframe[target].to_list(), "group": dataframe[group].to_list(), "data": files}

        if filename_output is None:
            filename_output = 'dataset.json'
        
        with open(os.path.join(path_output, filename_output), 'w') as fp:
            json.dump(data, fp)
    else: 
        print("Cannot save to json: fmri and behavioral data are not in the same order")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path_behavioral', type=str, default=None)
    parser.add_argument('--path_fmri', type=str, default=None)
    parser.add_argument('--path_output', type=str, default=None)
    parser.add_argument('--filename_output', type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.path_behavioral)
    hdr_files = read_files(args.path_fmri)

    save_to_json(df, 'CODE', 'FACS', 'GROUP', hdr_files, args.path_output)
