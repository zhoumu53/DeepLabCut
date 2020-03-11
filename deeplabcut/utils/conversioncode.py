"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os, pickle, yaml
import pandas as pd
from pathlib import Path
import numpy as np
from deeplabcut.utils import auxiliaryfunctions

def convertannotationdata_fromwindows2unixstyle(config,userfeedback=True,win2linux=True):
    """
    Converts paths in annotation file (CollectedData_*user*.h5) in labeled-data/videofolder1, etc.

    from windows to linux format. This is important when one e.g. labeling on Windows, but
    wants to re-label/check_labels/ on a Linux computer (and vice versa).

    Note for training data annotated on Windows in Linux this is not necessary, as the data
    gets converted during training set creation.

    config : string
        Full path of the config.yaml file as a string.

    userfeedback: bool, optional
        If true the user will be asked specifically for each folder in labeled-data if the containing csv shall be converted to hdf format.

    win2linux: bool, optional.
        By default converts from windows to linux. If false, converts from unix to windows.
    """
    cfg = auxiliaryfunctions.read_config(config)
    folders = [Path(config).parent / 'labeled-data' / Path(vid).stem for vid in cfg['video_sets']]

    for folder in folders:
        if userfeedback:
            print("Do you want to convert the annotationdata in folder:", folder, "?")
            askuser = input("yes/no")
        else:
            askuser="yes"

        if askuser=='y' or askuser=='yes' or askuser=='Ja' or askuser=='ha':
            fn=os.path.join(str(folder),'CollectedData_' + cfg['scorer'])
            Data = pd.read_hdf(fn+'.h5', 'df_with_missing')
            if win2linux:
                convertpaths_to_unixstyle(Data,fn)
            else:
                convertpaths_to_windowsstyle(Data,fn)

def convertpaths_to_unixstyle(Data,fn):
    ''' auxiliary function that converts paths in annotation files:
        labeled-data\\video\\imgXXX.png to labeled-data/video/imgXXX.png '''
    Data.to_csv(fn + "windows" + ".csv")
    Data.to_hdf(fn + "windows" + '.h5', 'df_with_missing', format='table', mode='w')
    Data.index = Data.index.str.replace('\\', '/')
    Data.to_csv(fn + ".csv")
    Data.to_hdf(fn + '.h5', 'df_with_missing', format='table', mode='w')
    return Data

def convertpaths_to_windowsstyle(Data,fn):
    ''' auxiliary function that converts paths in annotation files:
        labeled-data/video/imgXXX.png to labeled-data\\video\\imgXXX.png '''
    Data.to_csv(fn + "unix" + ".csv")
    Data.to_hdf(fn + "unix" + '.h5', 'df_with_missing', format='table', mode='w')
    Data.index = Data.index.str.replace('/', '\\')
    Data.to_csv(fn + ".csv")
    Data.to_hdf(fn + '.h5', 'df_with_missing', format='table', mode='w')
    return Data


## ConvertingMulti2Standard...
def conversioncodemulti2single(config,userfeedback=True,multi2single=True):
    """
    TODO: TBA.
    """
    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg['video_sets'].keys()
    video_names = [Path(i).stem for i in videos]
    folders = [Path(config).parent / 'labeled-data' /Path(i) for i in video_names]
    if scorer==None:
        scorer=cfg['scorer']

    for folder in folders:
        try:
            if userfeedback==True:
                print("Do you want to convert the labeled data in folder:", folder, "?")
                askuser = input("yes/no")
            else:
                askuser="yes"

            if askuser=='y' or askuser=='yes' or askuser=='Ja' or askuser=='ha': # multilanguage support :)
                fn=os.path.join(str(folder),'CollectedData_' + cfg['scorer'] + '.h5')
                data=pd.read_hdf(fn)

                #nlines,numcolumns=data.shape
                if multi2single: #cfg.get('multianimalproject', False):
                    print("Multi-animal data conversion...")
                    #orderofindividuals=list(data.values[0,1:])
                    #orderofbpincsv=list(data.values[1,1:])
                    xyvalue=data.columns.get_level_values(2)
                    scorers=data.columns.get_level_values(0) #len(orderofbpincsv)*[scorer]
                    bpts=data.columns.get_level_values(1)
                    orderofindividuals=['spider1']*len(bpts)
                    #xyvalue=int(len(orderofbpincsv)/2)*['x', 'y']
                    index=pd.MultiIndex.from_arrays(np.vstack([scorers,orderofindividuals,bpts,xyvalue]),names=['scorer', 'individuals','bodyparts', 'coords'])

                    imageindex=data.index #list(data.values[3:,0])
                    '''
                    print("Num of images in index:", len(imageindex))
                    images=[fns for fns in os.listdir(str(folder)) if '.png' in fns]
                    print("Num of images in folder:", len(images))

                    #assert(len(orderofbpincsv)==len(cfg['bodyparts']))
                    print(orderofbpincsv)
                    print(cfg['bodyparts'])

                    #TODO: test len of images vs. len of imagenames for another sanity check
                    #index = pd.MultiIndex.from_product([[scorer], orderofindividuals, orderofbpincsv, ['x', 'y']],names=['scorer', 'individuals','bodyparts', 'coords'])
                    '''
                    #frame = pd.DataFrame(np.array(data.values[3:,1:],dtype=float), columns = index, index = imageindex)
                    frame = pd.DataFrame(np.array(data.values,dtype=float), columns = index, index = imageindex)
                    print(frame.head())
                else:
                    #orderofbpincsv=list(data.values[0,1:-1:2])
                    #imageindex=list(data.values[2:,0])

                    #assert(len(orderofbpincsv)==len(cfg['bodyparts']))
                    #print(orderofbpincsv)
                    #print(cfg['bodyparts'])

                    xyvalue=data.columns.get_level_values(3)
                    scorers=data.columns.get_level_values(0) #len(orderofbpincsv)*[scorer]
                    bpts=data.columns.get_level_values(2)
                    #orderofindividuals=['spider1']*len(bpts)
                    #xyvalue=int(len(orderofbpincsv)/2)*['x', 'y']
                    index=pd.MultiIndex.from_arrays(np.vstack([scorers,bpts,xyvalue]),names=['scorer','bodyparts', 'coords'])
                    imageindex=data.index #list(data.values[3:,0])

                    #TODO: test len of images vs. len of imagenames for another sanity check
                    #index = pd.MultiIndex.from_product([[scorer], orderofbpincsv, ['x', 'y']],names=['scorer', 'bodyparts', 'coords'])
                    frame = pd.DataFrame(np.array(data.values,dtype=float), columns = index, index = imageindex)
                    print(frame.head())
                if cfg.get('multianimalproject', False):
                    data.to_hdf(os.path.join(str(folder),'CollectedData_'+ cfg['scorer']+"single.h5"), key='df_with_missing', mode='w')
                    data.to_csv(os.path.join(str(folder),'CollectedData_'+ cfg['scorer']+"single.csv"))
                else:
                    data.to_hdf(os.path.join(str(folder),'CollectedData_'+ cfg['scorer']+"multi.h5"), key='df_with_missing', mode='w')
                    data.to_csv(os.path.join(str(folder),'CollectedData_'+ cfg['scorer']+"multi.csv"))

                frame.to_hdf(os.path.join(str(folder),'CollectedData_'+ cfg['scorer']+".h5"), key='df_with_missing', mode='w')
                frame.to_csv(os.path.join(str(folder),'CollectedData_'+ cfg['scorer']+".csv"))

        except FileNotFoundError:
            print("Attention:", folder, "does not appear to have labeled data!")

def convertcsv2h5(config,userfeedback=True,scorer=None):
    """
    Convert (image) annotation files in folder labeled-data from csv to h5.
    This function allows the user to manually edit the csv (e.g. to correct the scorer name and then convert it into hdf format).
    WARNING: conversion might corrupt the data.

    config : string
        Full path of the config.yaml file as a string.

    userfeedback: bool, optional
        If true the user will be asked specifically for each folder in labeled-data if the containing csv shall be converted to hdf format.

    scorer: string, optional
        If a string is given, then the scorer/annotator in all csv and hdf files that are changed, will be overwritten with this name.

    Examples
    --------
    Convert csv annotation files for reaching-task project into hdf.
    >>> deeplabcut.convertcsv2h5('/analysis/project/reaching-task/config.yaml')

    --------
    Convert csv annotation files for reaching-task project into hdf while changing the scorer/annotator in all annotation files to Albert!
    >>> deeplabcut.convertcsv2h5('/analysis/project/reaching-task/config.yaml',scorer='Albert')
    --------
    """
    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg['video_sets'].keys()
    video_names = [Path(i).stem for i in videos]
    folders = [Path(config).parent / 'labeled-data' /Path(i) for i in video_names]
    if scorer==None:
        scorer=cfg['scorer']

    for folder in folders:
        try:
            if userfeedback==True:
                print("Do you want to convert the csv file in folder:", folder, "?")
                askuser = input("yes/no")
            else:
                askuser="yes"

            if askuser=='y' or askuser=='yes' or askuser=='Ja' or askuser=='ha': # multilanguage support :)
                fn=os.path.join(str(folder),'CollectedData_' + cfg['scorer'] + '.csv')
                data=pd.read_csv(fn)

                #nlines,numcolumns=data.shape

                orderofbpincsv=list(data.values[0,1:-1:2])
                imageindex=list(data.values[2:,0])

                #assert(len(orderofbpincsv)==len(cfg['bodyparts']))
                print(orderofbpincsv)
                print(cfg['bodyparts'])

                #TODO: test len of images vs. len of imagenames for another sanity check

                index = pd.MultiIndex.from_product([[scorer], orderofbpincsv, ['x', 'y']],names=['scorer', 'bodyparts', 'coords'])
                frame = pd.DataFrame(np.array(data.values[2:,1:],dtype=float), columns = index, index = imageindex)

                frame.to_hdf(os.path.join(str(folder),'CollectedData_'+ cfg['scorer']+".h5"), key='df_with_missing', mode='w')
                frame.to_csv(fn)
        except FileNotFoundError:
            print("Attention:", folder, "does not appear to have labeled data!")

def analyze_videos_converth5_to_csv(videopath,videotype='.avi'):
    """
    By default the output poses (when running analyze_videos) are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position \n
    in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) \n
    in the same directory, where the video is stored. If the flag save_as_csv is set to True, the data is also exported as comma-separated value file. However,
    if the flag was *not* set, then this function allows the conversion of all h5 files to csv files (without having to analyze the videos again)!

    This functions converts hdf (h5) files to the comma-separated values format (.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.

    Parameters
    ----------

    videopath : string
        A strings containing the full paths to videos for analysis or a path to the directory where all the videos with same extension are stored.

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\nOnly videos with this extension are analyzed. The default is ``.avi``

    Examples
    --------

    Converts all pose-output files belonging to mp4 videos in the folder '/media/alex/experimentaldata/cheetahvideos' to csv files.
    deeplabcut.analyze_videos_converth5_to_csv('/media/alex/experimentaldata/cheetahvideos','.mp4')

    """
    start_path=os.getcwd()
    os.chdir(videopath)
    Videos=[fn for fn in os.listdir(os.curdir) if (videotype in fn) and ('_labeled.mp4' not in fn)] #exclude labeled-videos!

    Allh5files=[fn for fn in os.listdir(os.curdir) if (".h5" in fn) and ("resnet" in fn)]

    for video in Videos:
         vname = Path(video).stem
         #Is there a scorer for this?
         PutativeOutputFiles=[fn for fn in Allh5files if vname in fn]
         for pfn in PutativeOutputFiles:
             scorer=pfn.split(vname)[1].split('.h5')[0]
             if "DeepCut" in scorer:
                 DC = pd.read_hdf(pfn, 'df_with_missing')
                 print("Found output file for scorer:", scorer)
                 print("Converting to csv...")
                 DC.to_csv(pfn.split('.h5')[0]+'.csv')

    os.chdir(str(start_path))
    print("All pose files were converted.")


def merge_windowsannotationdataONlinuxsystem(cfg):
    ''' If a project was created on Windows (and labeled there,) but ran on unix then the data folders
    corresponding in the keys in cfg['video_sets'] are not found. This function gets them directly by
    looping over all folders in labeled-data '''

    AnnotationData = []
    data_path = Path(cfg['project_path'],'labeled-data')
    annotationfolders=[fn for fn in os.listdir(data_path) if "_labeled" not in fn]
    print("The following folders were found:", annotationfolders)
    for folder in annotationfolders:
        filename = os.path.join(data_path , folder, 'CollectedData_'+cfg['scorer']+'.h5')
        try:
            data = pd.read_hdf(filename,'df_with_missing')
            AnnotationData.append(data)
        except FileNotFoundError:
            print(filename, " not found (perhaps not annotated)")

    return AnnotationData
