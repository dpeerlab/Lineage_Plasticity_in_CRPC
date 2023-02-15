from typing import Optional
import phenograph
import scanpy as sc
from anndata import AnnData
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import os
import pandas as pd
import numpy as np
from scipy.stats import entropy
import subprocess

def CalcPlasticity (
        adata: AnnData, 
        WT_cell_type: pd.core.series.Series = None,
        n_neighbors: int = 25, 
        num_dcs: int = 20, 
        diffmap_rep: str = 'X_diffmap_wtdeg', 
        sample_key: Optional[str] = 'batch', 
        initial_sample: Optional[str] = 'Wk0'
) -> AnnData:
    """Function for computing per-cell plasticity scores based on cell-type probabilities. Cell-type  probabilties are based on random walk classification using a diffusion graph as well as training labels corresponding to wild-type Basal and Luminal cell types. Per-cell plasticity is then measured as entropy of cell-type probabilties, 
    :param adata: adata object of single cells from the combined time-course  
    :param WT_cell_type: Series object of Basal and Luminal cell type labels for Wild-type cells (Pandas Series). If not specified, will be extracted from the column name sample_key + '_plus_wt_cells'.
    :param n_neighbors: Number of kNN neighbors. Default 25 (int)  
    :param num_dcs: Number of diffusion components to retain. Default 20 (int)
    :param diffmap_rep: Key name in adata.uns for diffusion map (str)
    :param sample_key: Column name for samples/batches stored in adata.obs (str) 
    :param initial_sample: Name of the wild-type sample (str)
    :return: Updated adata object
    """

    # Error check

    if diffmap_rep not in adata.obsm.keys():
        raise ValueError('Diffusion map representation not available.')

    if WT_cell_type == None:
        WT_cell_type = adata.obs.loc[adata.obs.loc[:,sample_key]==initial_sample, 
                                     sample_key + '_plus_wt_cells'].str.replace('%s: ' % initial_sample,'').astype('category')

    cell_types = sorted(WT_cell_type.cat.categories)

    dm_merge = pd.DataFrame(adata.obsm[diffmap_rep],
                            index = adata.obs.index).loc[:,:num_dcs]

    train = np.empty((len(cell_types),),dtype=object)
    for c,cell_type in enumerate(cell_types):
        labels = WT_cell_type.index[WT_cell_type == cell_type]
        ind = [adata.obs.index.get_loc(i) for i in labels]
        train[c] = dm_merge.iloc[ind,:]
    test = dm_merge 

    ct_metacluster_preds_dm = phenograph.classify(train, test, k=n_neighbors, metric='euclidean')

    pval_df = pd.DataFrame(ct_metacluster_preds_dm[1],
                           index = test.index,
                           columns = cell_types)


    pval_df.columns = 'pval_randomwalk_' + pval_df.columns

    tmp = pval_df.subtract(pval_df.min(axis=1), axis=0)
    pval_df = tmp.div(tmp.sum(axis=1), axis=0)

    for i in pval_df.columns:
        adata.obs.loc[:, i] = pval_df.loc[:,i]

    H = pval_df.apply(entropy, axis=1)
    adata.obs.loc[:,'plasticity_randomwalk_entropy'] = H

    return adata


def BoxplotProbabilities (
        adata: AnnData,
        sample_key: Optional[str] = 'batch',
        initial_sample: Optional[str] = 'Wk0',
        save: Optional[str] = None,
        colors:Optional[list] = None
):
    """Function for plotting the distribution of random walk probabilties across samples for each Basal and Luminal cell type 
    :param adata: adata object of single cells from the combined time-course
    :param sample_key: Column name for samples/batches stored in adata.obs (str)
    :param initial_sample: Name of the wild-type sample (str)
    :param save: File name to save figure
    :param colors: List of colors for boxplot (length corresponding to number of samples)
    """

    # Error check

    if ~adata.obs.columns.str.contains('pval_randomwalk').any():
        raise RuntimeError('Run RandomWalk.CalcProbabilities first.')

    labels = adata.obs.columns[adata.obs.columns.str.contains('pval_randomwalk')]

    plot_df = adata.obs.loc[adata.obs.loc[:,sample_key] != initial_sample, labels].melt(var_name = 'Cell Type', value_name = 'p-value')
    plot_df.loc[:,'Cell Type'] = plot_df.loc[:,'Cell Type'].str.replace('pval_randomwalk_','')

    sc.set_figure_params(fontsize = 20)
    sns.set_style('ticks')

    fig, ax = plt.subplots(1,1, figsize = (5, 5))

    if colors == None:
        sns.boxplot(data = plot_df, x = 'Cell Type', y = 'p-value', ax=ax, showfliers = False)
    else:
        sns.boxplot(data = plot_df, x = 'Cell Type', y = 'p-value', ax=ax, showfliers = False, palette=colors)

    ax.set_xticklabels(ax.get_xticklabels(), rotation = 30)
    ax.set_ylabel('Cell type probability')
    ax.axvline(x=2.5, color = 'black', linestyle='--')
    if save != None:
        plt.savefig(save)


def BarycentricProbabilities (
        adata: AnnData,
        save: str = None,
        sample_key: Optional[str] = 'batch',
        initial_sample: Optional[str] = 'Wk0',
        num_rows: Optional[int] = 3,
        width: Optional[float] = 7,
        height: Optional[float] = 7,
):
    """Function for plotting random walk probabilties for the 3 predominant Basal and Luminal cell types represented among mutant cells  
    :param adata: adata object of single cells from the combined time-course
    :param save: File name to save figure. Must be PDF
    :param sample_key: Column name for samples/batches stored in adata.obs (str)
    :param initial_sample: Name of the wild-type sample (str)
    :param num_rows: Number of rows of subpanels in figure
    :param width: Width of figure
    :param height: Height of figure
    """

    # Error check

    if ~adata.obs.columns.str.contains('pval_randomwalk').any():
        raise RuntimeError('Run RandomWalk.CalcProbabilities first.')

    rscript_path = str(pathlib.Path().resolve())

    if save == None:
        save = os.path.join(rscript_path,'figures','ternary.RandomWalk.pdf')

    obs_input_path = str(rscript_path)

    adata.obs.to_csv(os.path.join(obs_input_path, 'tmp.input.csv'), sep ='\t')

    subprocess.call("~/anaconda3/envs/r_4.0.3c/bin/Rscript --vanilla %s/_BarycentricPlot.R %s/tmp.input.csv %s %s %s %d %d %d" % (rscript_path, obs_input_path, sample_key, initial_sample, save, num_rows, width, height), shell=True)
    #NEED TO CHANGE RSCRIPT PATH

    os.remove(os.path.join(obs_input_path, 'tmp.input.csv'))

def PlotEntropy (
        adata: AnnData,
        sample_key: Optional[str] = 'batch',
        save: Optional[str] = None,
        colors:Optional[list] = None
        width: Optional[float] = 5,
        height: Optional[float] = 5,
):
    """Function for plotting entropy of random walk probabilties as a per-cell measure of plasticity
    :param adata: adata object of single cells from the combined time-course
    :param sample_key: Column name for samples/batches stored in adata.obs (str)
    :param save: File name to save figure
    :param colors: List of colors for boxplot (length corresponding to number of samples)
    :param width: Width of figure
    :param height: Height of figure
    """

    # Error check

    if ~adata.obs.columns.str.contains('pval_randomwalk').any():
        raise RuntimeError('Run RandomWalk.CalcProbabilities first.')

    plot_df = adata.obs.loc[:,[sample_key,'time','treatment','plasticity_randomwalk_entropy']]
    plot_df.columns = ['Sample','Timepoint','Treatment','Entropy']

    sc.set_figure_params(fontsize = 16)
    sns.set_style('ticks')

    fig, ax= plt.subplots(1,1,figsize=(width,height))

    if colors == None:
        sns.boxplot(data=plot_df, x='Sample', y='Entropy', ax=ax,showfliers = False)
    else:
        sns.boxplot(data=plot_df, x='Sample', y='Entropy', ax=ax,showfliers = False, palette=colors)

    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
    ax.set_title('Plasticity')
    ax.set_ylabel('Entropy of\ncell type probabilities')

    if save != None:
        plt.savefig(save)
