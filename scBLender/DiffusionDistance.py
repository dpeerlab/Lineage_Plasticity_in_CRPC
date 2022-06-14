from typing import Optional
from sklearn.metrics import pairwise_distances
import scanpy as sc
from anndata import AnnData
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import os
import pandas as pd
import numpy as np
import subprocess
import glob
import re
from scipy.io import mmwrite


def _CalcSBM_per_sample(
        adata_ind: AnnData,
        save_rds: Optional[str] = None,
        node_attribute: Optional[str] = 'Basal.Correlation',
        node_init: Optional[str] = 'Basal.vs.Luminal.by.Correlation',
) -> AnnData:
    """Function for calculating attributed stochastic block models (aSBM) on a kNN graph of metacells in a single sample. This function calls an R script, that creates an igraph object and performs aSBM using code written by Natalie Stanley, downloaded from https://github.com/stanleyn/AttributedSBM. In short, the aSBM identifies an optimal graph partition of k=2 that maximally separates basal-correlated from luminal-correlated metacells. Results from igraph and aSBM are stored in an RDS file used for downstream analysis (plotting). aSBM partitioning is also stored in the returned adata. 
    :param adata: adata object of metacells from a single sample
    :param save_rds: RDS file name to save igraph and aSBM results  
    :parma node_attribute: Name of (metacell) node attribute to perform aSBM on (default is correlation to Basal vs Luminal cells)
    :parma node_init: Name of (metacell) node attribute to initialize aSBM (default is Basal vs Luminal label based on correlation alone)
    :return: Updated adata object
    """

    # Error check
    if 'distances' not in adata_ind.obsp.keys():
        raise RuntimeError('Run sc.pp.neighbors first.')

    rscript_path = str(pathlib.Path().resolve())

    if save_rds == None:
        save_path = os.path.join(rscript_path, 'results')
        if ~os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok = True)
        save_rds = os.path.join(save_path, 'SBM_results.RDS')

    # temporary files to be read by R script file
    obs_input_file = os.path.join(rscript_path, 'tmp.input.csv')
    knn_input_file = os.path.join(rscript_path, 'tmp.input.mtx')

    # temporary file to be written in this function
    sbm_output_file = os.path.join(rscript_path, "tmp.output.txt")

    adata_ind.obs.to_csv(obs_input_file,sep='\t')
    mmwrite(knn_input_file, adata_ind.obsp['distances'])
    
    subprocess.call("~/anaconda3/envs/r_4.0.3c/bin/Rscript --vanilla %s/_Calc_aSBM.R %s %s %s %s %s %s" % (rscript_path, obs_input_file, knn_input_file, node_attribute, node_init, sbm_output_file, save_rds), shell=True)

    adata_ind.obs.loc[:,'SBM_cluster'] = pd.read_csv(sbm_output_file, sep=',').SBM_cluster

    os.remove(obs_input_file)
    os.remove(knn_input_file)
    os.remove(sbm_output_file)

    return adata_ind


def CalcSBM(
        adata: AnnData,
        individual_dir: str,
        sample_key: Optional[str] = 'batch',
        save_rds_path: Optional[str] = None,
        node_attribute: Optional[str] = 'Basal.Correlation',
        node_init: Optional[str] = 'Basal.vs.Luminal.by.Correlation',
) -> AnnData:
    """Function for calculating attributed stochastic block models (aSBM) on a kNN graph of metacells in a time-course. This function calls _CalcSBM_per_sample for each sample, and stores the aSBM partitions for each sample in the adata of the combined dataset.  
    :param adata: adata object of metacells from the combined time-course
    :param individual_dir: Directory that contains individual adata's for each sample in the time-course
    :param sample_key: Column name for samples/batches stored in adata.obs (str)
    :param node_attribute: Name of (metacell) node attribute to perform aSBM on (default is correlation to basal vs luminal cells)
    :param node_init: Name of (metacell) node attribute to initialize partition of graph (default is basal vs luminal label based on correlation alone)
    :return: Updated adata object
    """

    # Error check
    adata_ind_paths = glob.glob(os.path.join(individual_dir, '**/adata*h5ad'), recursive = True)
 

    if ~np.all([np.any([re.search(j,os.path.basename(i))!=None for i in adata_ind_paths]) \
               for j in adata.obs.loc[:,sample_key].cat.categories.str.replace('+','_')]):
        raise RuntimeError('Ensure that all individual adata''s are placed in individual_dir.')
    
    rscript_path = str(pathlib.Path().resolve())

    if save_rds_path == None:
        save_rds_path = os.path.join(rscript_path, 'results')

    obs_list = []
    last_mc = 0
    for sample in adata.obs.loc[:,sample_key].cat.categories: 
        sample = re.sub('\+','_',sample)
        adata_ind_path = glob.glob(os.path.join(individual_dir,'%s/adata*h5ad' % sample))[0]
        adata_ind = sc.read_h5ad(adata_ind_path) 
        sample = os.path.basename(os.path.dirname(adata_ind_path))
        save_rds = os.path.join(save_rds_path, 'SBM_results.%s.RDS' % sample)  
        adata_ind = _CalcSBM_per_sample(adata_ind, save_rds = save_rds)
        
        obs_tmp = adata_ind.obs.loc[:,['SBM_cluster']]
        obs_tmp.index = adata_ind.obs.index.astype(int) + last_mc
        obs_list += [obs_tmp]
        last_mc += (adata_ind.obs.index.astype(int)[-1] + 1)
    obs = pd.concat(obs_list, axis=0)

    return adata

def Plot_kNN_aSBM(
        adata: AnnData,
        individual_dir: str,
        RDS_input_dir: str,
        save: Optional[str] = None, 
        sample_key: Optional[str] = 'batch',
        node_attribute: Optional[str] = 'Basal.Correlation',
        node_init: Optional[str] = 'Basal.vs.Luminal.by.Correlation',
        num_rows: Optional[int] = 3,
        width: Optional[float] = 9,
        height: Optional[float] = 9,
):
    """Function for plotting force-directed graphs of metacells, with edges determined by kNN (predetermined by sc.pp.neighbors), nodes colored by Basal vs Luminal correlation, and Basal vs Luminal graph partitions based on aSBM highlighted. These plots can visualize changes in edge connectivity between Basal and Luminal partitions.    
    :param adata: adata object of metacells from the combined time-course
    :param individual_dir: Directory that contains individual adata's for each sample in the time-course
    :param RDS_input_dir: Directory that contains RDS files created from running CalcSBM
    :param save: File name to save figure. Must be PDF
    :param sample_key: Column name for samples/batches stored in adata.obs (str)
    :parma node_attribute: Name of (metacell) node attribute that was used to perform aSBM (default is correlation to Basal vs Luminal cells)
    :parma node_init: Name of (metacell) node attribute that was used to initialize aSBM (default is Basal vs Luminal label based on correlation alone)
    :param num_rows: Number of rows of subpanels in figure
    :param width: Width of figure
    :param height: Height of figure
    """

    # Error check
    if os.path.isdir(RDS_input_dir) != True:
        raise RuntimeError('Input directory for RDS files does not exist. Ensure that CalcSBM is run first.')

    adata_ind_paths = glob.glob(os.path.join(individual_dir,'*/adata*h5ad'))

    if ~np.all([np.any([re.search(j,os.path.basename(i))!=None for i in adata_ind_paths]) \
               for j in adata.obs.loc[:,sample_key].cat.categories.str.replace('+','_')]):
        raise RuntimeError('Ensure that all individual adata''s are placed in individual_dir.')

    rscript_path = str(pathlib.Path().resolve())

    if save == None:
        save = os.path.join(rscript_path,'figures','edge_connectivity.kNN.aSBM.pdf')

    # temporary files to be read by R script file
    obs_input_file = os.path.join(rscript_path, 'tmp.input.csv')
    adata.obs.to_csv(obs_input_file,sep='\t')

    for sample in adata.obs.loc[:,sample_key].cat.categories:
        sample = re.sub('\+','_',sample)
        adata_ind_path = glob.glob(os.path.join(individual_dir,'%s/adata*h5ad' % sample))[0]
        adata_ind = sc.read_h5ad(adata_ind_path)

        # temporary files to be read by R script file
        fdl_input_file = os.path.join(rscript_path, 'tmp.input.%s.fdl' % sample)

        df = pd.DataFrame(adata_ind.obsm['X_draw_graph_fa'],index=adata_ind.obs.index)
        df.index.name = 'Cell'
        df.columns = ['x','y']
        df.to_csv(fdl_input_file, sep = '\t')

    sample_order = ','.join(adata.obs.loc[:,sample_key].cat.categories.str.replace('+','_'))

    subprocess.call("~/anaconda3/envs/r_4.0.3c/bin/Rscript --vanilla %s/_Plot_kNN_aSBM.R %s %s %s %s %s %s %s %d %d %d" % (rscript_path, obs_input_file, RDS_input_dir, sample_key, node_attribute, node_init, sample_order, save, num_rows, width, height), shell=True)

    os.remove(obs_input_file)

    for sample in adata.obs.loc[:,sample_key].cat.categories:
        sample = re.sub('\+','_',sample)

        # temporary files to be read by R script file
        fdl_input_file = os.path.join(rscript_path, 'tmp.input.%s.fdl' % sample)

        os.remove(fdl_input_file)


def PlotDiffusionDistances(
        adata: AnnData,
        num_dcs: Optional[int] = 6, 
        num_trials: Optional[int] = 1000, 
        subsample_size: Optional[int] = 50, 
        diffmap_rep: Optional[str] = 'X_diffmap',
        sample_key: Optional[str] = 'batch',
        save: Optional[str] = None,
        colors: Optional[list] = None,
        width: Optional[float] = 5,
        height: Optional[float] = 5,
):
    """Function for computing diffusion distances between subsampled basal and luminal metacells as a per-sample measure of plasticity in a time-course. 
    :param adata: adata object of metacells from the combined time-course
    :param num_dcs: Number of diffusion components to retain. Default 6 (int)
    :param num_trials: Number of times to randomly subsample Basal and Luminal metacells. Default 1000 (int)
    :param subsample_size: Number of Basal (or Luminal) metacells to randomly subsample for each trial. Default 50 (int)
    :param diffmap_rep: Key name in adata.uns for diffusion map (str)
    :param sample_key: Column name for samples/batches stored in adata.obs (str)
    :param save: File name to save figure
    :param colors: List of colors for boxplot (length corresponding to number of samples)
    :param width: Width of figure
    :param height: Height of figure
    """

    # Error check
    if diffmap_rep not in adata.obsm.keys():
        raise RuntimeError('Diffusion map representation not available.')

    if num_dcs > adata.obsm[diffmap_rep].shape[1]:
        raise RuntimeError('Number of eigenvectors should not exceed the number of diffusion components calculated.')

    if 'SBM_cluster' not in adata.obs.columns:
        raise RuntimeError('Run DiffusionDistance.CalcSBM first.')

    # Create multi-scaled diffusion map
    use_eigs = list(range(1, num_dcs))
    eig_vals = np.ravel(adata.uns['diffmap_evals'][use_eigs])
    ms_data = pd.DataFrame(adata.obsm[diffmap_rep][:,use_eigs] * (eig_vals / (1-eig_vals)))
    ms_data.index = adata.obs_names

    dd = pd.DataFrame(0, index=np.arange(num_trials), columns = adata.obs.loc[:,sample_key].cat.categories)
    cells_b = adata.obs.index[adata.obs.loc[:,'SBM_cluster']==1]
    cells_l = adata.obs.index[adata.obs.loc[:,'SBM_cluster']==2]
    for sample in adata.obs.loc[:,sample_key].cat.categories:
        cells_sample = adata.obs.index[adata.obs.loc[:,sample_key] == sample]

        b_data = ms_data.loc[cells_sample.intersection(cells_b),:]
        l_data = ms_data.loc[cells_sample.intersection(cells_l),:]

        for n in range(num_trials):
            sub_mat = pairwise_distances(b_data.iloc[np.random.randint(0, b_data.shape[0], subsample_size),:], 
                                         l_data.iloc[np.random.randint(0, l_data.shape[0], subsample_size),:], 
                                         metric='euclidean')
            dd.loc[n,sample] = np.log10(np.mean(list(np.ravel(sub_mat))))

    plot_df = dd.melt(var_name = sample_key, value_name = 'diffusion_distance')

    sc.set_figure_params(fontsize=16)
    sns.set_style('ticks')

    fig, ax = plt.subplots(1, 1, figsize = (width,height))

    if colors == None:
        sns.boxplot(data=plot_df, x=sample_key, y ='diffusion_distance', showfliers = False, ax=ax)
    else:
        sns.boxplot(data=plot_df, x=sample_key, y ='diffusion_distance', showfliers = False, ax=ax, palette = colors)

    ax.set_ylabel('Log Mean\nDiffusion Distance', fontsize=25)
    ax.set_xlabel('Sample', fontsize=25)

    ax.set_xticklabels(adata.obs.loc[:,sample_key].cat.categories, rotation = 45, fontsize=25)

    ax.tick_params(axis='both', labelsize=20)

    ax.set_title('Basal-Luminal Plasticity', fontsize=25)

    if save != None:
        plt.savefig(save)
