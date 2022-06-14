from typing import Optional
import scanpy as sc
from anndata import AnnData
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def SelectBasalLuminalDEGs(
        num_degs: int
) -> list:
    """Function that returns the top Basal and Lumianl DEGs gained following mutation. 
    :param num_degs: Number of top Basal (and Luminal) DEGs to obtain to calculate Basal vs Luminal Z-scores per cell
    :return: A list of length 2, containing a list of the top Basal and Luminal genes gained following mutation
    """

    ref_dir = os.path.join(str(pathlib.Path().resolve()), 'genelists') 

    deg_file = os.path.join(ref_dir,  'deg.mast.MUT_WT.CD49f_only.filtered.csv')
    mut_wt_b_df = pd.read_csv(deg_file, sep =',', index_col = 'primerid')

    ind1 = mut_wt_b_df.loc[:,'Bonferroni.Padj'] < 0.05
    ind2 = mut_wt_b_df.coef > np.log2(1.5)
    mut_wt_b_up = mut_wt_b_df.index[ind1.values & ind2.values]
    ind2 = mut_wt_b_df.coef < -np.log2(1.5)
    mut_wt_b_dn = mut_wt_b_df.index[ind1.values & ind2.values]
    mut_wt_b = mut_wt_b_up.union(mut_wt_b_dn)

    deg_file = os.path.join(ref_dir,  'deg.mast.MUT_WT.CD24_only.filtered.csv')
    mut_wt_l_df = pd.read_csv(deg_file, sep =',', index_col = 'primerid')

    ind1 = mut_wt_l_df.loc[:,'Bonferroni.Padj'] < 0.05
    ind2 = mut_wt_l_df.coef > np.log2(1.5)
    mut_wt_l_up = mut_wt_l_df.index[ind1.values & ind2.values]
    ind2 = mut_wt_l_df.coef < -np.log2(1.5)
    mut_wt_l_dn = mut_wt_l_df.index[ind1.values & ind2.values]
    mut_wt_l = mut_wt_l_up.union(mut_wt_l_dn)

    filter_genes = []
    for term in ['apoptosis','hypoxia','cell_cycle']:
        gmt = os.path.join(ref_dir, term + '.gmt')
        with open(gmt,'r') as f:
            lines = f.readlines()
            for line in lines:
                filter_genes += line.strip().split('\t')[2:]

    filter_paths = ['HALLMARK_INTERFERON_ALPHA_RESPONSE',
                    'HALLMARK_INTERFERON_GAMMA_RESPONSE',
                    'HALLMARK_IL6_JAK_STAT3_SIGNALING',
                    'FGFR_signature.Acevedo_etal', 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION']

    gmt_fn = os.path.join(ref_dir, 'curated.010122.gmt')

    with open(gmt_fn,'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.split('\t')[0] in filter_paths:
                filter_genes += line.strip().split('\t')[2:]

    mut_wt_l_up2 = mut_wt_l_up[~mut_wt_l_up.isin(filter_genes)]
    mut_wt_b_up2 = mut_wt_b_up[~mut_wt_b_up.isin(filter_genes)]


    deg_file = os.path.join(ref_dir,  'Basal_vs_Luminal.CD49f.csv')
    df = pd.read_csv(deg_file, index_col=0)
    df.loc[:,'Bonferroni.Padj'] = [min(1,i) for i in df.loc[:,'Pr(>Chisq)']*df.shape[0]]
    ind2 = df.loc[:,'Bonferroni.Padj'] < 0.05

    ind1 = df.coef > np.log2(1.5)
    b_all =  df.loc[ind1.values & ind2.values ,:].sort_values('coef',ascending=False)
    b_all = b_all.loc[b_all.primerid.isin(mut_wt_l_up2),:]

    ind1 = df.coef < -np.log2(1.5)
    l_all =  df.loc[ind1.values & ind2.values ,:].sort_values('coef',ascending=True)
    l_all = l_all.loc[l_all.primerid.isin(mut_wt_b_up2),:]

    b = b_all.sort_values('coef',ascending=False).primerid.values[:num_degs]
    l = l_all.sort_values('coef',ascending=True).primerid.values[:num_degs]
    return b,l 

def PlotDensity_BasalLuminalSpace(
        adata: AnnData,
        num_degs: Optional[int] = 150,
        layer: Optional[str] = 'imputed',
        sample_key: Optional[str] = 'batch',
        initial_sample: Optional[str] = 'Wk0',
        basal_vs_luminal: Optional[str] = 'Basal vs Luminal by Correlation',
        num_rows: Optional[int] = 3,
        save: Optional[str] = None,
        width: Optional[float] = 9,
        height: Optional[float] = 9,
):
    """Function for plotting density of single cells based on Basal and Luminal scores calculated in the feature space of Basal and Luminal genes, stratified by sample timepoint.
    :param adata: adata object of single cells from the combined time-course
    :param num_degs: Number of top Basal (and Luminal) DEGs to obtain to calculate Basal vs Luminal Z-scores per cell
    :param layer: Layer of adata to obtain expression matrix. Default is 'imputed'. If the value does not match any of the keys in adata.layers, then adata.X is used to populate the expression matrix.
    :param sample_key: Column name for samples/batches stored in adata.obs (str)
    :param initial_sample: Name of the wild-type sample (str)
    :param basal_vs_luminal: Name of labels for basal vs luminal cell (default is based on correlation).
    :param save: File name to save figure    
    :param width: Width of figure
    :param height: Height of figure
    """

    #Rename CD24 as Luminal, CD49f as Basal
    adata.obs.loc[:,basal_vs_luminal] = adata.obs.loc[:,basal_vs_luminal].str.replace('CD24','Luminal').str.replace('CD49f','Basal')

    b,l = SelectBasalLuminalDEGs(num_degs)

    if layer in adata.layers.keys():
        mat = pd.DataFrame(adata.layers[layer], index=adata.obs.index, columns = adata.var_names)
    else:
        mat = pd.DataFrame(adata.X, index=adata.obs.index, columns = adata.var_names)

    z1 = mat.loc[:, l].apply(zscore).mean(axis=1)
    z2 = mat.loc[:, b].apply(zscore).mean(axis=1)

    adata.obs.loc[:,'MUT_plus_wt_bl'] = adata.obs.loc[:,sample_key]
    adata.obs.loc[:,'MUT_plus_wt_bl'] = adata.obs.loc[:,'MUT_plus_wt_bl'].astype(str)
    adata.obs.loc[adata.obs.loc[:,'MUT_plus_wt_bl'] != initial_sample,'MUT_plus_wt_bl'] = 'Mutant'
    adata.obs.loc[adata.obs.MUT_plus_wt_bl=='Wk0','MUT_plus_wt_bl'] = initial_sample + ': ' + adata.obs.loc[adata.obs.MUT_plus_wt_bl == initial_sample, basal_vs_luminal].astype(str)
    adata.obs.MUT_plus_wt_bl = adata.obs.MUT_plus_wt_bl.astype('category')

    plot_df = pd.DataFrame({'Luminal Z-score': z1, 'Basal Z-score': z2, 'Condition': adata.obs.MUT_plus_wt_bl})

    sc.set_figure_params(fontsize=18)
    sns.set_style('ticks')

    samples = adata.obs.loc[:,sample_key].cat.categories

    num_cols = len(samples)/num_rows
    fig, axes = plt.subplots(num_rows,num_cols, figsize = (10,10))

    for ax, sample in zip(np.ravel(axes), samples):

        sns.kdeplot(data = plot_df.loc[plot_df.Condition == initial_sample + ': Luminal',:], 
                    x='Luminal Z-score', y='Basal Z-score', label='WT Luminal', ax=ax, color='lightblue')
        sns.kdeplot(data = plot_df.loc[plot_df.Condition == initial_sample + ': Basal',:], 
                    x='Luminal Z-score', y='Basal Z-score', label='WT Basal', ax=ax, color='pink')
        if sample!=initial_sample:
            ind1 = adata.obs.loc[:,sample_key]==sample
            ind2 = adata.obs.loc[:,basal_vs_luminal] == 'Basal'
        
            sns.kdeplot(data = plot_df.loc[ind1.values & ~ind2.values,:], 
                        x='Luminal Z-score', y='Basal Z-score', label='MUT Luminal', ax=ax, color = 'blue')
            sns.kdeplot(data = plot_df.loc[ind1.values & ind2.values,:], 
                        x='Luminal Z-score', y='Basal Z-score', label='MUT Basal', ax=ax, color = 'red')
        ax.axvline(x=0, linestyle='--', color='lightgray')
        ax.axhline(y=0, linestyle='--', color='lightgray')
        ax.set_title(sample)
    ax.legend(bbox_to_anchor=(1,1), loc = 'upper left')
    plt.tight_layout()

    if save != None:
        plt.savefig(save)

def PlotLSAP(
        adata: AnnData,
        num_degs: int,
        layer: Optional[str] = 'imputed',
        num_trials: Optional[int] = 100,
        subsample_size: Optional[int] = 100,
        sample_key: Optional[str] = 'batch',
        initial_sample: Optional[str] = 'Wk0',
        basal_vs_luminal: Optional[str] = 'Basal vs Luminal by Correlation',
        save: Optional[str] = None,
        colors: Optional[list] = None,
        width: Optional[float] = 5,
        height: Optional[float] = 5,
):
    """Function for computing and plotting the optimized cost of the Linear Sum Assignment Problem (LSAP) mapping subsampled basal to luminal single cells. This LSAP cost serves as a per-sample measure of plasticity in a time-course.
    :param adata: adata object of single cells from the combined time-course
    :param layer: Layer of adata to obtain expression matrix. Default is 'imputed'. If the value does not match any of the keys in adata.layers, then adata.X is used to populate the expression matrix. 
    :param num_trials: Number of times to randomly subsample Basal and Luminal cells. Default 100 (int)
    :param subsample_size: Number of Basal (or Luminal) cells to randomly subsample for each trial. Default 100 (int)
    :param sample_key: Column name for samples/batches stored in adata.obs (str)
    :param initial_sample: Name of the wild-type sample (str)
    :param basal_vs_luminal: Name of labels for basal vs luminal cell (default is based on correlation). 
    :param save: File name to save figure
    :param colors: List of colors for boxplot (length corresponding to number of samples)
    :param width: Width of figure
    :param height: Height of figure
    """

    b,l = SelectBasalLuminalDEGs(num_degs)

    #Rename CD24 as Luminal, CD49f as Basal
    adata.obs.loc[:,basal_vs_luminal] = adata.obs.loc[:,basal_vs_luminal].str.replace('CD24','Luminal').str.replace('CD49f','Basal')

    if layer in adata.layers.keys():
        mat = pd.DataFrame(adata.layers[layer], index=adata.obs.index, columns = adata.var_names)
    else:
        mat = pd.DataFrame(adata.X, index=adata.obs.index, columns = adata.var_names)

    d_lsap = pd.DataFrame(0, index=np.arange(num_trials), columns = sorted(set(adata.obs.loc[:,sample_key])))
    for sample in adata.obs.loc[:,sample_key].cat.categories:
        ind1 = adata.obs.loc[:,sample_key]==sample
        sub_mat = mat.apply(zscore).loc[ind1,list(b)+list(l)]
        
        ind2 = adata.obs.loc[ind1, basal_vs_luminal] == 'Basal'
        for n in range(num_trials):
            b_data = sub_mat.loc[ind2,:]
            l_data = sub_mat.loc[~ind2,:]
            Y1 = b_data.iloc[np.random.randint(0, b_data.shape[0], subsample_size),:]
            Y2 = l_data.iloc[np.random.randint(0, l_data.shape[0], subsample_size),:]
            d = cdist(Y1, Y2)
            assignment = linear_sum_assignment(d)
            d_lsap.loc[n,sample] = d[assignment].mean()

    d_lsap = d_lsap.melt(var_name = 'Sample', value_name = 'Distance after LSAP')

    d_lsap.Sample = d_lsap.Sample.astype('category').cat.reorder_categories(adata.obs.loc[:,sample_key].cat.categories)

    sc.set_figure_params(fontsize=16)
    sns.set_style('ticks')
    fig, ax= plt.subplots(1,1,figsize=(width,height))

    if colors == None:
        sns.boxplot(data=d_lsap, x='Sample', y='Distance after LSAP', showfliers = False, ax=ax)
    else:
        sns.boxplot(data=d_lsap, x='Sample', y='Distance after LSAP', showfliers = False, ax=ax, palette = colors)

    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
    ax.set_title('1/Plasticity')
    ax.set_ylabel('Mean Distance after LSAP')

    if save != None:
        plt.savefig(save)
