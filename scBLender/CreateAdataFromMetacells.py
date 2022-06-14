import sys
import os
import pandas as pd
import re
import numpy as np
import glob
from scipy import sparse
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import scanpy as sc
import phenograph
from sklearn.metrics import adjusted_rand_score
from scipy.interpolate import UnivariateSpline
import numpy.matlib

def kneepoint(vec):
    curve =  [1-x for x in vec]
    nPoints = len(curve)
    allCoord = np.vstack((range(nPoints), curve)).T
    np.array([range(nPoints), curve])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * numpy.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint

###INPUT 
adata_fn = sys.argv[1]
metacell_dir = sys.argv[2] # Assumes that metacells have been calculated over a large range of number of metacells at intervals of 1 
out_dir = sys.argv[3]

###PARAMETERS
hvg = 5000
fraction_metacells_for_knn = 0.03
spl_degree = 4

###LOAD DATA
fig_dir = out_dir + 'figures/'
os.makedirs(fig_dir,exist_ok=True)

mc_results = {}
for fn in glob.glob(metacell_dir + 'metacell_runs/metacell.*.results.p'):
    k = int(os.path.basename(fn).split('.')[1])
    mc_results[k] = pickle.load(open(fn,'rb'))

adata = sc.read_h5ad(adata_fn)

'''
Choosing number of MetaCells
'''

ks = sorted(mc_results.keys())

rand_scores = np.zeros((len(ks),len(ks)))
for i,k1 in enumerate(ks):
    for j,k2 in enumerate(ks):
        clusterings1 = mc_results[k1]['assignments'].argmax(axis=1)
        clusterings2 = mc_results[k2]['assignments'].argmax(axis=1)
        rand_scores[i,j] = adjusted_rand_score(clusterings1,clusterings2)

rand_scores = pd.DataFrame(rand_scores, index=ks, columns=ks)

fig,ax = plt.subplots(1,1,figsize=(10,10))
sns.heatmap(rand_scores,
           cmap = 'seismic', xticklabels = ks, yticklabels = ks, ax=ax)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=5)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)

plt.savefig(fig_dir + 'heatmap.metacells.ARI.png', dpi=300, bbox_inches = 'tight')

sns.set_style('ticks')
plot_df = pd.Series([rand_scores.iloc[i,i+1] for i in range(rand_scores.shape[0]-1)],
                    index = rand_scores.columns[1:])

spl = UnivariateSpline(plot_df.index, plot_df, k=spl_degree)
k_kneepoint = plot_df.index[kneepoint(spl(plot_df.index))]

fig,ax = plt.subplots(1,1) 
ax.scatter(plot_df.index, plot_df, color = 'black')
ax.plot(plot_df.index, spl(plot_df.index), color='r',linewidth=3)
    
ax.set_xlabel('Number of MetaCells')
ax.set_ylabel('Adjusted Rand Index')
ax.axvline(k_kneepoint, c ='blue')
ax.set_ylim([0.5,1])
plt.savefig(fig_dir + 'scatter.metacells.ARI.kneepoint.png', dpi=300, bbox_inches='tight')

'''
Metacell analysis
'''

centers = mc_results[k_kneepoint]['centers']
sizes = mc_results[k_kneepoint]['sizes']
coords = mc_results[k_kneepoint]['coords']

g = adata.var_names

# create AnnData
metacell_adata = sc.AnnData(coords)
metacell_adata.var_names = g
metacell_adata.obs["batch"] = adata.obs.batch.iloc[centers].values
metacell_adata.obs["centers"] = centers
metacell_adata.obs["sizes"] = sizes

# store raw counts
metacell_adata.raw = metacell_adata

# normalize and visualize
sc.pp.filter_genes(metacell_adata, min_cells=1)
sc.pp.normalize_total(metacell_adata)
sc.pp.log1p(metacell_adata)

#from decimal import localcontext, Decimal, ROUND_HALF_UP
n_neighbors = int(np.ceil(metacell_adata.shape[0] * fraction_metacells_for_knn))

sc.pp.highly_variable_genes(metacell_adata, n_top_genes = hvg)
sc.tl.pca(metacell_adata, use_highly_variable = True, random_state=2)
sc.pp.neighbors(metacell_adata, n_neighbors=n_neighbors)
sc.tl.diffmap(metacell_adata)
sc.tl.draw_graph(metacell_adata, random_state=0)

pca_merge = pd.DataFrame(metacell_adata.obsm['X_pca'], index = metacell_adata.obs.index)
clusters_merge, _, _ = phenograph.cluster(pca_merge, k = 20)
clusters_merge = pd.Series(clusters_merge, pca_merge.index)
metacell_adata.obs['phenograph'] = clusters_merge.loc[metacell_adata.obs_names].astype(int).astype('category')

sc.tl.paga(metacell_adata, groups = 'phenograph')
sc.pl.paga(metacell_adata, plot=False)
sc.tl.umap(metacell_adata, init_pos='paga', min_dist=0.3)

metacell_adata.write_h5ad(out_dir + 'adata.metacells.%d.h5ad' % (k_kneepoint))
