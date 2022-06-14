import os
import pandas as pd
import sys
import scanpy as sc
from importlib import reload
import build_graph # script for building shared NN graph
import metacells # script for finding metacells

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'metacells-master')

adata_fn = sys.argv[1]
out_dir = sys.argv[2]
N_METACELLS = int(sys.argv[3])

#################
'''
LOAD DATA
'''

adata = sc.read_h5ad(adata_fn)

counts = adata.raw.X
bc = adata.raw.obs_names
g = adata.raw.var_names
counts = pd.DataFrame(counts.tocsr().toarray(), index=bc, columns = g)

#################
'''
METACELLS
'''

# input to graph construction is PCA/SVD
kernel_model = build_graph.MetacellGraph(adata.obsm["X_pca"], verbose=True)

# K is a sparse matrix representing input to metacell alg
K = kernel_model.rbf()

reload(metacells)

# set number of metacells

graph_model = metacells.Metacells(n_metacells=N_METACELLS)

# use K from above
graph_model.fit(K);

cx = graph_model.get_centers()

from collections import Counter
sizes = graph_model.get_sizes()#Counter(np.argmax(graph_model.A_, axis=0))

coords = graph_model.get_coordinates(counts)

assgts = graph_model.get_assignments()

mc_results = {}
mc_results['centers'] = cx
mc_results['sizes'] = sizes
mc_results['assignments'] = assgts
mc_results['coords'] = coords

import pickle

with gzip.open(out_dir + 'metacell.%d.results.p.gz' % N_METACELLS, 'wb') as f:
    pickle.dump(mc_results, f)







