suppressMessages(library(dplyr))
suppressMessages(library(igraph))
suppressMessages(library(Matrix))
suppressMessages(library(mvtnorm))
source('FitAttribute.k_init.R')

args <- commandArgs(trailingOnly = TRUE)
obs_file <- args[1]
knn_file <- args[2]
node_attribute <- args[3]
node_init <- args[4]
sbm_file <- args[5]
save_rds <- args[6]

ind_dir = '/home/chanj3/data/Prostate.LP.publication.010122/out.metacells.individual.010122/'

G_all = list()
SBM_all = list()
col_all = list()
l_all = list()

mat = readMM(knn_file)
mat = (mat+t(mat))/2


obs_df = read.table(obs_file, header = T, row.names = 1, sep = '\t')

BL = obs_df %>% dplyr::select(one_of(c(node_attribute))) %>% as.matrix

G = graph_from_adjacency_matrix(mat, mode = 'undirected', weighted =T)
clusters = cluster_spinglass(G, spins=2)

mat2 = mat !=0

Out =FitAttribute(mat2, BL, 0, 2, as.numeric(factor(obs_df[, node_init])))

results = list()
results$SBM = Out
results$G = G
saveRDS(results, save_rds)

SBM_cluster = Out$Comm
names(SBM_cluster) = rownames(obs_df)

write.table(data.frame(SBM_cluster), file = sbm_file, sep = ',', quote = F, row.names = T)
