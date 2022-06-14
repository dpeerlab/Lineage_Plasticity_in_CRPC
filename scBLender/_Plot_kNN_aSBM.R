suppressMessages(library(dplyr))
suppressMessages(library(igraph))
suppressMessages(library(Matrix))
suppressMessages(library(stringr))

args <- commandArgs(trailingOnly = TRUE)
obs_input_file <- args[1]
RDS_input_dir <- args[2]
sample_key <- args[3]
node_attribute <- args[4]
node_init <- args[5]
sample_order <- args[6]
save <- args[7]
num_rows <- as.numeric(args[8])
width <- as.numeric(args[9])
height <- as.numeric(args[10])

obs_df = read.table(obs_input_file, header = T, row.names = 1, sep = '\t')

breaks = c(-.35, seq(-0.2, 0.2, length.out = 11), .35)
cuts = as.numeric(cut(obs_df[,node_attribute],breaks = breaks))

col_bl_corr_all <- colorRampPalette(c('blue','yellow','red'))(max(cuts))[cuts]
names(col_bl_corr_all) = rownames(obs_df)

samples = str_split(sample_order,',')[[1]]

G_all = list()
col_all = list()
SBM_all = list()
l_all = list()
max_mc = 0
for (sample in samples) {
    RDS_file = file.path(RDS_input_dir, sprintf('SBM_results.%s.RDS', sample))
    results = readRDS(RDS_file)

    l = as.matrix(read.table(file.path(dirname(obs_input_file), sprintf('tmp.input.%s.fdl', sample)), sep = '\t', row.names = 1, header = T))

    G_all[[sample]] = results$G
    l_all[[sample]] = l

    col_bl = c('blue','red')[factor(obs_df[, node_init])]

    rbPal <- colorRampPalette(c('blue','red'))
    col_bl_corr <- rbPal(10)[as.numeric(cut(obs_df[,node_attribute],breaks = 10))]

    mc_id = as.numeric(rownames(l_all[[sample]])) + max_mc
    max_mc = max(mc_id) + 1

    col_all[[sample]] <- col_bl_corr_all[as.character(mc_id)]

    SBM	= obs_df[as.character(mc_id), 'SBM_cluster']
    SBM_all[[sample]] = SBM

}

pdf(file = save,width=width,height=height)

num_cols = ceiling(length(samples)/3)

par(mfrow= c(num_rows,num_cols), mar = c(0,0,1,0))
for (sample in names(G_all)) {
  plot(G_all[[sample]], vertex.color = col_all[[sample]], vertex.size = 8, vertex.label = NA, layout = l_all[[sample]], main = sample,
       mark.groups=list(which(SBM_all[[sample]]==1), which(SBM_all[[sample]]==2)), mark.col = rev(rainbow(2, alpha = 0.1)), mark.border= NA)
}

while (!is.null(dev.list()))  dev.off()
