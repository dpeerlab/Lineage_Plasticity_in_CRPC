suppressMessages(library(dplyr))
suppressMessages(library(ggtern))
suppressMessages(library(grid))
suppressMessages(library(gridExtra))

args <- commandArgs(trailingOnly = TRUE)
obs_file <- args[1]
sample_key <- args[2]
initial_sample <- args[3]
save <- args[4]
num_rows <- as.numeric(args[5])
width <- as.numeric(args[6])
height <- as.numeric(args[7])

obs_sc_df = read.table(obs_file, header=T, stringsAsFactors = F, sep = '\t', row.names = 'Cell')

pval_df = obs_sc_df %>% dplyr::select(colnames(obs_sc_df)[grepl('pval_markov_',colnames(obs_sc_df))])
pval_df = pval_df - apply(pval_df, 1, min)
pval_df = pval_df/rowSums(pval_df)

colnames(pval_df) = gsub('pval_markov_','',colnames(pval_df))

ct_mc_pval_preds = colnames(pval_df)[max.col(pval_df)]

#Identify top 3 predominant cell type probabilties
ind1 = obs_sc_df[,sample_key] != initial_sample
top3 = sort(names(sort(table(ct_mc_pval_preds[ind1]), decreasing=T))[1:3])

ind2 = ct_mc_pval_preds %in% top3
pval_df = pval_df[ind1 | ind2, top3]/rowSums(pval_df[ind1 | ind2,c('Basal_Org2','Basal_Org3','Luminal_Org2')])


pval_df = pval_df + 0.02
sigma = 0.02
pval_df = pval_df + cbind(rnorm(n = nrow(pval_df), mean = 0, sd = sigma),
                          rnorm(n = nrow(pval_df), mean = 0, sd = sigma),
                          rnorm(n = nrow(pval_df), mean = 0, sd = sigma))

pval_df = pval_df - apply(pval_df, 1, function(x) min(c(x,0)) )
pval_df = pval_df/rowSums(pval_df)


col = rgb(pval_df[,2], pval_df[,1], pval_df[,3])

pval_gg = as.data.frame(cbind(pval_df, col))


tmp = obs_sc_df[rownames(pval_df),sample_key]
pval_gg = as.data.frame(cbind(pval_df, Sample = tmp))

samples = sort(unique(pval_gg$Sample))
pval_gg$Sample = factor(pval_gg$Sample, levels=c(initial_sample, samples[samples != initial_sample]))

p = list()
for (sample in levels(pval_gg$Sample)) {
  plot_gg = pval_gg[pval_gg$Sample==sample,]
  plot_gg$color = rgb(plot_gg[,top3[2]], plot_gg[,top3[1]], plot_gg[,top3[3]])
  p[[sample]] <- ggtern(data = plot_gg,
                 aes_string(y = top3[2], x = top3[1], z = top3[3], col='color')) +
    geom_point(aes_string(col = 'color'), size=0.1) +
    labs(y=top3[2], x=top3[1], z=top3[3]) +
    theme_bw() +
    scale_color_identity() +
    ggtitle(sample) +
    guides(fill = guide_legend(override.aes = list(size=7))) +
    theme(tern.panel.mask.show = FALSE,
          axis.title = element_text(size=7), plot.title = element_text(hjust=0.5),
          tern.axis.title.L = element_text(hjust = 0,vjust=1.8),
          tern.axis.title.R = element_text(hjust = 1,vjust=1.8),
          text = element_text(size=10))
}


p$nrow = num_rows
p$top = textGrob('Cell Type Probabilities',gp=gpar(fontsize=20,font=3))

cairo_pdf(file = save, width=width, height=height)
do.call(ggtern::grid.arrange, p)
dev.off()
