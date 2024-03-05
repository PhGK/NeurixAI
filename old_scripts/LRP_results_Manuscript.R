library(magrittr)
library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)
library(Hmisc)
library(cowplot)
library(pROC)
library(stringr)
library(RColorBrewer)
library(netresponse)
library(igraph)
library(ComplexHeatmap)
#genes that are relevant across drugs? sensitivity genes? essentiality?



setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

read_dat <- function() {
  files <- list.files(paste0('../results/LRP/'))
  print(files)
  d <- rbindlist(lapply(files, function(f) fread(paste0('../results/LRP/', f))[,-1] %>% unique()))
  #d <- rbindlist(lapply(files, function(f) fread(paste0('../results/LRP/', f))[,-1] ))
  d
}

dat <- read_dat()

get_slope <- function(y,x) {
  model <- lm(y~x)$coef[2]
  model
}

across_cell_lines <- dat %>%group_by(molecular_names, DRUG) %>%
  group_by(molecular_names) %>%
  dplyr::mutate(maxexpr = max(abs(expression))) %>%
  filter(maxexpr<10) %>% # filter genes with outliers on test set
  group_by(molecular_names, DRUG) %>%
  dplyr::summarize(meanabsLRP = mean(abs(LRP)), slope = cor(LRP, expression), ncells = n(), maxexpr = max(expression)) %>%
  #dplyr::summarize(meanabsLRP = mean(abs(LRP)), ncells = n()) %>%
  group_by(DRUG) %>%
  mutate(rankl = rank((meanabsLRP))) %>%
  mutate(pos = rankl/max(rankl)) %>%
  mutate(percent_rank = pos * 100)

dat <- dat %>% filter(molecular_names %in% (across_cell_lines$molecular_names %>% unique()))

genes <- data.frame(gene = dat$molecular_names %>% unique())

sel_genes <-  c('ERBB2', 'EGFR', 'MDM2', 'ABCB1', 'MAP2K1','MAP2K6', 'MAP3K21', 'TUSC3')

rel_gene_names <- across_cell_lines %>%
  filter(molecular_names %in% sel_genes)




#############################################
#find Cobimetinib import features
cob <- across_cell_lines %>% filter(grepl('MAP',molecular_names), DRUG == 'COBIMETINIB')

churc1 <- across_cell_lines %>% filter(grepl('CHURC', molecular_names), grepl('VIN', DRUG))
#############################################

#moa data
MOA <- fread('../data/secondary-screen-dose-response-curve-parameters.csv') %>%
  select(name, moa, target) %>%
  unique() %>%
  mutate(DRUG = toupper(name), unique_moa = str_split_fixed(moa, ',', 2)[,1]==moa) %>%
  mutate(moa = ifelse(DRUG=='DASATINIB', 'Bcr-Abl and Src kinase inhibitor', moa))


relevantometer_data <- across_cell_lines %>% filter(molecular_names %in% sel_genes) %>%
  left_join(MOA)

order_by_moa <- relevantometer_data %>%
  select(moa, DRUG) %>%
  unique() %>%
  dplyr::arrange(moa)

relevantometer_data <- relevantometer_data %>%
    mutate(DRUG = factor(DRUG, levels = order_by_moa$DRUG))
across_cell_lines$molecular_names %>% unique() %>% length()

high_relevantometer_data <- relevantometer_data %>% filter(pos > 0.9)
low_relevantometer_data <- relevantometer_data %>% filter(pos <= 0.9)

#################for manuscript##################################################
read_percent_rank <- relevantometer_data %>% filter(molecular_names == 'MAP3K21')
#################################################################################

line_data <- relevantometer_data %>% cross_join(data.frame(x=c(0,1)))

relevantometer <- ggplot(relevantometer_data, aes(y = DRUG, x = pos, label = molecular_names)) +
  geom_line(data = line_data, aes(y = DRUG, x= x, color=moa), linewidth=1) +
  #geom_point(aes(color=molecular_names), size=3)+
  geom_point(data = high_relevantometer_data, size=3, alpha = 1.0) +
  geom_point(data = low_relevantometer_data, size=3) +
  geom_text_repel(data = high_relevantometer_data, aes(y = DRUG, x = pos, label = molecular_names),direction = 'y', color = 'black') +
  geom_text_repel(data = low_relevantometer_data, aes(y = DRUG, x = pos, label = molecular_names),direction = 'y', color = 'black') +
  theme_minimal() +
  theme(#panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y = element_text(size=15),
        axis.text.x = element_text(size=15),
        legend.title = element_blank(),
        legend.text = element_text(size=15),
        legend.position = 'bottom') +
  #ylab('Low importance rank     \U2194     High importance rank')  +
  xlab('xAI-assigned importance rank')+
  scale_y_discrete(limits=rev) +
  scale_x_continuous(labels=scales::percent, limits=c(0.0,1.0)) +
  coord_cartesian(xlim =c(0.0,1.0))


relevantometer


get_nth <- function(x, n=NULL) {
  x <- x[order(x, decreasing=T)]
  return(x[n])
}

most_important <- across_cell_lines %>%
  group_by(DRUG) %>%
  dplyr::mutate(cutoff = get_nth(meanabsLRP,30), scaled_LRP = meanabsLRP/max(meanabsLRP)) %>%
  filter(meanabsLRP>=cutoff) %>%
  mutate(rel = molecular_names %in% sel_genes) %>%
  mutate(DRUG = factor(DRUG, levels = order_by_moa$DRUG))

most_important10 <- across_cell_lines %>%
  group_by(DRUG) %>%
  dplyr::mutate(cutoff = get_nth(meanabsLRP,10), scaled_LRP = meanabsLRP/max(meanabsLRP)) %>%
  filter(meanabsLRP>=cutoff) %>%
  mutate(rel = molecular_names %in% sel_genes) %>%
  mutate(DRUG = factor(DRUG, levels = order_by_moa$DRUG))


most_important_only_red <- most_important %>%
  filter(molecular_names %in% sel_genes)
most_important_without_red <- most_important %>%
  filter(!(molecular_names %in% sel_genes))

most_important_genes_plot_rank <- ggplot(most_important, aes(x = DRUG, y = scaled_LRP, label = molecular_names)) +
  #geom_line(data = line_data, aes(x = DRUG, y= x, color=moa), linewidth=1) +
  geom_line(data = line_data, aes(x = DRUG, y= x, color=moa), linewidth=2, arrow = arrow(length=unit(1,"cm"), type = 'closed')) +

  geom_text_repel(data = most_important_without_red, show.legend=F, size=6, color='black', max.overlaps=1000) +
  geom_label_repel(data = most_important_only_red, show.legend=F, size=6, color='red', max.overlaps = 1000) +
  geom_point(data=most_important_without_red, aes(x = DRUG, y = scaled_LRP),color='black', size=4.0) +
  geom_point(data=most_important_only_red, aes(x = DRUG, y = scaled_LRP),color='red') +
  theme_minimal() +
  scale_y_continuous(trans='log10') +
  theme(axis.text = element_text(size=20),
        axis.title.y = element_blank(),
        axis.title.x = element_text(size=30),
        legend.title = element_blank(),
        legend.text = element_text(size=20),
        legend.position = 'bottom') +
  ylab('xAI-assigned Importance') +
  coord_flip() +
  scale_x_discrete(limits=rev)


most_important_genes_plot_rank
png('./figures/important_genes.png', width=3000, height=1500, res=200)
relevantometer
dev.off()

png('./figures/important_genes_scaled.png', width=3200, height=4000, res=150)
most_important_genes_plot_rank
dev.off()


##########################
#groups_of_important genes across drugs
#########################

group_importance <- most_important10 %>%
  left_join(MOA) %>%
  #filter((moa == 'EGFR inhibitor') | (DRUG == 'IBRUTINIB')) %>%
  filter(moa == 'MEK inhibitor') %>%
  group_by(molecular_names) %>%
  dplyr::summarize(N=n())

#############################
##################################################################################
###############
#save important genes for reactome
##############
reactome_genes <- most_important$molecular_names %>% unique()
write.csv(reactome_genes,'figures/reactome_genes.csv', row.names=F)
write.csv(most_important10$molecular_names %>% unique(),'figures/reactome_genes10.csv', row.names=F)


######
#make networks
#####

all_network_genes <- read.csv('../data/string_interactions_short.tsv',sep='\t')[,c(1,2)]
colnames(all_network_genes) <- c('V1', 'V3')
vertices_not_in_graph <- most_important10$molecular_names[!(most_important10$molecular_names %in% all_network_genes$V1) &
                                                          !(most_important10$molecular_names %in% all_network_genes$V3)] %>% unique()


most_important10 <- most_important10 %>% dplyr::select(molecular_names, meanabsLRP, DRUG, slope) %>%
  group_by(DRUG) %>%
  mutate(node_size = meanabsLRP/max(meanabsLRP), node_color = ifelse(slope<0, 'blue', 'red'))

g <- graph_from_data_frame(all_network_genes, directed=F)
g <- g + vertices(vertices_not_in_graph)
V(g)$name
V(g)$vertexsize <- 5
V(g)$vertexcolor = 'gray60'
E(g)$weight <- 0.01

#l <- layout_with_lgl(g)
set.seed(1)
l <- layout_nicely(g)
#l <- layout_with_fr(g)
#l <- layout_with_lgl(g)
png('./figures/networks_reactome.png', width = 3000, height=3500, res=150)
par(mfrow=c(4,3), mar=c(0,0,2,0))
for (current_drug in order_by_moa$DRUG){
  current_g <- g
  current_data <- most_important10 %>% filter(DRUG == current_drug)
  current_genes <- current_data$molecular_names
  print(length(current_genes))
  node.size<-setNames(current_data$node_size, current_data$molecular_names)
  node.color<-setNames(current_data$node_color, current_data$molecular_names)

  for (k in names(node.size)) {
    if (k %in% V(current_g)$name) {
      print(V(current_g)[k]$name)
    V(current_g)[k]$vertexsize <- node.size[k]**2*20
    V(current_g)[k]$vertexcolor <- node.color[k]}
  }
  print(V(current_g)$vertexsize)
  #current_network_genes <- all_network_genes %>%
  #mutate(important = ifelse((V1 %in% current_genes)| (V3 %in% current_genes), 1,0))
  V(current_g)$color = ifelse(V(current_g)$name %in% current_genes, 'red', 'gray60')
  V(current_g)$vertexcolor[!((degree(current_g)>0) | (V(current_g)$name %in% current_genes))] <- NA

  plot(current_g, vertex.size=V(current_g)$vertexsize, #ifelse(V(current_g)$name %in% current_genes,as.matrix(10), 5),
       layout=l, vertex.label.cex=2.0, vertex.label.color = 'black', edge.width=0.4, edge.color = 'black',
     vertex.color = adjustcolor(V(current_g)$vertexcolor,0.4), vertex.frame.color = NA,
     vertex.label = ifelse(V(current_g)$name %in% current_genes, V(g)$name, NA))#,
     #vertex.label = V(current_g)$name,
     #main = current_drug, main.size=15)
  title(current_drug, cex.main = 2.5)
}
dev.off()

######
#make networks individually
#####
#all_network_genes <- read.csv('../data/10genes.sif', sep='\t', header=F)[,c(1,3)] #%>%
#filter((V1 %in% most_important10$molecular_names) | (V3 %in% most_important10$molecular_names))

most_important10 <- most_important10 %>% dplyr::select(molecular_names, meanabsLRP, DRUG, slope) %>%
  group_by(DRUG) %>%
  mutate(node_size = meanabsLRP/max(meanabsLRP), node_color = ifelse(slope<0, 'blue', 'red'))


#png('./figures/networks_reactome.png', width = 3000, height=3500, res=150)
par(mfrow=c(4,4), mar=c(0,0,1,0))
for (current_drug in order_by_moa$DRUG){

  current_data <- most_important10 %>% filter(DRUG == current_drug)
  current_genes <- current_data$molecular_names

  temp <- all_network_genes %>%
    filter((V1 %in% current_genes) | (V3 %in% current_genes))
  current_genes_plus_linker <- c(temp$V1, temp$V3) %>% unique()

  current_network <- all_network_genes %>%
    filter((V1 %in% current_genes_plus_linker) | (V3 %in% current_genes_plus_linker))

  vertices_not_in_graph <- current_data$molecular_names[!(current_data$molecular_names %in% current_network$V1) &
                                                              !(current_data$molecular_names %in% current_network$V3)] %>% unique()

  g <- graph_from_data_frame(current_network, directed=F)
  g <- g + vertices(vertices_not_in_graph)

  V(g)$vertexsize <- 5
  V(g)$vertexcolor = 'gray60'
  E(g)$weight <- 0.1

  #l <- layout_with_lgl(g)
  l <- layout_nicely(g)

  print(length(current_genes))
  node.size<-setNames(current_data$node_size, current_data$molecular_names)
  node.color<-setNames(current_data$node_color, current_data$molecular_names)

  for (k in names(node.size)) {
    if (k %in% V(g)$name) {
      print(V(g)[k]$name)
      V(g)[k]$vertexsize <- node.size[k]**2*20
      V(g)[k]$vertexcolor <- node.color[k]}
  }
  print(V(g)$vertexsize)
  #V(current_g)$color = ifelse(V(current_g)$name %in% current_genes, 'red', 'gray60')
  #V(current_g)$color[!((degree(current_g)>0) | (V(current_g)$name %in% current_genes))] <- NA

  plot(g, vertex.size=V(g)$vertexsize, #ifelse(V(current_g)$name %in% current_genes,as.matrix(10), 5),
       layout=l, vertex.label.cex=1.0, vertex.label.color = 'black', edge.width=0.4, edge.color = 'black',
       vertex.color = adjustcolor(V(g)$vertexcolor,0.6), vertex.frame.color = NA,
       vertex.label = ifelse(V(g)$name %in% current_genes, V(g)$name, NA),
       #vertex.label = V(current_g)$name,
       main = current_drug)
}
dev.off()






#find similar genes within moa
###########
most_important_plus_moa <- most_important10 %>% left_join(MOA)
ggplot(most_important_plus_moa, aes(y= 0, x = scaled_LRP, color = moa, label=DRUG)) + #geom_boxplot()
  geom_point() +
  geom_text_repel() +
  theme_classic() +
  facet_wrap(~molecular_names)

#find different genes within moa
difference_plus_moa <- across_cell_lines %>%
  inner_join(MOA) %>%
  mutate(moa = ifelse((DRUG=='VINCRISTINE')| (DRUG == 'VINBLASTINE'), 'CHEMO', moa)) %>%
  group_by(moa, molecular_names) %>%
  dplyr::summarize(diffv = max(rankl)- min(rankl), maxv = max(pos), ncells = mean(ncells)) %>%
  arrange(desc(diffv)) %>%
  filter(maxv > 0.9)
moas <- difference_plus_moa$moa %>% unique()

number <- 2

moas[number]

diff_data <- across_cell_lines %>%
  left_join(MOA) %>%
  mutate(moa = ifelse((DRUG=='VINCRISTINE')| (DRUG == 'VINBLASTINE'), 'CHEMO', moa)) %>%
  filter(molecular_names %in% (difference_plus_moa %>% filter(moa ==moas[number]) %>% .$molecular_names %>% .[1:10])) %>%
  filter(moa %in% moas[number]) %>%
  mutate(DRUGpos = paste0(DRUG, ' (', round(pos*100), '%)'))


ggplot(diff_data, aes(y = molecular_names, x = pos, label = DRUGpos)) +
  geom_text()

png(paste0('./figures/DiffScores', moas[number], '.png'), width=2000, height=1000, res=150)
ggplot(diff_data, aes(y = molecular_names, x = percent_rank/100, label = DRUG)) +
  geom_point(size=0.5)+
  geom_text_repel(direction = 'y') +
  xlab('Importance percentile rank') +
  theme(axis.title.y = element_blank(), axis.text = element_text(size=12),
        axis.title.x = element_text(size=14)) +
  scale_x_continuous(labels = scales::percent)
dev.off()

#########
# see relationship between expression and LRP
##########
#only selected genes
rel_gene_names_here <- rel_gene_names %>% filter(molecular_names!= 'MAP2K1')
selected_genes_dynamics <- dat %>% filter(molecular_names %in% rel_gene_names_here$molecular_names)
last_name <- selected_genes_dynamics %>% group_by(DRUG, molecular_names) %>%
  dplyr::mutate(smooth = loess(LRP~expression)$fitted) %>%
  filter(expression == max(expression))

most_important_genes_dynamics <- dat %>% filter(molecular_names %in% most_important$molecular_names)

resistance <- data.frame('resistance' = c('higher resistance', 'higher_sensitivity'), y = c(1,-1))
arrows <- selected_genes_dynamics %>%
  group_by(molecular_names, DRUG) %>%
  dplyr::summarize(x_coord = min(expression)-1, DRUG=first(DRUG)) %>%
  dplyr::select(x_coord, molecular_names, DRUG) %>%
  unique() %>%
  cross_join(resistance)

dynamics_selected_genes <- ggplot(selected_genes_dynamics, aes(x = expression, y = LRP, color=DRUG)) +
  geom_point(size=0.1) +
  geom_smooth(se=F, linewidth=0.4) +
  geom_text_repel(data = last_name, aes(x = expression, y =smooth, label = DRUG), size=4, color = 'black', hjust= -0.5, direction = 'y') +
  facet_wrap(~molecular_names, scales='free', ncol=2, dir='v') +
  theme_classic() +
  guides(color = guide_legend(override.aes = list(size = 5))) +
    xlab('Gene expression') +
  ylab('LRP contribution') +
  xlim(c(-2,6)) +
  theme(strip.text = element_text(size=15),
        axis.text = element_text(size=12),
        axis.title = element_text(size=20),
        legend.text = element_text(size=13),
        legend.title = element_blank(),
        legend.position = 'bottom')
dynamics_selected_genes


png('./figures/dynamics_of_LRPcontribution.png', width=4000, height=4500, res=300)
dynamics_selected_genes
dev.off()


arrow <- ggplot() + geom_line(aes(x = c(0,0), y = c(0.1,1)), linewidth=2, arrow=arrow(length=unit(0.3,"cm"), type = 'closed'), color = 'red') +
  geom_line(aes(x = c(0,0), y = c(-0.1,-1)), linewidth=2, arrow=arrow(length=unit(0.3,"cm"), type = 'closed'), color = 'darkgreen') +
  geom_text(aes(x = 0, y = 1.2, label = 'Resistance'),size=6)+
  geom_text(aes(x = 0, y = -1.2, label = 'Sensitivity'), size=6) +

  theme_void()
arrow_plot <- plot_grid(NULL, arrow, NULL, ncol=1, rel_heights = c(0.8,1,0.95) )

png('./figures/dynamics_of_LRPcontribution_with_arrow.png', width=4000, height=4500, res=300)
plot_grid(dynamics_selected_genes, arrow_plot, rel_widths = c(100,10))
dev.off()

##########
#plot heatmaps
##########
#most_important_gene_names <- most_important$molecular_names %>% unique()
#####################
#analyze MEK inhibitors
cell_line_names <- fread('../data/secondary-screen-dose-response-curve-parameters.csv') %>%
  select(cell_line = depmap_id, ccle_name) %>%
  unique() %>%
  mutate(ORGAN = ccle_name %>% str_split('_') %>% sapply(function(x) x[2])) %>%
  filter(!(ORGAN %in% c('SOFT', NA))) %>%
  filter(!is.na(ORGAN))
cell_line_names$ccle_name %>% unique() %>% length()

observemoa  <- dat %>%
  left_join(cell_line_names) %>%
  left_join(MOA) %>%   filter(moa == 'MEK inhibitor') %>%
  group_by(ORGAN, molecular_names) %>%
  dplyr::summarize(meanabsLRP = (mean(abs(LRP))))

observe <- observemoa %>%
  group_by(ORGAN) %>%
  mutate(rankl = rank(meanabsLRP)) %>%
  filter(molecular_names %in% c('MAP2K1', 'MAP2K2', 'BRAF'))

ggplot(observe, aes(y = ORGAN, x = meanabsLRP)) +
  geom_point() +
  facet_wrap(~molecular_names, ncol = 1)

##################

most_important10 <- across_cell_lines %>%
  group_by(DRUG) %>%
  dplyr::mutate(cutoff = get_nth(meanabsLRP,10), scaled_LRP = meanabsLRP/max(meanabsLRP)) %>%
  filter(meanabsLRP>=cutoff) %>%
  mutate(rel = molecular_names %in% sel_genes) %>%
  mutate(DRUG = factor(DRUG, levels = order_by_moa$DRUG))

most_important30 <- across_cell_lines %>%
  group_by(DRUG) %>%
  dplyr::mutate(cutoff = get_nth(meanabsLRP,30), scaled_LRP = meanabsLRP/max(meanabsLRP)) %>%
  filter(meanabsLRP>=cutoff) %>%
  mutate(rel = molecular_names %in% sel_genes) %>%
  mutate(DRUG = factor(DRUG, levels = order_by_moa$DRUG))

most_important_gene_names10 <- most_important10 %>% ungroup() %>% select(molecular_names) %>% unique()
write.csv(most_important_gene_names10, '../results/important_genes.csv')

library(ComplexHeatmap)
drug = 'POZIOTINIB'

get_heatmap <- function(drug) {
  for_heatmap <- dat %>% dplyr::filter(molecular_names %in% most_important_gene_names10$molecular_names, DRUG == drug)

  heatmap_frame <- for_heatmap %>% dplyr::select(LRP, molecular_names, cell_line) %>%
    pivot_wider(names_from = cell_line, values_from = LRP)

  heatmap_matrix <- heatmap_frame[,-1] %>% as.matrix()
  rownames(heatmap_matrix) <- heatmap_frame$molecular_names
  Heatmap(heatmap_matrix, row_names_gp = grid::gpar(fontsize = 14), show_column_dend=F, show_row_dend=F, column_title = drug,
          show_column_names=F,show_heatmap_legend =F)

  #heatmap_matrix
}


real_ht_list <- get_heatmap('POZIOTINIB')
draw(real_ht_list)
#ht_list <- lapply(c('POZIOTINIB',  'DACOMITINIB',  'IBRUTINIB','TRAMETINIB','COBIMETINIB', 'VINCRISTINE', 'IDASANUTLIN', 'UPROSERTIB', 'TASELISIB', 'VOLASERTIB'),get_heatmap)
some_selected_drugs <- order_by_moa$DRUG

ht_list <- lapply(some_selected_drugs,get_heatmap)
some_selected_drugs %>% length()


real_ht_list <- ht_list[[1]] +  ht_list[[2]] +  ht_list[[3]] + ht_list[[4]]+ ht_list[[5]] +  ht_list[[6]] +  ht_list[[7]] +ht_list[[8]] +
  ht_list[[9]] +  ht_list[[10]] +ht_list[[11]] +  ht_list[[12]] #+ht_list[[13]] #+ht_list[[14]] +  ht_list[[15]]

png('./figures/heatmaps.png', width=6000, height=5000, res=300)
draw(real_ht_list)
dev.off()



getPalette = colorRampPalette(brewer.pal(11, 'Spectral'))

get_heatmap10_per_drug <- function(drug) {

  most_important_now <- across_cell_lines %>%
    filter(DRUG==drug) %>%
    dplyr::mutate(cutoff = get_nth(meanabsLRP,50), scaled_LRP = meanabsLRP/max(meanabsLRP)) %>%
    filter(meanabsLRP>=cutoff) %>%
    mutate(rel = molecular_names %in% sel_genes) %>%
    mutate(DRUG = factor(DRUG, levels = order_by_moa$DRUG))

  for_heatmap <- dat %>% dplyr::filter(molecular_names %in% most_important_now$molecular_names, DRUG == drug)


  heatmap_frame <- for_heatmap %>% dplyr::select(LRP, molecular_names, cell_line) %>%
    pivot_wider(names_from = cell_line, values_from = LRP)

  heatmap_matrix <- heatmap_frame[,-1] %>% as.matrix()
  rownames(heatmap_matrix) <- heatmap_frame$molecular_names

  dend = as.dendrogram(hclust(dist(heatmap_matrix)))

  description <- data.frame(cell_line = colnames(heatmap_matrix)) %>%
    left_join(cell_line_names) %>%
    dplyr::select(cell_line,ORGAN)
  unique_organs <- cell_line_names$ORGAN %>% unique() %>% sort()
  colors =  getPalette(length(unique_organs)) #colorRamp2(seq(length(unique_organs), ))
  #col_fun = colorRamp2(unique_organs, colors)
  names(colors) = unique_organs
  ha = HeatmapAnnotation(df = description %>% select(ORGAN), col = list(ORGAN = colors))

  Heatmap(heatmap_matrix, row_names_gp = grid::gpar(fontsize = 14), show_column_dend=F, show_row_dend=T, column_title = drug,
          show_column_names=F,show_heatmap_legend =F, row_dend_width = unit(4,'cm'),row_km = 10, bottom_annotation = ha)

  #heatmap_matrix
}

for (drug in some_selected_drugs) {
  png(paste0('./figures/heatmap_', drug, '.png'), width = 2000, height=1400, res=150)
  draw(get_heatmap10_per_drug(drug))
  dev.off()
}

png(paste0('./figures/heatmapscombined.png'), width = 2000, height=2000, res=150)
#par(mfrow = c(length(some_selected_drugs[1:2]), 2))
layout(seq(2))
for (drug in some_selected_drugs[1:2]) {
  print(drug)
  draw(get_heatmap10_per_drug(drug))
}
dev.off()

##########################
#show cell lines for which gene group contributes strongly for drug sensitivity
##########################
dasatinib_genes <- across_cell_lines %>%
  filter(DRUG=='DASATINIB') %>%
  dplyr::mutate(cutoff = get_nth(meanabsLRP,50), scaled_LRP = meanabsLRP/max(meanabsLRP)) %>%
  filter(meanabsLRP>=cutoff) %>%
  mutate(rel = molecular_names %in% sel_genes) %>%
  mutate(DRUG = factor(DRUG, levels = order_by_moa$DRUG))

dasatinib_data <- dat %>% dplyr::filter(molecular_names %in% dasatinib_genes$molecular_names, DRUG == 'DASATINIB')  %>%
  left_join(cell_line_names) %>%
  filter(molecular_names %in% c('BCO1', 'HNF1B', 'HAVCR1', 'PDZK1IP1'))
dasatinib_data$molecular_names %>% unique()

mean_dasatinib_data <- dasatinib_data %>% group_by(cell_line, ORGAN) %>%
  dplyr::summarize(LRP = mean(LRP))

ggplot(mean_dasatinib_data, aes(x = LRP, y = ORGAN, fill = ORGAN)) + geom_boxplot()
kidney_LRP<- mean_dasatinib_data %>% filter(ORGAN == 'KIDNEY')
no_kidney_LRP<- mean_dasatinib_data %>% filter(ORGAN != 'KIDNEY')
wilcox.test(kidney_LRP$LRP, no_kidney_LRP$LRP)
mean(no_kidney_LRP$LRP)
mean(kidney_LRP$LRP)

##########################
#plot drug resistance in several cell  types against Vincristine and Vinclastine
##########################
ABCB1_resistance <- dat %>% filter((molecular_names == 'ABCB1'), grepl('VIN', DRUG)) %>%
  left_join(cell_line_names)

ABCB1_labels <- ABCB1_resistance %>%
  group_by(ORGAN, molecular_names) %>%
  mutate(smooth = loess(LRP~expression, span=0.9)$fitted) %>%
  filter(expression ==  max(expression))

ggplot(ABCB1_resistance, aes(x = expression, y = LRP, color = ORGAN)) +
  geom_point(size=1) +
  #geom_smooth(se=F, linewidth=0.2, span=0.9) +
  geom_text_repel(aes(x = expression, y = LRP, label=ORGAN)) +
  #geom_text_repel(data = ABCB1_labels, aes(x = expression, y = smooth, label=ORGAN)) +
  theme_classic() +
  facet_grid(DRUG~molecular_names, scales='free') +
  theme_minimal()

png('./figures/VincristineVinblastine_EffectperOrgan.png', width=2000, height=1200, res=150)
ggplot(ABCB1_resistance %>% filter(!is.na(ORGAN)), aes(y = ORGAN, x = LRP, fill = DRUG)) + geom_boxplot() +
  theme_minimal() +
  theme(
    legend.title = element_blank(),
    axis.title.y = element_blank(),
    axis.text= element_text(size=13),
    axis.title = element_text(size=15),
    legend.text = element_text(size=13)

  ) +
  xlab('LRP score')

dev.off()

###########################################
#
#########################################
##########################
#plot drug resistance in several cell  types for MYOM3
##########################
MYOM3_resistance <- dat %>% filter((molecular_names == 'MYOM3'), grepl('POZIO', DRUG)) %>%
  left_join(cell_line_names)

MYOM3_labels <- MYOM3_resistance %>%
  group_by(ORGAN, molecular_names) %>%
  mutate(smooth = loess(LRP~expression, span=0.9)$fitted) %>%
  filter(expression ==  max(expression))

ggplot(MYOM3_resistance, aes(x = expression, y = LRP, color = ORGAN)) +
  geom_point(size=1) +
  geom_smooth(se=F, linewidth=0.2, span=0.9) +
  geom_text_repel(aes(x = expression, y = LRP, label=ORGAN)) +
  #geom_text_repel(data = ABCB1_labels, aes(x = expression, y = smooth, label=ORGAN)) +
  theme_classic() +
  facet_grid(DRUG~molecular_names, scales='free') +
  theme_minimal()

#png('./figures/VincristineVinblastine_EffectperOrgan.png', width=2000, height=1200, res=150)
ggplot(MYOM3_resistance %>% filter(!is.na(ORGAN)), aes(y = ORGAN, x = LRP, fill = DRUG)) + geom_boxplot() +
  theme_minimal() +
  theme(
    legend.title = element_blank(),
    axis.title.y = element_blank(),
    axis.text= element_text(size=13),
    axis.title = element_text(size=15),
    legend.text = element_text(size=13)
  ) +
  xlab('LRP score')

#dev.off()
#####################################################
#cell perspective
##########################################

cell_perspective <- dat %>% filter(molecular_names %in% most_important_gene_names) %>% filter(DRUG=='VINCRISTINE') %>%
  dplyr::group_by(cell_line, DRUG) %>%
  dplyr::mutate(sumLRP = sign(sum(LRP)) * log(abs(sum(LRP))+1))

ord <- cell_perspective %>% group_by(molecular_names) %>%
  dplyr::summarize(meanabsLRP = sd(LRP)) %>%
  arrange(desc(meanabsLRP))

cell_perspective$molecular_names <- factor(cell_perspective$molecular_names, levels = ord$molecular_names)


ggplot(cell_perspective, aes(x = molecular_names, y = LRP, group=cell_line, color = cell_line)) +
  geom_line(show.legend=F, linewidth = 0.7, alpha= 0.1) +
  #geom_point(size=0.5, alpha=0.2) +
  theme_minimal() +
  #scale_color_gradient(low='blue', high='red') +
  theme(axis.text.x = element_text(angle=90))

###############
#compare
#############
