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

#genes that are relevant across drugs? sensitivity genes? essentiality?


setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

read_dat <- function() {
  files <- list.files(paste0('../results_with_compound_embedding/LRP/'))
  print(files)
  d <- rbindlist(lapply(files, function(f) fread(paste0('../results_with_compound_embedding/LRP/', f))[,-1] %>% unique()))
  #d <- rbindlist(lapply(files, function(f) fread(paste0('../results/LRP/', f))[,-1] ))
  
  d
}

dat <- read_dat() 


across_cell_lines <- dat %>%group_by(molecular_names, DRUG) %>%
  dplyr::summarize(meanabsLRP = mean(abs(LRP))) %>%
  group_by(DRUG) %>%
  mutate(rankl = rank((meanabsLRP))) %>%
  mutate(pos = rankl/max(rankl)) %>%
  mutate(percent_rank = pos * 100)

genes <- data.frame(gene = dat$molecular_names %>% unique())

sel_genes <-  c('ERBB2', 'EGFR', 'MDM2', 'TP53', 'MAP2K1','MAP2K6', 'MAP3K21')

rel_gene_names <- across_cell_lines %>%
  filter(molecular_names %in% sel_genes)

#############################################
#find Cobimetinib import features
cob <- across_cell_lines %>% filter(grepl('MAP',molecular_names), DRUG == 'COBIMETINIB')

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
        axis.title = element_blank(),
        axis.text.y = element_text(size=15),
        axis.text.x = element_text(size=15),
        legend.title = element_blank(),
        legend.text = element_text(size=15),
        legend.position = 'bottom') +
  #ylab('Low importance rank     \U2194     High importance rank')  +
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

#for reactome
#write.csv(most_important %>% filter(DRUG == 'DACOMITINIB') %>%.$molecular_names %>% unique(), './results/data/important_genes.csv', row.names = FALSE, quote=FALSE)
#write.csv(most_important %>% filter(DRUG == 'IDASANUTLIN') %>%.$molecular_names %>% unique(), './results/data/important_genes.csv', row.names = FALSE, quote=FALSE)
#write.csv(most_important %>% filter(DRUG == 'POZIOTINIB') %>%.$molecular_names %>% unique(), './results/data/important_genes.csv', row.names = FALSE, quote=FALSE)
#write.csv(most_important %>% filter(DRUG == 'VINCRISTINE') %>%.$molecular_names %>% unique(), './results/data/important_genes.csv', row.names = FALSE, quote=FALSE)
#write.csv(most_important %>% filter(DRUG == 'TRAMETINIB') %>%.$molecular_names %>% unique(), './results/data/important_genes.csv', row.names = FALSE, quote=FALSE)
#write.csv(most_important %>% filter(DRUG == 'TASELISIB') %>%.$molecular_names %>% unique(), './results/data/important_genes.csv', row.names = FALSE, quote=FALSE)
#write.csv(most_important %>% filter(DRUG == 'UPROSERTIB') %>%.$molecular_names %>% unique(), './results/data/important_genes.csv', row.names = FALSE, quote=FALSE)
#write.csv(most_important %>% filter(DRUG == 'IBRUTINIB') %>%.$molecular_names %>% unique(), './results/data/important_genes.csv', row.names = FALSE, quote=FALSE)
#write.csv(most_important %>% filter(DRUG == 'PELITINIB') %>%.$molecular_names %>% unique(), './results/data/important_genes.csv', row.names = FALSE, quote=FALSE)
#write.csv(most_important %>% filter(DRUG == 'AZD8330') %>%.$molecular_names %>% unique(), './results/data/important_genes.csv', row.names = FALSE, quote=FALSE)
#write.csv(most_important %>% filter(DRUG == 'AZD8931') %>%.$molecular_names %>% unique(), './results/data/important_genes.csv', row.names = FALSE, quote=FALSE)


most_important_only_red <- most_important %>%
  filter(molecular_names %in% sel_genes)
most_important_without_red <- most_important %>%
  filter(!(molecular_names %in% sel_genes))

most_important_genes_plot_rank <- ggplot(most_important, aes(x = DRUG, y = scaled_LRP, label = molecular_names)) + 
  geom_line(data = line_data, aes(x = DRUG, y= x, color=moa), linewidth=1) +
  geom_text_repel(data = most_important_without_red, show.legend=F, size=4, color='black', max.overlaps=100) +
  geom_text_repel(data = most_important_only_red, show.legend=F, size=4, color='red', max.overlaps = 1000) +
  geom_point(data=most_important_without_red, aes(x = DRUG, y = scaled_LRP),color='black', size=0.2) +
  geom_point(data=most_important_only_red, aes(x = DRUG, y = scaled_LRP),color='red') +
  theme_minimal() +
  scale_y_continuous(trans='log10') +
  theme(axis.text = element_text(size=15),
        axis.title.y = element_blank(),
        axis.title.x = element_text(size=20),
        legend.title = element_blank(),
        legend.text = element_text(size=15),
        legend.position = 'bottom') +
  ylab('xAI-assigned Importance') +
  coord_flip() + 
  scale_x_discrete(limits=rev) 
  

most_important_genes_plot_rank
png('./figures/important_genes.png', width=3000, height=1500, res=200)
relevantometer
dev.off()

png('./figures/important_genes_scaled.png', width=3000, height=2200, res=150)
most_important_genes_plot_rank
dev.off()

############
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
  dplyr::summarize(diffv = max(rankl)- min(rankl), maxv = max(pos)) %>%
  arrange(desc(diffv)) %>%
  filter(maxv > 0.99)
moas <- difference_plus_moa$moa %>% unique()

number <- 4

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
ggplot(diff_data, aes(y = molecular_names, x = meanabsLRP, label = DRUGpos)) +
  geom_point(size=0.5)+
  geom_text_repel() +
  xlab('LRP importance') +
  theme(axis.title.y = element_blank(), axis.text = element_text(size=12),
        axis.title.x = element_text(size=14))
dev.off()5##########
# see relationship between expression and LRP
##########
#only selected genes
selected_genes_dynamics <- dat %>% filter(molecular_names %in% rel_gene_names$molecular_names)
last_name <- selected_genes_dynamics %>% group_by(DRUG, molecular_names) %>%
  dplyr::mutate(smooth = loess(LRP~expression)$fitted) %>%
  filter(expression == max(expression))

most_important_genes_dynamics <- dat %>% filter(molecular_names %in% most_important$molecular_names)
  
dynamics_selected_genes <- ggplot(selected_genes_dynamics, aes(x = expression, y = LRP, color=DRUG)) + 
  geom_point(size=0.1) +
  geom_smooth(se=F, linewidth=0.4) +
  geom_text_repel(data = last_name, aes(x = expression, y =smooth, label = DRUG), size=3, color = 'black', hjust= -0.5, direction = 'y') +
  facet_wrap(~molecular_names, scales='free') +
  theme_classic() +
  xlim(c(-2,6)) +
  guides(color = guide_legend(override.aes = list(size = 5))) +
  xlab('Gene expression') +
  ylab('LRP contribution') +
  theme(strip.text = element_text(size=15),
        axis.text = element_text(size=10),
        axis.title = element_text(size=13))

png('./figures/dynamics_of_LRPcontribution.png', width=4000, height=2500, res=250)
dynamics_selected_genes
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

observe <- dat %>% filter(molecular_names %in% c('MAP2K1', 'MAP2K2', 'BRAF')) %>%
  left_join(cell_line_names) %>%
  group_by(ORGAN, molecular_names) %>%
  dplyr::summarize(meanabsLRP = mean(abs(LRP)))

ggplot(observe, aes(y = ORGAN, x = meanabsLRP)) +
  geom_point() +
  facet_wrap(~molecular_names, ncol = 1)

##################


most_important10 <- across_cell_lines %>%
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
  Heatmap(heatmap_matrix, row_names_gp = grid::gpar(fontsize = 10), show_column_dend=F, show_row_dend=F, column_title = drug, 
          show_column_names=F,show_heatmap_legend =F)
  
  #heatmap_matrix
}
real_ht_list <- get_heatmap('POZIOTINIB')
draw(real_ht_list)
#ht_list <- lapply(c('POZIOTINIB',  'DACOMITINIB',  'IBRUTINIB','TRAMETINIB','COBIMETINIB', 'VINCRISTINE', 'IDASANUTLIN', 'UPROSERTIB', 'TASELISIB', 'VOLASERTIB'),get_heatmap)
some_selected_drugs <- order_by_moa$DRUG

ht_list <- lapply(some_selected_drugs,get_heatmap)
some_selected_drugs


real_ht_list <- ht_list[[1]] +  ht_list[[2]] +  ht_list[[3]] + ht_list[[4]]+ ht_list[[5]] +  ht_list[[6]] +  ht_list[[7]] +ht_list[[8]] +  
  ht_list[[9]] +  ht_list[[10]] +ht_list[[11]] +  ht_list[[12]] +ht_list[[13]] #+ht_list[[14]] +  ht_list[[15]]

png('./figures/heatmaps.png', width=4000, height=4000, res=250)
draw(real_ht_list)
dev.off()



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
