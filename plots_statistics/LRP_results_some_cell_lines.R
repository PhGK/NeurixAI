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
library(circlize)
#genes that are relevant across drugs? sensitivity genes? essentiality?



setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
#cell_line_names <- fread('../data/secondary-screen-dose-response-curve-parameters.csv') %>%
#  select(ccle_name, cell_line = depmap_id) %>% unique()

MOA <- fread('../data/secondary-screen-dose-response-curve-parameters.csv') %>%
  select(name, moa, target, disease.area, indication) %>%
  unique() %>%
  mutate(drugs = toupper(name), unique_moa = str_split_fixed(moa, ',', 2)[,1]==moa, 
         disease = (grepl( 'malignancy',disease.area, fixed=T)) |(grepl( 'oncology',disease.area, fixed=T))) %>%
  select(DRUG = drugs, moa) %>%
  group_by(moa) %>%
  mutate(N=n()) %>%
  mutate(selected_moa = ifelse(moa %in% c('EGFR inhibitor', 'HDAC inhibitor','aromatase inhibitor', 'CDK inhibitor', 'MEK inhibitor',
                                          'topoisomerase inhibitor', 'mTOR inhibitor', 'tubulin polymerization inhibitor', 'Aurora kinase inhibitor',
                                          'glucocorticoid receptor agonist'), moa, 'other'))


cell_line_names <- fread('../data/secondary-screen-dose-response-curve-parameters.csv') %>%
  select(cell_line = depmap_id, ccle_name) %>%
  unique() %>%
  mutate(ORGAN = ccle_name %>% str_split('_') %>% sapply(function(x) x[2])) %>%
  filter(!(ORGAN %in% c('SOFT', NA))) %>%
  filter(!is.na(ORGAN))

read_dat <- function() {
  files <- list.files(paste0('../results/LRP_chosen_cell_lines/'))
  print(files)
  d <- rbindlist(lapply(files, function(f) fread(paste0('../results/LRP_chosen_cell_lines/', f))[,-1] %>% unique()))
  #d <- rbindlist(lapply(files, function(f) fread(paste0('../results/LRP/', f))[,-1] ))
  d
}



dat <- read_dat() %>% left_join(cell_line_names) %>% unique()# %>%
  #filter(abs(LRP)>0.5)

drug_dependent_genes <- dat %>% 
  group_by(cell_line, molecular_names) %>%
  dplyr::summarize(absmeanLRP = abs(mean(LRP)), std = sd(LRP)) %>%
  group_by(cell_line) %>%
  filter(absmeanLRP > quantile(absmeanLRP, 0.9)) %>%
  left_join(cell_line_names)

labels = drug_dependent_genes %>% filter(absmeanLRP > quantile(absmeanLRP,0.99))
important_genes <- dat %>%
  group_by(molecular_names) %>%
  dplyr::summarize(meanabsLRP = mean(abs(LRP))) %>%
  ungroup() %>%
  filter(meanabsLRP > quantile(meanabsLRP, 0.9))
  
important_dat <- dat %>% filter(molecular_names %in% important_genes$molecular_names)
ht_list = list()
lines <- dat$cell_line %>% unique()

for (i in seq(3)) {
for_heatmap <- important_dat %>% filter(cell_line == lines[i]) %>%
  group_by(molecular_names) %>%
  mutate(meanabsLRP = mean(abs(LRP))) %>%
  ungroup() %>%
  #filter(meanabsLRP>quantile(meanabsLRP,0.8)) %>%
  select(DRUG, molecular_names, LRP) %>%
  pivot_wider(names_from = molecular_names, values_from = LRP)

for_heatmap_matrix <- for_heatmap[,-1] %>% as.matrix()       
rownames(for_heatmap_matrix) <- for_heatmap$DRUG

cancervsnoncancer <- read.csv('../data/biomarkers.csv') %>%
  select(drugs=name, drug_category) %>%
  mutate(drugs = toupper(drugs)) %>% unique()

description <- for_heatmap %>%
  select(DRUG) %>% left_join(cancervsnoncancer %>% select(DRUG = drugs, drug_category)) %>%
  left_join(MOA)
unique_drug_types = description %>%
  mutate(drug_category = ifelse(is.na(drug_category), 'NA', drug_category)) %>%
  .$drug_category %>% unique() 

unique_selected <- description$selected_moa%>% unique()

getPalette <- colorRampPalette(brewer.pal(11, 'Spectral'))
#col_fun = colorRamp2(as.numeric(as.factor(unique_drug_types)), c('blue', 'red', 'darkgreen', 'black'))

some_colors <- getPalette(length(unique_selected))
names(some_colors) <- unique_selected
col_list = list(selected_moa = some_colors)
col_list$selected_moa['other'] <- 'white'

ha = rowAnnotation(df = description %>% select(selected_moa), col =col_list)
ht_list[i] <- Heatmap(for_heatmap_matrix, show_column_dend=F, show_column_names=F, show_row_names=F, right_annotation = ha)

}

heatmap_list <- ht_list[[1]] %v% ht_list[[2]] %v% ht_list[[3]] 

png('./figures/heatmap_prostate.png', width=3000, height=2000, res=200)
heatmap_list
dev.off()





####################
#urse par
######################
for (i in seq(4)) {
  for_heatmap <- important_dat %>% filter(cell_line == lines[i]) %>%
    group_by(molecular_names) %>%
    mutate(meanabsLRP = mean(abs(LRP))) %>%
    ungroup() %>%
    #filter(meanabsLRP>quantile(meanabsLRP,0.8)) %>%
    select(DRUG, molecular_names, LRP) %>%
    pivot_wider(names_from = molecular_names, values_from = LRP)
  
  for_heatmap_matrix <- for_heatmap[,-1] %>% as.matrix()       
  rownames(for_heatmap_matrix) <- for_heatmap$DRUG
  
  cancervsnoncancer <- read.csv('../data/biomarkers.csv') %>%
    select(drugs=name, drug_category) %>%
    mutate(drugs = toupper(drugs)) %>% unique()
  
  description <- for_heatmap %>%
    select(DRUG) %>% left_join(cancervsnoncancer %>% select(DRUG = drugs, drug_category)) %>%
    left_join(MOA)
  unique_drug_types = description %>%
    mutate(drug_category = ifelse(is.na(drug_category), 'NA', drug_category)) %>%
    .$drug_category %>% unique() 
  
  unique_selected <- description$selected_moa%>% unique()
  
  getPalette <- colorRampPalette(brewer.pal(11, 'Spectral'))
  #col_fun = colorRamp2(as.numeric(as.factor(unique_drug_types)), c('blue', 'red', 'darkgreen', 'black'))
  
  some_colors <- getPalette(length(unique_selected))
  names(some_colors) <- unique_selected
  col_list = list(`Drug class` = some_colors)
  col_list$`Drug class`['other'] <- 'white'
  
  if (lines[i]=='ACH-000019') title_name <- 'MCF7 (Breast)'
  if (lines[i]=='ACH-000090') title_name <- 'PC-3 (Prostate)'
  if (lines[i]=='ACH-000971') title_name <- 'HCT116 (Colon)'
  if (lines[i]=='ACH-000681') title_name <- 'A549 (Lung)'
  
  
  ha = rowAnnotation(df = description %>% select('Drug class' = selected_moa), col =col_list)
  png(paste0('./figures/heatmap_',lines[i], '_', title_name,'.png'), width=3000, height=1000, res=200)
  draw(Heatmap(for_heatmap_matrix[,], name = title_name, show_column_dend=F, show_column_names=F, show_row_names=F, right_annotation = ha))
  dev.off()
}

####################################
#compare general same effet vs distinctly different effects (for targeted therapies?)
######################################

LRP_variance_across_drugs <- dat %>% group_by(molecular_names, cell_line) %>%
  #dplyr::summarize(std = sd(LRP)) %>%
  dplyr::summarize(pos = sum(LRP>0), neg = sum(LRP<0), meanabsLRP = mean(abs(LRP))) %>%
  mutate(allsum = pos  +neg)  %>%
  filter(allsum>0) %>%
  mutate(perc = ifelse(pos>neg,pos/allsum, neg/allsum))
  #dplyr::arrange(desc(std))

high_variance_genes <- LRP_variance_across_drugs %>%
  filter(allsum > 800, pos>5, neg>5, meanabsLRP>1.0) %>%
  filter((neg<20)|(pos<20))

high_variance_dat <- dat %>% filter(molecular_names %in% high_variance_genes$molecular_names) %>%
  left_join(moa)





############################################
###'most genes work in the same direction for all drugs'
###############################################

summarized_results <- LRP_variance_across_drugs %>%
  group_by(cell_line) %>%
  dplyr::summarize(meanperc = mean(perc), almost_all = mean(perc<0.6), more_pos = mean(pos/neg>3), more_neg = mean(neg/pos>3), either = 1-more_pos-more_neg)

summarized_results

ggplot(LRP_variance_across_drugs, aes(x = perc)) + geom_density() +
  facet_wrap(~cell_line) +
  theme_minimal()


###########################################################
#
##########################################################

line_dat <- dat %>% filter(cell_line == lines[1]) %>%
  dplyr::select(DRUG, molecular_names, LRP) %>%
  pivot_wider(names_from = DRUG, values_from = LRP)

matrix_for_corr <- line_dat %>% select(-molecular_names) %>% as.matrix()
rownames(matrix_for_corr) <- line_dat$molecular_names

corrs <- cor(matrix_for_corr) 

description1 <- description %>%
  select(f1 = DRUG, m1 = moa, c1 = drug_category)
  
description2 <- description %>%
  select(f2 = DRUG, m2 = moa, c2 = drug_category)
  
corrs_long <-  corrs %>% 
  as.data.frame() %>%
  mutate(f1 = rownames(.)) %>%
  pivot_longer(!f1, names_to = 'f2', values_to = 'r') %>%
  left_join(description1) %>%
  left_join(description2)


corrs_long_summarized <- corrs_long %>% group_by(m1,m2, c1, c2) %>%
  filter(m1!=m2) %>%
  dplyr::summarize(meancorr = mean(r)) %>%
  filter(!is.na(c1), !is.na(c2)) %>%
  filter(m1>m2)


png('./figures/single_cell_correlations.png', width=2000, height=2000, res=200)
ggplot(corrs_long_summarized, aes(x = c1, y = meancorr)) + geom_boxplot(width=0.1) +
  facet_grid(c2~1) +
  theme_minimal()
dev.off()

