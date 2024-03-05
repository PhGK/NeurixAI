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
  #d <- rbindlist(lapply(files, function(f) fread(paste0('../results/LRP/', f))[,-1] %>% unique()))
  d <- rbindlist(lapply(files, function(f) fread(paste0('../results_with_compound_embedding/LRP/', f))[,-1] ))
  gc()
  d
}

dat <- read_dat()


across_cell_lines <- dat %>%group_by(molecular_names, DRUG) %>%
  dplyr::summarize(meanabsLRP = mean(abs(LRP))) %>%
  mutate(molecular_names = ifelse(molecular_names == 'ERBB2', 'EGFR', molecular_names)) %>%
  group_by(DRUG, molecular_names) %>%
  dplyr::summarize(meanabsLRP = mean(meanabsLRP)) %>%
  mutate(molecular_names = ifelse(molecular_names == 'EGFR', 'EGFR/ERBB2', molecular_names))


genes <- data.frame(gene = dat$molecular_names %>% unique())
dat$DRUG %>% unique()

############
#CRISPR data
#########
crispr <- fread('../data/CRISPR_(Project_Score,_Chronos).csv')[,-1]
crispr_frame <- data.frame(molecular_names = colnames(crispr), 'crispr_score' = apply(crispr,2,mean)) %>%
  mutate(molecular_names = ifelse(molecular_names == 'EGFR', 'EGFR/ERBB2', molecular_names))

viable_genes <- crispr_frame %>% filter(crispr_score<(-0.0)) %>%
  mutate(molecular_names = ifelse(molecular_names == 'EGFR', 'EGFR/ERBB2', molecular_names)) %>%
  .$molecular_names

################################
#ground truth
#############################

gt_files <- list.files('../data/drug_targets_science/')

gt_dat <- rbindlist(lapply(gt_files, function(f) fread(paste0('../data/drug_targets_science/', f))))


for_auc_gt <- gt_dat %>% dplyr::select(DRUG = Drug, molecular_names = `Gene Name`, Target = `Target Classification`, R2,
                                       nm3 = `Relative Intensity 3 nM`,
                                       nm10 = `Relative Intensity 10 nM`,
                                       nm30 = `Relative Intensity 30 nM`,
                                       nm100 = `Relative Intensity 100 nM`,
                                       nm300 = `Relative Intensity 300 nM`,
                                       nm1000 = `Relative Intensity 1000 nM`,
                                       nm3000 = `Relative Intensity 3000 nM`,
                                       nm30000 = `Relative Intensity 30000 nM`) %>%
  mutate(DRUG = toupper(DRUG)) %>%
  mutate(molecular_names = ifelse(molecular_names == 'EGFR', 'EGFR/ERBB2', molecular_names))


for_auc_combined <- for_auc_gt %>% inner_join(across_cell_lines) %>% left_join(crispr_frame) %>%
  mutate(scaled_meanabsLRP = meanabsLRP * exp(crispr_score))

for_auc_combined$Target %>% unique()
for_auc_combined_long <- for_auc_combined %>%
  pivot_longer(!c(DRUG, molecular_names, meanabsLRP, scaled_meanabsLRP, Target,R2), names_to = 'method', values_to = 'score')

ggplot(for_auc_combined, aes(x = DRUG, y = scaled_meanabsLRP, fill = Target)) + geom_boxplot()

gt_dat_long <- gt_dat %>%
  pivot_longer(!c(Drug, Lysate, Beads, `Gene Name`, `Target Classification`, `Intensity Type`, `Inflection`, EC50,
                  `Apparent Kd`), names_to = 'variable', values_to = 'score') %>%
  mutate(score = as.numeric(score)) %>%
  filter(abs(score)<1e4)

ggplot(gt_dat_long, aes(x = `Target Classification`, y = score)) +  geom_boxplot(outlier.shape=NA ) +
  facet_wrap(~variable, scales = 'free')



##################################
#AUC computation
##################################
get_auc <- function(t, v){
  res <- auc(t,v,levels = c('No binding', 'Binding'), direction = '<')
  print(res)
  res[1]
}

confident_pred <- for_auc_combined %>% filter(Target != 'Low confidence')


auc_scores <-  for_auc_combined %>%
  #filter(Target  != 'Low confidence') %>%
  mutate(Target = ifelse(Target == 'No binding', 'No binding', 'Binding')) %>%
  group_by(DRUG) %>%
  dplyr::summarize(auc_score = get_auc(Target, meanabsLRP))

auc_scores


get_nth <- function(x,n){
  x_new <- sort(x, decreasing =T)
  x_new[n]
}

for_auc_combined_description <- for_auc_combined %>%
  left_join(auc_scores) %>%
  mutate(DRUG_desription = paste0(DRUG, ' (AUC: ', round(auc_score,3), ')')) %>%
  filter(as.numeric(R2)<1e6) %>%
  mutate(EC50 = 1/log(1+as.numeric(R2))) %>%
  filter(molecular_names %in% viable_genes)


#some plotting
target_names <- for_auc_combined_description %>%
  group_by(DRUG_desription) %>%
  mutate(nthv = get_nth(meanabsLRP, 5)) %>%
  filter((Target != 'No binding')| (meanabsLRP >=nthv))

ggplot(for_auc_combined_description, aes(y = Target, x = meanabsLRP)) +
  geom_point()+
  geom_text_repel(data = target_names, aes(y = Target, x = meanabsLRP, label = molecular_names), max.overlaps=10) +
  geom_violin(data = for_auc_combined %>% filter(Target == 'No binding'),
              mapping = aes(y = Target, x = meanabsLRP), fill = 'blue',alpha=0.1) +
  facet_wrap(~DRUG_desription) +
  theme_classic()



##################################







results_overall <-  for_auc_combined_long %>% group_by(method) %>%
  dplyr::summarize(rho = cor(meanabsLRP, R2, method='spearman',use= 'pairwise.complete.obs'), r =  cor(meanabsLRP, R2, method='pearson', use= 'pairwise.complete.obs'))

results <- for_auc_combined_long %>% group_by(method, DRUG) %>%
  dplyr::summarize(rho = cor(meanabsLRP, score, method='spearman',use= 'pairwise.complete.obs'), r =  cor(meanabsLRP, score, method='pearson',use= 'pairwise.complete.obs'))

ggplot(results, aes(x = DRUG, y = r)) + geom_boxplot() +
  facet_wrap(~method)
