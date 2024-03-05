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
#genes that are relevant across drugs? sensitivity genes? essentiality?




setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#read training dat

read_dat <- function() {
  #files <- list.files(paste0('../results_without_compound_embedding/training/'))
  files <- list.files(paste0('../results/training/'))

  print(files)
  #d <- rbindlist(lapply(files, function(f) fread(paste0('../results_without_compound_embedding/training/', f))))
  d <- rbindlist(lapply(files, function(f) fread(paste0('../results/training/', f))))

  d
}

meth <- 'spearman'


nn_dat <- read_dat() %>%
  group_by(drugs, cells, fold, epoch) %>%
  dplyr::summarize(ground_truth = mean(ground_truth), prediction = mean(prediction))%>%
  dplyr::filter(epoch ==50) %>%
  select(-epoch) %>%
  mutate('type' = 'nn')

cor_random <- nn_dat %>% group_by(fold, drugs) %>%
  dplyr::summarize(nn_r = cor(ground_truth, prediction, method = meth), ncells = n()) %>%
  filter(ncells>=10)

cor_random_average_over_fold <- cor_random %>%
  group_by(drugs) %>%
  dplyr::summarize(meanr = mean(nn_r), minv = min(nn_r), maxv = max(nn_r)) %>%
  ungroup() %>%
  filter(meanr>0.2) %>%
  dplyr::select(DRUG = drugs, meanr)


cancervsnoncancer <- read.csv('../data/biomarkers.csv') %>%
  select(DRUG=name, drug_category) %>%
  mutate(DRUG = toupper(DRUG)) %>% unique()


read_dat_LRP <- function() {
  files <- list.files(paste0('../results/LRP_specific_genes/'))
  print(files)
  d <- rbindlist(lapply(files, function(f) fread(paste0('../results/LRP_specific_genes/', f))[,-1] %>% unique()))
  #d <- rbindlist(lapply(files, function(f) fread(paste0('../results/LRP/', f))[,-1] ))
  d
}

dat_LRP <- read_dat_LRP() %>%
  left_join(cor_random_average_over_fold) %>%
  filter(meanr>0.2) %>%
  left_join(cancervsnoncancer)

names <- dat_LRP %>%
  group_by(DRUG) %>%
  filter(expression == max(expression))

ggplot(dat_LRP, aes(x = expression, y = LRP, group = DRUG, color = drug_category)) +
  #geom_point(size=2.0, alpha=0.5) +
  geom_smooth(se=F, linewidth=0.2) +
  geom_text_repel(data = names, aes(x = expression, y = LRP, label = DRUG)) +
  theme_minimal()

get_slope <- function(x,y) {
  coef(lm(y~x))[2]
}
get_slope(dat_LRP$expression, dat_LRP$LRP)

summarized <- dat_LRP %>%
  group_by(DRUG, drug_category) %>%
  dplyr::summarize(corr = get_slope(expression, LRP), N=n()) %>%
  filter(!is.na(corr), !is.na(drug_category))

png('./figures/compare_ABCB1_among_drugs.png', width=1500, height=1000, res=150)
ggplot(summarized, aes (x = drug_category, y =corr, label = DRUG)) +
  geom_boxplot() +
  geom_jitter(position = position_jitter(seed = 1)) +
  geom_text_repel(position = position_jitter(seed = 1), size=3) +
  theme_classic() +
  theme(axis.title.x = element_blank()) +
  ylab('Slope')
dev.off()
res_formanuscript <- summarized %>% group_by(drug_category) %>%
  dplyr::summarize(medianv = median(corr), iqr = IQR(corr))

wilcox.test(summarized %>% filter(drug_category == 'chemo') %>% .$corr, summarized %>% filter(drug_category == 'noncancer') %>% .$corr)
wilcox.test(summarized %>% filter(drug_category == 'chemo') %>% .$corr, summarized %>% filter(drug_category == 'targeted cancer') %>% .$corr)
