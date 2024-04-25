library(magrittr)
library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)
library(Hmisc)
library(cowplot)
library(DescTools)


setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

read_dat_no_embedding <- function() {
  files <- list.files(paste0('../results_without_compound_embedding//training/'))
  print(files)
  d <- rbindlist(lapply(files, function(f) fread(paste0('../results_without_compound_embedding//training/', f))))
  d %>% mutate('embedding' = F)
}

read_dat_embedding <- function() {
  files <- list.files(paste0('../results/training/'))
  print(files)
  d <- rbindlist(lapply(files, function(f) fread(paste0('../results/training/', f))))
  d %>% mutate('embedding' = T) 
  
}

dat_no_embedding <- read_dat_no_embedding() %>%
  group_by(drugs, cells, fold, epoch, embedding) %>%
  dplyr::summarize(ground_truth = mean(ground_truth), prediction = mean(prediction)) %>%
  dplyr::filter(epoch %in% c(50))

dat_embedding <- read_dat_embedding() %>%
  group_by(drugs, cells, fold, epoch, embedding) %>%
  dplyr::summarize(ground_truth = mean(ground_truth), prediction = mean(prediction)) %>%
  dplyr::filter(epoch %in% c(50))


#dat <- inner_join(dat_no_embedding, dat_embedding, by = c('drugs', 'cells', 'fold', 'epoch'))
dat <- rbind(dat_no_embedding, dat_embedding) %>%
  mutate(fold = fold+1)

meth <- 'spearman'


corrs <- dat %>% group_by(epoch, fold, drugs, embedding) %>%
  dplyr::summarize(r = cor(ground_truth, prediction, method = meth)) %>%
  filter(!is.na(r)) %>%
  group_by(epoch, fold, embedding) %>%
  dplyr::summarize(meanr = mean(r))


corrs_wide <- corrs %>% pivot_wider(names_from = embedding, values_from = meanr)


#get_p <- function(x,y) {
#  ifelse(length(x)>4,  rcorr(x, y, type = meth)$P[1,2], NA)
#}


cor_random <- dat %>% group_by(fold, drugs, embedding) %>%
  dplyr::summarize(r = cor(ground_truth, prediction, method = meth), N=n()) %>%
  filter(N>10)

cor_random_average_over_fold <- cor_random %>%
  group_by(drugs, embedding) %>%
  dplyr::summarize(meanr = mean(r)) %>%
  mutate(embedding = ifelse(embedding, 'emb', 'noemb'))

same_wide <- cor_random_average_over_fold %>% pivot_wider(names_from = embedding, values_from = meanr) %>% mutate(diffv = emb-noemb)

ggplot(same_wide, aes(x = emb, y = noemb)) + geom_point() + geom_abline()

ggplot(cor_random_average_over_fold, aes(x = embedding, y = meanr)) + geom_boxplot()

ggplot(same_wide, aes(x = 0, y = diffv)) + geom_boxplot()

#####manuscript results#####
cor_random_average_over_fold
#############################

number_of_hits <- cor_random %>% mutate(hit = (r>=0.2)) %>%
  group_by(embedding, fold) %>%
  dplyr::summarize(perc_hits = mean(hit)) %>%
  mutate(embedding = ifelse(embedding, 'With Node2Vec embedding', 'Control'))

number_of_hits_wide <- number_of_hits %>%
  pivot_wider(names_from = embedding,values_from = perc_hits)

ggplot(number_of_hits, aes(x = as.factor(embedding),  y = perc_hits,group = as.factor(fold), label = as.factor(fold))) + 
  geom_line() +
  geom_label() +
  theme_minimal()

png('./figures/compareNode2Vec_Control.png', width=2000, height=1000, res= 250)
ggplot(number_of_hits_wide, aes(x = `With Node2Vec embedding`, y = Control, label = as.factor(fold))) + 
  geom_point() +
  geom_label_repel() +
  geom_abline() +
  theme_minimal() +
  scale_y_continuous(labels = scales::percent)+ 
  scale_x_continuous(labels = scales::percent)
  
dev.off()

