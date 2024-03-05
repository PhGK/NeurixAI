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

meth <- 'spearman'

sanger_dat <- rbindlist(lapply(seq(5)-1, function(x) read.csv(paste0('../results/data/sanger_results', x,'.csv'))))

sanger_result <- sanger_dat %>%
  group_by(drug, fold) %>%
  dplyr::summarize(rho = cor(auc_per_drug, prediction, method = meth)) %>%
  group_by(drug) %>%
  dplyr::summarize(meanv = mean(rho), maxv = max(rho), minv = min(rho)) %>%
  arrange(meanv) %>%
  mutate(drug = factor(drug, levels = drug), above_thresh = meanv>0.2)


sanger_plot <- ggplot(result, aes(y = drug, x = meanv)) +
  geom_point() +
  geom_errorbar(aes(xmin=minv, xmax = maxv)) +
  theme_classic() +
  geom_vline(xintercept = 0.2, linetype = 'dashed')


##################
#average found drugs across folds
##############

sanger_average_captured <- dat %>%
  group_by(drug, fold) %>%
  dplyr::summarize(rho = cor(auc_per_drug, prediction, method = meth)) %>%
  mutate(above_thresh = (rho>0.2)) %>%
  group_by(fold) %>%
  dplyr::summarize(sum_found = sum(above_thresh), mean_found = mean(above_thresh))%>%
  ungroup() %>%
  dplyr::summarize(mean_captured_sum = mean(sum_found), mean_captured_mean = mean(mean_found),  min_captured_mean = min(mean_found), max_captured_mean = max(mean_found) )
