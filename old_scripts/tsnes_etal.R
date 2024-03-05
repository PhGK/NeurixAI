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
library(Rtsne)

#genes that are relevant across drugs? sensitivity genes? essentiality?



setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
#some_dat <- fread('../data/secondary-screen-dose-response-curve-parameters.csv') %>%
#  select(depmap_id, ccle_name) %>% unique()


cell_line_names <- fread('../data/secondary-screen-dose-response-curve-parameters.csv') %>%
  select(cell_line = depmap_id, ccle_name) %>%
  unique() %>%
  mutate(ORGAN = ccle_name %>% str_split('_') %>% sapply(function(x) x[2])) %>%
  filter(!(ORGAN %in% c('SOFT', NA))) %>%
  filter(!is.na(ORGAN))


read_dat <- function() {
  files <- list.files(paste0('../results_with_compound_embedding/LRP/'))
  print(files)
  d <- rbindlist(lapply(files, function(f) fread(paste0('../results_with_compound_embedding/LRP/', f))[,-1] %>% unique()))
  #d <- rbindlist(lapply(files, function(f) fread(paste0('../results/LRP/', f))[,-1] ))
  d
}

dat <- read_dat() %>%
  filter(DRUG!= 'SELUMETINIB')

important_genes <- dat %>%
  group_by(molecular_names, DRUG) %>%
  dplyr::summarize(meanabsLRP = mean(abs(LRP))) %>%
  group_by(DRUG) %>%
  mutate(rankl = rank(desc(meanabsLRP))) %>%
  filter(rankl<100) %>%
  .$molecular_names %>% unique()

selected_cell_lines <- cell_line_names %>%
  filter(!startsWith(ORGAN, 'L'), !startsWith(ORGAN, 'O'), !startsWith(ORGAN, 'P')) %>%
  .$cell_line


head(dat)
dat_wide <- dat %>%
  dplyr::select(DRUG, cell_line, molecular_names, LRP) %>%
  filter(molecular_names %in% important_genes) %>%
  pivot_wider(names_from = molecular_names, values_from = LRP) %>%
  filter(cell_line %in% selected_cell_lines)
drugs <- dat_wide$DRUG %>% unique()

get_tsne <- function(drug) {
  print(drug)
  dat_wide_filtered <- dat_wide %>% filter(DRUG == drug)

  dat_matrix <- dat_wide_filtered %>% .[,-c(1,2)] %>% as.matrix()
  pca_dat <- Rtsne(dat_matrix, perplexity = 50)

  pca_dat_all <- pca_dat$Y %>%
    as.data.frame() %>%
    cbind(dat_wide_filtered %>% select(DRUG, cell_line)) %>%
    left_join(cell_line_names)

  pca_dat_all
}


all_tsne_dat <- rbindlist(lapply(drugs[4:12], get_tsne))

png(paste0('./figures/tsne.png'), width = 2000, height=2000, res=200)
ggplot(all_tsne_dat, aes(x = V1, y = V2, color = ORGAN)) + geom_point(size=0.5) +
  facet_wrap(~DRUG, scales = 'free') +
  theme_classic()
dev.off()
a <- head(dat)


#####################
#skin vs no skin
#######################

skinvsnoskin <- dat %>% left_join(cell_line_names) %>%
  dplyr::mutate(isskin = ifelse(ORGAN == 'SKIN', 'SKIN', 'OTHER')) %>%
  group_by(molecular_names, isskin) %>%
  dplyr::summarize(average_LRP = mean(LRP))


skinvsnoskin_wide <- skinvsnoskin %>% pivot_wider(names_from = isskin, values_from = average_LRP) %>%
  dplyr::mutate(diffv = SKIN-OTHER) %>%
  arrange(desc(diffv))






####################################################
#how many genes contribute to outcome
#######################################################
#cumsum <- function(values, ranks) {
#  ordered_values <- values[ranks]
#  output <- seq(length(ordered_values))*(0)
#  output[1] <- ordered_values[1]
#  for (i in seq(2, length(ordered_values))) {
#    output[i] <- output[i-1] + ordered_values[i]
#  }
#  output
#
#}

mean_contribution_per_gene_per_cell_line <- dat %>%
  filter(DRUG %in% c('POZIOTINIB', 'VINCRISTINE'), cell_line %in% cell_line_names$cell_line[1:30]) %>%
  group_by(molecular_names, DRUG, cell_line) %>%
  dplyr::summarize(meanabsLRP = mean(abs(LRP))) %>%
  group_by(DRUG, cell_line) %>%
  dplyr::mutate(rankl = rank(desc(meanabsLRP))) %>%
  arrange(desc(meanabsLRP)) %>%
  mutate(cumulative = base::cumsum(meanabsLRP)) %>%
  mutate(cumulative = cumulative / max(cumulative))

mean_contribution_per_gene <-mean_contribution_per_gene_per_cell_line %>%
  group_by(DRUG, rankl) %>%
  dplyr::summarize(mean_cumulative = mean(cumulative))

res1 <- mean_contribution_per_gene #%>% filter(DRUG == 'IDASANUTLIN')
text <- res1 %>% group_by(DRUG) %>%filter(rankl == max(rankl))

png(paste0('./figures/cumulative_contribution.png'), width = 2000, height=2000, res=200)
ggplot(mean_contribution_per_gene_per_cell_line, aes(x = rankl, y = cumulative, color = DRUG, group = cell_line, label = DRUG)) +
  geom_line() +
  #geom_point() +
  #geom_text_repel(data=text) +
  facet_wrap(~DRUG)
dev.off()
