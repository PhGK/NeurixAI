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

#############################################
#moa data
MOA <- fread('../data/secondary-screen-dose-response-curve-parameters.csv') %>%
  select(name, moa, target) %>%
  unique() %>%
  mutate(DRUG = toupper(name), unique_moa = str_split_fixed(moa, ',', 2)[,1]==moa) %>%
  mutate(moa = ifelse(DRUG=='DASATINIB', 'Bcr-Abl and Src kinase inhibitor', moa))


order_by_moa <- MOA %>%
  select(moa, DRUG) %>%
  unique() %>%
  dplyr::arrange(moa)
###############################################################

read_dat <- function(FOLDER) {
  files <- list.files(paste0('../', FOLDER, '/LRP/'))
  print(files)
  #d <- rbindlist(lapply(files, function(f) fread(paste0('../results/LRP/', f))[,-1] %>% unique()))
  l = list()
  for (i in seq(5)) {
    print(i)
    dat <- fread(paste0('../', FOLDER, '/LRP/', files[i]))[,-1]
    dat$fold <- i
    l[[i]] <- dat
  }
  gc()
  rbindlist(l)
}

get_corrs <- function(FOLDER) {
  d <-read_dat(FOLDER)
  
  wide_d <- d %>%
    dplyr::select(LRP, DRUG, cell_line, molecular_names) %>%
    pivot_wider(names_from = DRUG, values_from = LRP)
  
  corr_mat <- cor(wide_d[,-c(1,2)], use="pairwise.complete.obs")
  corr_mat_long <- corr_mat %>% 
    as.data.frame() %>%
    mutate(col = rownames(.)) %>%
    pivot_longer(!col, names_to = 'row', values_to = 'corr')
  gc()
  
  corr_mat_long %>%mutate(col = factor(col, levels=order_by_moa$DRUG), row = factor(row, levels=order_by_moa$DRUG))
  
}

corr_no_embedding <- get_corrs('results') %>% mutate(type = 'no_embedding') 
ggplot(corr_no_embedding, aes(x = row, y = col, fill = corr, label=round(corr,2))) + geom_tile() + geom_text()

corr_embedding <- get_corrs('results_with_compound_embedding') %>% mutate(type = 'embedding')
ggplot(corr_no_embedding, aes(x = row, y = col, fill = corr, label=round(corr,2))) + geom_tile() + geom_text()

both_embeddings <- rbind(corr_embedding, corr_no_embedding) %>% mutate(new_type = ifelse(type=='embedding', 'Prior knowledge', 'No prior knowledge'))

heatmaps <- ggplot(both_embeddings, aes(x = row, y = col, fill = corr, label=round(corr,2))) + 
  geom_tile(show.legend=F) + 
  geom_text(data = both_embeddings %>% filter(corr!=1)) +
  facet_wrap(~new_type) +
  theme(axis.text.x = element_text(angle=90, size=13), 
        axis.text.y = element_text(size=13),  
        strip.text = element_text(size=13),
        axis.title = element_blank()) 

heatmaps

both_embeddings_wide <- both_embeddings %>%
  dplyr::select(-type) %>%
  filter(corr!=1) %>%
  pivot_wider(names_from = new_type, values_from = corr) 

comps <- ggplot(both_embeddings_wide, aes(x = `Prior knowledge`, y = `No prior knowledge`)) + 
  geom_point() + 
  geom_abline() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=90, size=13), 
        axis.text.y = element_text(size=13),  
        strip.text = element_text(size=13),
        axis.title = element_text(size=13), ) 

png('./figures/compare2noembedding.png', width=3000, height=2000, res=200)
plot_grid(heatmaps, NULL, comps, labels=c('A', 'B', ''), rel_heights = c(10,0.5,4), ncol=1)
dev.off()

############################################
####compare LRP scores directly
important_genes <- read.csv('../results/important_genes.csv')$molecular_names

get_corrs_overall <- function() {
  no_embed_ <- read_dat('results') %>% mutate(no_embed = LRP) %>% select(DRUG, cell_line, molecular_names, no_embed) %>% 
    filter(molecular_names %in% important_genes)
  embed_ <- read_dat('results_with_compound_embedding') %>% mutate(embed =LRP)  %>% select(DRUG, cell_line, molecular_names, embed) %>%
    filter(molecular_names %in% important_genes)
  
  combined <- inner_join(embed_, no_embed_)
  l <- list()
  cuts <- c(0,0.9,0.99, 0.999, 0.999999)
  for (i in seq(5)){
   q_emb = quantile(abs(embed_$embed), cuts[i], na.rm=T) %>% as.numeric()
   q_noemb = quantile(abs(no_embed_$no_embed), cuts[i], na.rm=T) %>% as.numeric()
   q_min = min(q_emb, q_noemb)
   
    combined_quantiled <- combined %>%
      filter(abs(embed)>=q_min, no_embed>=q_min)
    
    l[[i]] <- data.frame('corr' = cor(combined_quantiled$no_embed, combined_quantiled$embed), 'quantile' = cuts[i], nsamples = dim(combined_quantiled)[1])
  }
  gc()
  rbindlist(l)
}
gc()
corrs_overall <- get_corrs_overall()
#############################################

