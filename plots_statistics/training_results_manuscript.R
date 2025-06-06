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

read_dat <- function() {
  #files <- list.files(paste0('../results_without_compound_embedding/training/'))
  files <- list.files(paste0('../results/training/'))
  
  print(files)
  #d <- rbindlist(lapply(files, function(f) fread(paste0('../results_without_compound_embedding/training/', f))))
  d <- rbindlist(lapply(files, function(f) fread(paste0('../results/training/', f))))
  
  d
}

dat <- read_dat() %>%
  group_by(drugs, cells, fold, epoch) %>%
  dplyr::summarize(ground_truth = mean(ground_truth), prediction = mean(prediction))

meth <- 'spearman'

######################################################################################################
#analysis starting here
#####################################################################################################
chosen_epoch <-50

nn_dat <- dat %>% dplyr::filter(epoch %in% c(chosen_epoch)) %>%
  select(-epoch) %>%
  mutate('type' = 'nn')

cor_random <- nn_dat %>% group_by(fold, drugs) %>%
  dplyr::summarize(nn_r = cor(ground_truth, prediction, method = meth), ncells = n()) %>%
  filter(ncells>=10)

cor_random_average_over_fold <- cor_random %>%
  group_by(drugs) %>%
  dplyr::summarize(meanr = mean(nn_r), minv = min(nn_r), maxv = max(nn_r))

#####manuscript results#####
cor_random_average_over_fold
#############################


################
## drug table
library(stringr)
MOA <- fread('../data/secondary-screen-dose-response-curve-parameters.csv') %>%
  select(name, moa, target, disease.area, indication) %>%
  unique() %>%
  mutate(drugs = toupper(name), unique_moa = str_split_fixed(moa, ',', 2)[,1]==moa, 
         disease = (grepl( 'malignancy',disease.area, fixed=T)) |(grepl( 'oncology',disease.area, fixed=T))) 

results_with_moa <- cor_random_average_over_fold %>%
  left_join(MOA)

best_drugs_across_folds <- results_with_moa %>%
  arrange(desc(meanr)) %>%
  mutate(r = round(meanr,2), range = paste0(round(minv,2), ' - ', round(maxv, 2))) %>%
  dplyr::select(Drug = drugs, MOA=moa, Target = target, r, range)

write.csv(best_drugs_across_folds[1:35,] ,'./figures/best_drugs_across_folds.csv')

#########################################
#plot predictability by drug class
##########################################

cancervsnoncancer <- read.csv('../data/biomarkers.csv') %>%
  select(drugs=name, drug_category) %>%
  mutate(drugs = toupper(drugs)) %>% unique()

get_highest_value <- function(cat, N) {
  mask <- !is.na(cat)
  cat[mask][which.max(N[mask])]
}

predictability_histogram_data <- results_with_moa %>% 
  filter(unique_moa) %>%
  group_by(moa) %>%
  mutate(N=n()) %>% filter(N>=5) %>%
  left_join(cancervsnoncancer) %>%
  mutate(isnoncancer = drug_category =='noncancer') %>%
  group_by(moa, drug_category) %>%
  mutate(ncategory_per_moa = n()) 

summarized_predictabilty <- predictability_histogram_data %>% 
  group_by(moa) %>%
  dplyr::summarize(meanv= median(meanr), iqr = IQR(meanr), Ndrugs = n(), malignant = mean(disease), lowiqr = quantile(meanr, 0.25), 
                   highiqr = quantile(meanr,0.75), noncancer = mean(isnoncancer, na.rm=T)*100 %>% round(),
                   most_frequent = get_highest_value(drug_category, ncategory_per_moa)) %>%
  mutate(moa_short = ifelse(moa =='histone lysine methyltransferase inhibitor', 'HLM inhibitor', moa)) %>%
  mutate(moa_short = ifelse(moa_short == 'ALK tyrosine kinase receptor inhibitor', 'ALK TKR inhibitor', moa_short)) %>%
  mutate(moa_short = factor(moa_short, levels = moa_short[order(meanv)]))
  

MOA_plot <- ggplot(summarized_predictabilty, aes(x = moa_short, y = meanv, size=Ndrugs, color = most_frequent)) + 
  geom_hline(yintercept = 0.2, linewidth=0.4, linetype = 'dashed') +
  geom_point(aes(x = moa_short, y = meanv, size=Ndrugs)) +
  geom_errorbar(data = summarized_predictabilty, aes(ymin=lowiqr, ymax=highiqr), linewidth=0.4, show.legend = F, width=0) +
  #geom_errorbar(data = summarized_predictabilty, aes(ymin=meanv-iqr, ymax=meanv+iqr), linewidth=0.4, show.legend = F, width=0) +
  coord_flip() +
  #geom_pointrange(aes(ymin=meanv-sdv, ymax=meanv+sdv)) +
  theme_minimal() +
  ylab('Performance across drugs') + 
  theme(axis.title.y = element_blank(),
        axis.title = element_text(size=15),
        axis.text= element_text(size=10)) +
  guides(size=guide_legend(title="Number of drugs"),
         color = guide_legend(title = '')) +
  scale_color_manual(labels = c('Chemotherapy', 'Non-cancer', 'Targeted therapy'),values = c('goldenrod', 'dodgerblue1', 'firebrick'))


  #scale_color_gradient(low= 'blue', high = 'red')
MOA_plot

####################################################
##################################
#ridge_files <- list.files('../results_with_compound_embedding/other_models/')

ridge_results <- rbindlist(lapply(c(0,1,2,3,4), function(x) {read.csv(paste0('../results/other_models/prism',x, '.csv'))})) %>%
  dplyr::select(ground_truth = 'label', lasso, ridge, elastic,svr, drugs, fold, cells) %>%
  pivot_longer(c(ridge, lasso, elastic,svr), names_to='type', values_to='prediction') %>% #%>% unique() # have to do unique because some experiments were conducted multiple times
  group_by(drugs, cells, fold, type) %>%
  dplyr::summarize(ground_truth = mean(ground_truth), prediction = mean(prediction)) 

ridge_results$type %>% unique()


all_results <- rbind(nn_dat, ridge_results) 

########################################################################################################
# ????
#this depends on correlations across cell lines, maybe use direct comparisons of individual experiments?
########################################################################################################

comparison <- all_results %>%
  group_by(type, fold, drugs) %>%
  dplyr::summarize(r = cor(ground_truth, prediction, method = meth), ncells = n()) %>%
  filter(ncells>=10) 

comparison_nnvsrest_acrossfolds <- comparison %>%
  ungroup() %>%
  dplyr::select(r, type, drugs, fold) %>%
  pivot_wider(names_from=type, values_from = r)  %>%
  pivot_longer(!c(drugs, nn, fold), names_to = 'type', values_to = 'other_r') %>% mutate(nnbetter = nn>other_r) %>%
  mutate(diffv = nn-other_r) 

renamed <- data.frame(type = c('elastic', 'lasso', 'ridge', 'svr', 'nn', 'NeurixAI'), new_type = c('Elastic Net', 'LASSO', 'Ridge', 'Support vector', 'NeurixAI', 'NeurixAI'))
comparison_nnvsrest_acrossfolds <- comparison_nnvsrest_acrossfolds %>% left_join(renamed)

nnvsrest <- ggplot(comparison_nnvsrest_acrossfolds, aes(x=other_r, y = nn)) + 
  geom_point(size=0.01, alpha=0.5) + 
  geom_abline(color='skyblue1', linewidth=1) +
  facet_wrap(~new_type, nrow=1, strip.position='bottom') +
  theme_classic() +
  theme(
    axis.title = element_text(size=15),
    axis.text = element_text(size=10), 
    strip.text = element_text(size=12)
  ) +
  xlab("Baseline performance") +
  ylab("NeurixAI's performance") +
  scale_x_continuous(breaks = c(-0.2,0.2,0.6))
nnvsrest

##########################
#drugs captured per drug type and across drug types
########################
drugs_captured  <- comparison %>%
  filter(type == 'nn') %>% 
  mutate(drug_captured = ifelse(r>=0.2,T,F)) %>%
  left_join(cancervsnoncancer)  %>%
  group_by(fold, drug_category) %>%
  dplyr::summarize(average = mean(drug_captured), number = sum(drug_captured), all_drugs = n()) %>%
  group_by(drug_category) %>% 
  dplyr::summarize(average = mean(average), number = mean(number), all_drugs = mean(all_drugs))
  
drugs_captured_integer  <- comparison %>%
  filter(type == 'nn') %>% 
  left_join(cancervsnoncancer)  %>%
  group_by(drugs, drug_category) %>%
  dplyr::summarize(average_r = mean(r),all_drugs = n()) %>%
  mutate(drug_captured = ifelse(average_r>=0.2,T,F)) %>%
  group_by(drug_category) %>% 
  dplyr::summarize(mean_captured = mean(drug_captured), sum_captured = sum(drug_captured), all_drugs = n())
  

all_drugs_captured <-  comparison %>%
  #filter(type == 'nn') %>% 
  mutate(drug_captured = ifelse(r>=0.2,T,F)) %>%
  left_join(cancervsnoncancer)  %>%
  group_by(fold, type) %>%
  dplyr::summarize(average_by_fold = mean(drug_captured), number = sum(drug_captured), all_drugs = n()) %>%
  group_by(type) %>%
  dplyr::summarize(average = mean(average_by_fold), minv = min(average_by_fold), maxv = max(average_by_fold), number = mean(number), all_drugs = mean(all_drugs))

all_drugs_captured_integer  <- comparison %>%
  #filter(type == 'nn') %>% 
  left_join(cancervsnoncancer)  %>%
  group_by(drugs, type) %>%
  dplyr::summarize(average_r = mean(r),all_drugs = n()) %>%
  mutate(drug_captured = ifelse(average_r>=0.2,T,F)) %>%
  group_by(type) %>% 
  dplyr::summarize(mean_captured = mean(drug_captured), sum_captured = sum(drug_captured), all_drugs = n())


################################
#####comapre prism and sanger against baselines
###############################################


sanger_dat <- rbindlist(lapply(seq(5)-1, function(x) read.csv(paste0('../results/data/sanger_results', x,'.csv')))) %>%
  mutate( NeurixAI = prediction) %>%
  dplyr::select(fold,auc_per_drug, NeurixAI, drug, cell_line) %>%
  group_by(fold,drug, cell_line, NeurixAI) %>%
  dplyr::summarize(auc_per_drug = mean(auc_per_drug))

sanger_other_models <- rbindlist(lapply(seq(5)-1, function(x) read.csv(paste0('../results/other_models/sanger', x,'.csv')))) %>%
  mutate(cell_line = cells, drug = drugs) %>% 
  select(-c(cells,X, drugs))# %>%

sanger_corrs <- sanger_dat %>% 
  left_join(sanger_other_models, by = c('cell_line', 'drug', 'fold')) %>%
  pivot_longer(!c(label, cell_line, drug, fold, auc_per_drug), names_to = 'model', values_to = 'score') %>%
  group_by(drug, fold, model) %>%
  dplyr::summarize(rho = cor(auc_per_drug, score, method = meth), ncells = n())%>%
  filter(ncells>10) %>%
  mutate(location = 'sanger') %>% 
  left_join(renamed, by = c('model' = 'type')) %>%
  select(-c(model,ncells))

prism_corrs <- comparison %>%
  mutate( location = 'prism') %>%
  left_join(renamed) %>%
  ungroup() %>%
  dplyr::select(drug = drugs, fold, rho = r, location, new_type) 

combine_prism_sanger <- rbind(prism_corrs, sanger_corrs)

combine_mean_rho <- combine_prism_sanger %>%
  group_by(new_type, fold, location) %>%
  dplyr::summarize(rho = mean(rho)) %>%
  mutate(Location = factor(ifelse(location == 'prism', 'Internal dataset', 'External dataset'), levels = c( 'Internal dataset', 'External dataset')))


combined_across_folds_prism_sanger <- combine_mean_rho %>%
  group_by(location, new_type) %>%
  dplyr::summarize(meanrho = mean(rho), minrho = min(rho), maxrho = max(rho))


sanger_prism_performance_plot <- ggplot(combine_mean_rho, aes(x = new_type, y = rho, fill = as.factor(new_type))) + 
  geom_boxplot(width=0.1, show.legend=F) +
  geom_label_repel(aes(x=new_type, y=rho, label = fold+1), box.padding=0.5, label.padding=0.2, label.size=0.5, segment.color=NA, 
                   min.segment.length=1.0,direction='x', show.legend = F, alpha=0.7)+
  facet_wrap(~Location, nrow=1, scales='free_x') +
  theme_minimal() +
  ylab('Average Performance') +
  theme(legend.title = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size=12),
        strip.text = element_text(size=15),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size=15),
        panel.spacing = unit(5, "lines")) +
  scale_y_continuous(labels = scales::percent) 


png('./figures/prism_sanger_performance.png', height=2000, width=3000, res=250)
sanger_prism_performance_plot
dev.off()


combined_hits <- combine_prism_sanger %>%
  group_by(fold, new_type, location) %>%
  dplyr::summarize(rate = mean(rho>0.2), nhits = sum(rho>0.2))%>%
  mutate(pasted = paste0(new_type, fold)) %>%
  mutate(Location = factor(ifelse(location == 'prism', 'Internal dataset', 'External dataset'), levels = c( 'Internal dataset', 'External dataset')))


combined_hits_across_folds <- combined_hits %>%
  group_by(location, Location, new_type) %>%
  dplyr::summarize(mean_rate = mean(rate), min_rate = min(rate), max_rate = max(rate), mean(nhits)) 

sanger_prism_plot <- ggplot(combined_hits, aes(x = new_type, y = rate, fill = as.factor(new_type))) + 
  geom_boxplot(width=0.1, show.legend=T) +
  #geom_point_repel(size=4) +
  #geom_hline(data = number_of_hits_average, aes(yintercept=meanrate, group =as.factor(new_type), color=as.factor(new_type)), linetype='longdash', linewidth=1.0) +
  geom_label_repel(aes(x=new_type, y=rate, label = fold+1), box.padding=0.5, label.padding=0.2, label.size=0.5, segment.color=NA, 
                   min.segment.length=1.0,direction='x', show.legend = F, alpha=0.7)+
  #geom_line(data = combined_hits, aes(x = new_type, y = rate)) +
  
  theme_minimal() +
  facet_wrap(~Location) + 
  ylab('Drugs captured') +
  theme(legend.title = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size=12),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size=15),
        panel.spacing = unit(10, "lines")) +
  scale_y_continuous(labels = scales::percent) 

sanger_prism_plot

######################################################
#prepare table for performance and number of hits
######################################################
overall_table <- left_join(combined_across_folds_prism_sanger, combined_hits_across_folds) %>%
  mutate(meanrho = round(meanrho,3), rho_range = paste0(round(minrho,2), ' - ', round(maxrho,2)), 'captured_perc' = round(mean_rate*100,1), perc_range = paste0(round(min_rate*100,1),  ' - ', round(max_rate*100,1))) %>%
  dplyr::select(Location, Model = new_type, meanrho, rho_range, captured_perc, perc_range) %>%
  arrange((Model))

prism_table <- overall_table %>%
  filter(location == 'prism') %>%
  select(-Location)

sanger_table <- overall_table %>%
  filter(location == 'sanger') %>%
  select(-Location)

write.csv(prism_table, './figures/prism_table.csv')
write.csv(sanger_table, './figures/sanger_table.csv')

########################################
################################################
prism_number_of_hits <- combined_hits %>% filter(location == 'prism')
  
byfold_plot<- ggplot(prism_number_of_hits, aes(x = new_type, y = rate, fill = as.factor(new_type))) + 
  geom_boxplot(width=0.1, show.legend=F) +
  #geom_point_repel(size=4) +
  #geom_hline(data = number_of_hits_average, aes(yintercept=meanrate, group =as.factor(new_type), color=as.factor(new_type)), linetype='longdash', linewidth=1.0) +
  geom_label_repel(aes(x=new_type, y=rate, label = fold+1), box.padding=0.5, label.padding=0.2, label.size=0.5, segment.color=NA, 
                   min.segment.length=1.0,direction='x', show.legend = F, alpha=0.7)+
  
  theme_minimal() +
  ylab('Drugs captured') +
  theme(legend.title = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size=12),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size=15)) +
  scale_y_continuous(labels = scales::percent) 
byfold_plot


lower <- plot_grid(byfold_plot, rel_widths = c(2,3), labels = c('B', 'C'))
right_plot <- plot_grid(NULL, nnvsrest, NULL, byfold_plot, ncol=1, labels = c('B','', '','C'), rel_heights = c(0.4, 4,0.5, 6))
right_plot

png('./figures/training_results.png', height=2800, width=3500, res=300)
plot_grid(MOA_plot,right_plot, nrow=1, labels = c('A', ''), rel_widths = c(5,5))
dev.off()

number_of_hits_wide
number_of_hits %>% group_by(type) %>% dplyr::summarize(meanv = mean(rate), maxv = max(rate), minv = min(rate))


##############################################################
######CNA as predictor
#############################################################
drug_outcomes <- fread('../data/secondary-screen-dose-response-curve-parameters.csv') %>%
  select(cell_line = depmap_id, auc, ccle_name, name) %>%
  mutate(DRUG = toupper(name)) %>%
  select(-name) %>%
  filter(DRUG %in% c('IDASANUTLIN', 'POZIOTINIB', 'DACOMITINIB'))

CNA <- fread('../data/Copy_Number_Public_23Q2.csv') %>%
  select(cell_line = V1, EGFR, ERBB2, MDM2)

res <- drug_outcomes %>% left_join(CNA)  %>%
  pivot_longer(!c(cell_line, ccle_name, DRUG, auc), names_to = 'gene', values_to = 'cna')

cna_corr <- res %>% group_by(DRUG, gene) %>%
  dplyr::summarize(rho = cor(auc, cna, method= 'spearman', use = 'complete.obs'),
                   r = cor(auc, cna, method= 'pearson', use = 'complete.obs'))


##########################################
#compare individual drugs between sanger and prism
#########################################
performance_change <- combine_prism_sanger %>%
  pivot_wider(names_from = location, values_from = rho) %>%
  filter(!is.na(sanger)) %>%
  mutate(diffv = prism-sanger) 

p_values_through_folds <- performance_change %>%
  #group_by(new_type, drug, prism, sanger) %>%
  #dplyr::summarize(diffv = mean(diffv)) %>%
  group_by(new_type) %>%
  dplyr::summarize(P = wilcox.test(prism, sanger, paired = T)$p.value, diffv = mean(prism-sanger), prism = mean(prism), sanger = mean(sanger))

average_performance_change <- performance_change %>%
  group_by(fold, new_type) %>%
  dplyr::summarize(meandiffv, mindiffv = min(diffv), maxdiffv = max(diffv))

summary_performance_change_counts <- performance_change %>%
  mutate(prism_better = prism>sanger) %>%
  group_by(fold, new_type) %>%
  dplyr::summarize(meanbetter = mean(prism_better))
  
  
  

ggplot(performance_change, aes(x = prism, y = sanger, color = new_type)) +geom_point() + 
  geom_abline() +
  facet_wrap(~new_type, nrow=1)
  

t.test(compare_sanger_prism$meanv, compare_sanger_prism$meanr, paired = T)
wilcox.test(compare_sanger_prism$meanv, compare_sanger_prism$meanr, paired=T)

#######
#compare nn vs baselines
#######
comparison_of_performance_models <- prism_corrs %>% 
  pivot_wider(names_from = new_type, values_from = rho) %>%
  pivot_longer(!c(fold, location, NeurixAI, drug), names_to = 'baseline', values_to = 'baseline_rho') %>%
  group_by(baseline) %>%
  dplyr::summarize(meanNeurixAI = mean(NeurixAI),iqrNeurixAI = IQR(NeurixAI),  meanbaseline = mean(baseline_rho), iqrbaseline = IQR(baseline_rho), P = wilcox.test(NeurixAI, baseline_rho, paired=T)$p.value)

#######
#validate
#######
biomarker <- abs_LRP_over_cell_lines %>% arrange(desc(meanLRP)) %>% .[20,]
biomarker

molecular_data <- read.csv('./results/rna_data.csv') %>% select(cells = X, expression =  biomarker$molecular_names[1]) %>%
  mutate(molecular_names =  biomarker$molecular_names[1])

mol <- 'TP63'
molecular_data <- read.csv('./results/rna_data.csv') %>% select(cells = X, expression = mol) %>%
  mutate(molecular_names =  mol)

val_dat <- dat %>% select(ground_truth, prediction, cells, DRUG= drugs)  %>% inner_join(molecular_data) %>% filter(DRUG %in% abs_LRP_over_cell_lines$DRUG)
#inner_join(biomarker)

ggplot(val_dat, aes(x = expression, y = ground_truth)) + geom_point() + geom_smooth() + 
  facet_wrap(~DRUG)

val_dat %>% group_by(DRUG) %>% dplyr::summarize(r = cor(expression, ground_truth, method ='spearman'))

rcorr(val_dat$ground_truth, val_dat$expression, type='spearman')$r
rcorr(val_dat$ground_truth, val_dat$expression, type='spearman')$P






