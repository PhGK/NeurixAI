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
  files <- list.files(paste0('../results_with_compound_embedding/training/'))
  print(files)
  d <- rbindlist(lapply(files, function(f) fread(paste0('../results_with_compound_embedding/training/', f))))
  d
}

dat <- read_dat() %>%
  group_by(drugs, cells, fold, epoch) %>%
  dplyr::summarize(ground_truth = mean(ground_truth), prediction = mean(prediction))

meth <- 'spearman'

corrs <- dat %>% group_by(epoch, fold, drugs) %>%
  dplyr::summarize(r = cor(ground_truth, prediction, method = meth)) %>%
  filter(!is.na(r)) %>%
  group_by(epoch, fold) %>%
  dplyr::summarize(meanr = mean(r))
ggplot(corrs, aes(x = epoch, y = meanr, color = as.factor(fold), group = as.factor(fold))) + geom_point() + geom_line() #+

get_p <- function(x,y) {
  ifelse(length(x)>4,  rcorr(x, y, type = meth)$P[1,2], NA)
}

######################################################################################################
#analysis starting here
#####################################################################################################
chosen_epoch <-50


nn_dat <- dat %>% dplyr::filter(epoch %in% c(chosen_epoch)) %>%
  select(-epoch) %>%
  mutate('type' = 'nn')

cor_random <- nn_dat %>% group_by(fold, drugs) %>%
  dplyr::summarize(nn_r = cor(ground_truth, prediction, method = meth), nn_p = get_p(ground_truth, prediction), ncells = n()) %>%
  group_by(fold) %>%
  mutate(nn_adjusted_p = p.adjust(nn_p, method = 'fdr')) %>% 
  filter(ncells>=10)

cor_random_average_over_fold <- cor_random %>%
  group_by(drugs) %>%
  dplyr::summarize(meanr = mean(nn_r), minv = min(nn_r), maxv = max(nn_r))

#####manuscript results#####
cor_random_average_over_fold
#############################

#########################################
#plot results back before normalization
#######################################
#AUC_ground_truth <-fread('../use_data/prediction_targets.csv') %>%
 # dplyr::select(cell_line, DRUG, auc, auc_per_drug, means, stds) %>%
 # inner_join(nn_dat, by = c('DRUG'='drugs','cell_line' = 'cells')) %>%
 # mutate(pred2_prenormal = (prediction*stds)+ means, control_ground_truth2_prenormal = (ground_truth*stds+means)) %>%
 #   mutate(diffs = abs(auc-control_ground_truth2_prenormal)) %>%
 #   filter(abs(diffs)<1e-5)

#AUC_ground_truth %>% dplyr::summarize(corrs = cor(pred2_prenormal, auc, method='spearman'), N=n())
#mean(abs(AUC_ground_truth$auc- AUC_ground_truth$control_ground_truth2_prenormal)<1e-2)

#ggplot(AUC_ground_truth, aes(x=auc, y = pred2_prenormal, pred)) + geom_point(size=0.2) +
#  theme_minimal()

#ggplot(AUC_ground_truth, aes(x = (1/auc))) + geom_density(fill='black')


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
  mutate(r = round(meanr,2)) %>%
  dplyr::select(Drug = drugs, MOA=moa, Target = target, r)

write.csv(best_drugs_across_folds[1:35,] ,'./figures/best_drugs_across_folds.csv')

#########################################
#plot predictability by drug class
##########################################

predictability_histogram_data <- results_with_moa %>% 
  filter(unique_moa) %>%
  group_by(moa) %>%
  mutate(N=n()) %>% filter(N>=5)

summarized_predictabilty <- predictability_histogram_data %>% 
  group_by(moa) %>%
  dplyr::summarize(meanv= median(meanr), iqr = sd(meanr), Ndrugs = n(), malignant = mean(disease), lowiqr = quantile(meanr, 0.25), highiqr = quantile(meanr,0.75)) %>%
  mutate(moa_short = ifelse(moa =='histone lysine methyltransferase inhibitor', 'HLM inhibitor', moa)) %>%
  mutate(moa_short = ifelse(moa_short == 'ALK tyrosine kinase receptor inhibitor', 'ALK TKR inhibitor', moa_short)) %>%
  mutate(moa_short = factor(moa_short, levels = moa_short[order(meanv)]))
  

MOA_plot <- ggplot(summarized_predictabilty, aes(x = moa_short, y = meanv, size=Ndrugs)) + 
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
  guides(size=guide_legend(title="Number of drugs")) #+
  #scale_color_gradient(low= 'blue', high = 'red')
MOA_plot

####################################################
##################################
ridge_files <- list.files('../results/other_models/')

ridge_results <- rbindlist(lapply(ridge_files, function(x) {read.csv(paste0('../results/other_models/',x))})) %>%
  dplyr::select(ground_truth = 'label', lasso, kernel_ridge, ridge, elastic,svr, drugs, fold, cells) %>%
  pivot_longer(c(ridge, kernel_ridge, lasso, elastic,svr), names_to='type', values_to='prediction') %>% #%>% unique() # have to do unique because some experiments were conducted multiple times
  group_by(drugs, cells, fold, type) %>%
  dplyr::summarize(ground_truth = mean(ground_truth), prediction = mean(prediction)) %>%
  filter(type != 'kernel_ridge')

ridge_results$type %>% unique()


all_results <- rbind(nn_dat, ridge_results) 

test <- ridge_results%>%
  #select(-c(prediction, ground_truth)) %>% unique()
  group_by(drugs, fold, cells, type) %>% 
  dplyr::mutate(N=n()) %>%
  filter(N>1)
  
########################################################################################################
# ????
#this depends on correlations across cell lines, maybe use direct comparisons of individual experiments?
########################################################################################################


comparison <- all_results %>%
  group_by(type, fold, drugs) %>%
  dplyr::summarize(r = cor(ground_truth, prediction, method = meth), p = get_p(ground_truth, prediction), ncells = n()) %>%
  group_by(type,fold) %>%
  dplyr:: mutate(adjusted_p = p.adjust(p, method = 'fdr')) %>%
  filter(ncells>10) 

comparison_wide <- comparison %>%
  ungroup() %>%
  dplyr::select(r, type, drugs, fold) %>%
  pivot_wider(names_from=type, values_from = r) 

comparison_nnvsrest_acrossfolds <- comparison_wide %>%
  pivot_longer(!c(drugs, nn, fold), names_to = 'type', values_to = 'other_r') %>% mutate(nnbetter = nn>other_r) %>%
  mutate(diffv = nn-other_r) 

# with folds results
comparison_nnvsrest_acrossfolds %>% 
  group_by(type, fold) %>% 
  dplyr::summarize(perc = mean(nnbetter, na.rm=T), N=n()) %>%
  group_by(type) %>%
  dplyr::summarize(mean_perc = mean(perc),  minv = min(perc), maxv = max(perc))


comparison_meandiff <- comparison_nnvsrest_acrossfolds %>%
  group_by(type) %>%
  dplyr::summarize(meandiff = mean(diffv)) 

number_of_hits <- comparison %>%
  #dplyr::mutate(hit = (adjusted_p<0.05 & r>0)) %>%
  #dplyr::mutate(hit = (adjusted_p<0.05 & r>0.2)) %>%
  dplyr::mutate(hit = (r>=0.2)) %>%
  filter(ncells>10) %>%
  group_by(type,fold) %>%
  dplyr::summarize(rate = sum(hit)/n()) %>%
  arrange(fold)

number_of_hits_average <- number_of_hits %>% group_by(type) %>%
  dplyr::summarize(meanrate=mean(rate), minrate=min(rate), maxrate = max(rate))


number_of_hits_wide <- number_of_hits %>%
  pivot_wider(names_from=type, values_from = c(rate)) 
number_of_hits_wide


renamed <- data.frame(type = c('elastic', 'lasso', 'kernel_ridge', 'ridge', 'svr', 'nn'), new_type = c('Elastic Net', 'LASSO', 'Kernel Ridge', 'Ridge', 'Support vector', 'Our model'))
comparison_nnvsrest_acrossfolds <- comparison_nnvsrest_acrossfolds %>% left_join(renamed)

nnvsrest <- ggplot(comparison_nnvsrest_acrossfolds, aes(x=other_r, y = nn)) + 
  geom_point(size=0.01, alpha=0.5) + 
  geom_abline(color='skyblue1', linewidth=1) +
  #geom_abline(color='gray90', linewidth=1) +
  facet_wrap(~new_type, nrow=1, strip.position='bottom') +
  #theme_minimal() +
  theme_classic() +
  theme(
    axis.title = element_text(size=15),
    axis.text = element_text(size=10), 
    strip.text = element_text(size=12)
  ) +
  xlab("Baseline performance") +
  ylab("Our model's performance") +
  scale_x_continuous(breaks = c(-0.2,0.2,0.6))
nnvsrest

#################################################################################################
####################################try direct comparisons of experiments
###################################################################################################
#no folds
compare_direct <- all_results %>% 
  mutate(error = abs(ground_truth-prediction)**2) %>%
  dplyr::select(drugs, cells, error, type, fold) %>%
  unique() %>%
  pivot_wider(names_from = type, values_from = error) %>%
  pivot_longer(c(ridge, elastic, lasso, svr), names_to = 'baseline', values_to = 'error') %>%
  mutate(error_diff = error-nn)


 compare_direct %>% group_by(fold, baseline) %>% 
  dplyr::summarize(meanv = mean(error_diff), mean_nn = mean(nn), mean_baseline = mean(error)) %>%
  #filter(baseline=='svr') %>%
  #ungroup() %>%
  group_by(baseline) %>%
  dplyr::summarize(mean_meanv = mean(meanv), minv = min(meanv), maxv = max(meanv), mean_mean_nn = mean(mean_nn),
                   min_mean_nn = min(mean_nn),max_mean_nn = max(mean_nn), mean_mean_baseline = mean(mean_baseline),
                   min_mean_baseline = min(mean_baseline), max_mean_baseline = max(mean_baseline))

################################################

number_of_hits <- number_of_hits %>% left_join(renamed)
number_of_hits_average <- number_of_hits_average %>% left_join(renamed)



byfold_plot1<- ggplot(number_of_hits, aes(x = new_type, y = rate, fill = as.factor(new_type))) + 
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
byfold_plot1

byfold_plot2<- ggplot(number_of_hits, aes(x = fold+1, y = rate, group=type, color = as.factor(new_type))) + 
  geom_point(size=4, alpha=1.0) +
  geom_hline(data = number_of_hits_average, aes(yintercept=meanrate, color=as.factor(new_type)), linetype='dashed', linewidth=1.1) +
  theme_minimal() +
  ylab('Drugs captured') +
  xlab('Fold') +
  theme(legend.title = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title = element_text(size=15),
        axis.text = element_text(size=12),
        panel.grid.major.x = element_line(color='gray60'),
        panel.grid.major.y = element_blank() 
  ) +
  scale_y_continuous(labels = scales::percent) #+
  #ylim(c(0.2,0.7))
byfold_plot2

byfold_plot <- byfold_plot1
byfold_plot


lower <- plot_grid(byfold_plot, rel_widths = c(2,3), labels = c('B', 'C'))
right_plot <- plot_grid(NULL, nnvsrest, NULL, byfold_plot, ncol=1, labels = c('B','', '','C'), rel_heights = c(0.4, 4,0.5, 6))
right_plot

png('./figures/training_results.png', height=2800, width=3500, res=300)
plot_grid(MOA_plot,right_plot, nrow=1, labels = c('A', ''), rel_widths = c(5,5))
dev.off()

number_of_hits_wide
number_of_hits %>% group_by(type) %>% dplyr::summarize(meanv = mean(rate), maxv = max(rate), minv = min(rate))

t.test(number_of_hits_wide$nn, number_of_hits_wide$ridge, paired=T)
t.test(number_of_hits_wide$nn, number_of_hits_wide$lasso, paired=T)
t.test(number_of_hits_wide$nn, number_of_hits_wide$elastic, paired=T)
t.test(number_of_hits_wide$nn, number_of_hits_wide$svr, paired=T)

shapiro.test(number_of_hits_wide$nn - number_of_hits_wide$svr)
shapiro.test(number_of_hits_wide$nn - number_of_hits_wide$ridge)
shapiro.test(number_of_hits_wide$nn - number_of_hits_wide$lasso)
shapiro.test(number_of_hits_wide$nn - number_of_hits_wide$elastic)

wilcox.test(number_of_hits_wide$nn, number_of_hits_wide$ridge, paired=T)
wilcox.test(number_of_hits_wide$nn, number_of_hits_wide$lasso, paired=T)
wilcox.test(number_of_hits_wide$nn, number_of_hits_wide$elastic, paired=T)
(mean(number_of_hits_wide$nn)- mean(number_of_hits_wide$ridge))/mean(number_of_hits_wide$ridge)
(mean(number_of_hits_wide$nn)- mean(number_of_hits_wide$ridge))/mean(number_of_hits_wide$lasso)
(mean(number_of_hits_wide$nn)- mean(number_of_hits_wide$ridge))/mean(number_of_hits_wide$elastic)

#df4chsq <- data.frame(c(number_of_hits$nn_sum, number_of_hits$N-number_of_hits$nn_sum), c(number_of_hits$ridge_sum, number_of_hits$N-number_of_hits$ridge_sum))
#chisq.test(df4chsq)$p.value


################################################
# specificity of drug effect
################################################
good_drugs <- cor_random %>% filter(adjusted_p<0.05, r>0) %>%.$drugs
#is this correct?
all_predictions <- read_dat() %>% filter(epoch==0, fold==4, drugs %in% good_drugs) %>%
  group_by(drugs, cells, fold, epoch) %>%
  dplyr::summarize(ground_truth = mean(ground_truth), prediction = mean(prediction), N=n()) %>%
  group_by(drugs, fold, epoch) %>%
  mutate(gt_rank = rank(desc(ground_truth)), pred_rank = rank(desc(prediction)), N_drugs=n()) %>%
  group_by(cells, fold, epoch) %>%
  dplyr::summarize(minrank_gt = min(gt_rank), minrank_pred = pred_rank[which(gt_rank == min(gt_rank))], N_drugs=mean(N_drugs)) %>%
  mutate(gt_perc = minrank_gt/N_drugs, pred_perc = minrank_pred/N_drugs)

rcorr(all_predictions$minrank_gt, all_predictions$minrank_pred, type='spearman')
ggplot(all_predictions, aes(x=as.factor(minrank_pred), y=as.factor(minrank_gt))) + geom_point()

summary(all_predictions$minrank_gt %>% as.factor())


best_sep <- all_predictions %>% filter(minrank_gt==1)
ggplot(best_sep, aes(x = fold, y = minrank_pred)) + geom_violin(fill='gray90') + geom_jitter() +theme_minimal()
ggplot(best_sep, aes(x = fold, y = minrank_pred)) + geom_boxplot(fill='gray90') + geom_jitter() +theme_minimal() +
  facet_grid(fold~epoch)






##############################################################
#sanger test results
##############################################################

sanger_results <- read.csv('./results/sanger_results.csv')
sanger_corr <- sanger_results %>% group_by(drug) %>%
  dplyr::summarize(r = cor(ground_truth, prediction, method='spearman'), p = get_p(ground_truth, prediction), ncells = n()) %>%
  ungroup() %>%
  dplyr::mutate(adjusted_p = p.adjust(p, method = 'fdr')) %>%
  mutate(hit = (r>0)& (adjusted_p<0.05))

sanger_corr$hit %>% mean()
sanger_corr$r %>% mean()

broad_results <- comparison %>%
  filter(type=='nn', fold==1)  %>%
  select(broad_r=r, broad_p = p, drug=drugs)

compare_brad_sanger <- sanger_corr %>% inner_join(broad_results)

compare_brad_sanger %>% dplyr::summarize(diesdas = cor(r, broad_r, method='pearson'))

sanger_performance<- ggplot(compare_brad_sanger, aes(x = r, y= broad_r)) + 
  geom_point() +
  geom_text_repel(aes(x = r, y= broad_r, label=drug)) +
  #geom_text(aes(x = r, y= broad_r, label=drug)) +
  geom_abline() +
  theme_minimal()+
  #ylim(c(-1,1))+
  #xlim(c(-1,1)) +
  ylab('Broad performance') +
  xlab('Sanger performance')

png('./plots_statistics/figures/broad_r_vs_sanger_r.png', width=1500, height=1500, res=150)
sanger_performance
dev.off()
g















#########################################################
#LRP
########################################################

relevant_drugs <-  dat %>% group_by(epoch, fold, drugs) %>%
  dplyr::summarize(r = cor(ground_truth, prediction, method = 'pearson'), p = get_r(ground_truth, prediction), ncells = n()) %>%
  group_by(epoch, fold) %>%
  mutate(adjusted_p = p.adjust(p, method = 'fdr')) %>%
  filter(adjusted_p<0.05, r >0.5) %>% .$drugs
write.csv(data.frame('relevant_drugs' = relevant_drugs), './results/relevant_drugs.csv')

read_LRP <- function(x) {
  fread(paste0('./results/LRP/LRP_',x,'.csv')) %>% filter(DRUG %in% relevant_drugs)
}
#LRP_scores <- fread('./results/LRP/LRP_0.csv') %>% filter(DRUG %in% relevant_drugs) %>% select(-V1) 
LRP_scores <- rbindlist(lapply(seq(4)-1, read_LRP))

LRP_scores_now <- LRP_scores %>% unique() # why is this necessary?

abs_LRP_over_cell_lines <- LRP_scores_now %>% 
  group_by(DRUG, molecular_names) %>% 
  dplyr::summarize(meanLRP = mean(abs(LRP))) %>%
  group_by(DRUG) %>% mutate(rankl = rank(desc(meanLRP)))



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






