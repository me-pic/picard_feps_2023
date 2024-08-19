library(lmerTest)
library(coefplot)
library(MuMIn)
library(qtl2)
library(sjPlot)
library(magrittr)
library(emmeans)
library(interactions)

################################################################################
# Description of the data

# `df_log` contains the behavioral data including the raw facs composite scores 
# (`facs`) and the log transformed facs scores (i.e., log(facs+1); `facs_log`), 
# the reported pain intensity, and the FEPS pattern expression scores 
# (dot product between the FEPS and the activation maps) for each trials. `feps` 
# contains the dot product computed with the FEPS trained on the raw facs 
# scores, and `feps_log` the dot product computed with the FEPS trained on the 
# transformed facs scores. That dataframe also has a column reporting subject id
# for each trial (`SJ`), and a column to identify in which run (`run`, value 1 
# or 2) each trial was.

# `df_dot_prod_log` contains the FEPS pattern expression scores on both the pain
# trials and the warm trials. A column `condition` separate each trial in either
# pain_facs > 0 (i.e., pain trial where a facial expression was displayed), 
# pain_facs = 0 (i.e., pain trial where no facial expression was displayed), and 
# warm. The column `warm_pain` only indicate the experimental condition without
# making any distinction between trials with facial expression vs without
# facial expression.
################################################################################


# Load data
df_log <- read_csv("<path_to_behav_ratings>") # Need to be changed according to your own path
df_dot_prod_log <- read_csv("<path_to_pattern_expression_results>") # Need to be changed according to your own path

# Transform variables
df_log['facs'] <- as.numeric(unlist(df_log['facs']))
df_log['facs_log'] <- as.numeric(unlist(df_log['facs_log']))
df_log['feps'] <- as.numeric(unlist(df_log['feps']))
df_log['feps_log'] <- as.numeric(unlist(df_log['dot_prod_log']))
df_log['trial'] <- as.numeric(unlist(df_log['trial']))
df_log['run'] <- as.numeric(unlist(df_log['run']))
df_log['int'] <- as.numeric(unlist(df_log['int']))

################################################################################
# LMM on behavioral data
################################################################################

# Model 1.1 : log(FACS+1) and pain ratings
model1_1 <- lmer(facs_log ~ int + (int | SJ),  data=df_log)
summary(model1_1)
tab_model(model1_1)

# Model 2.1: log(FACS+1) habituation or sensitization
model2_1 <- lmer(facs_log ~ trial*run + (1 | SJ), data=df_log)
summary(model2_1)
tab_model(model2_1)

# Model 2.2 : Pain ratings habituation or sensitization
model2_2 <- lmer(int ~ trial*run + (1 | SJ), data=df_log)
summary(model2_2)
tab_model(model2_2)

# Model 3.1: log(FACS+1), feps and pain ratings with interactions
model3_1 = lmer(facs_log ~ feps_log*int + (int | SJ), data=df_log)
summary(model3_1)
tab_model(model3_1)
# plot model
interact_plot(model3_1, pred=feps_log, modx=int) #library: interactions

################################################################################
# LMM on experimental conditions
################################################################################

# Model 4.1: FEPS expression scores, experimental conditions (aggregated pain trials)
df_dot_prod_log['dot_prod_log'] <- as.numeric(unlist(df_dot_prod_log['dot_prod_log']))
model4_1 = lmer(dot_prod_log ~ warm_pain + (1|id), data=df_dot_prod_log)
summary(model4_1)
anova4_1 <- anova(model4_1)
emm4_1 <- emmeans(model4_1, spec = pairwise ~ warm_pain, adjust = "tukey")
# plot model
plot(emm4_1$emmeans, comparaison=TRUE)
# compute effect size
eff_size(emm4_1, sigma=sigma(model4_1), edf=df.residual(model4_1))

# Model 4.2: FEPS expression scores, experimental conditions (separated pain trials)
df_dot_prod_log['dot_prod_log'] <- as.numeric(unlist(df_dot_prod_log['dot_prod_log']))
model4_2 = lmer(dot_prod_log ~ condition + (1|id), data=df_dot_prod_log)
summary(model4_2)
anova4_2 <- anova(model4_2)
emm4_2 <- emmeans(model4_2, spec = pairwise ~ condition, adjust = "tukey")
# plot model
plot(emm4_2$emmeans, comparaison=TRUE)
# compute effect size
eff_size(emm4_2, sigma=sigma(model4_2), edf=df.residual(model4_2))

emm4_2$emmeans
contrast(emm4_2, method = "revpairwise")

