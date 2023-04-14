# ------------------------------------------------------------------------------
# DATA PREPROCESSING
# ------------------------------------------------------------------------------

# Import raw data
dat <- read.table("athlete_events.csv", sep = ",", header = TRUE)

# Observations and rows of raw data
dim(dat)

# Remove fields ID, Name, Event, NOC, Games
library(dplyr)
dat <- select(dat, -c("ID", "Name", "Team","Event", "City", "Games", "Year"))

# Remove outliers - refer to Unsupervised Outlier Detection
dat <- dat[-OutlierIndex,]

# Change Medal NA to "No medal"
dat$Medal[is.na(dat$Medal)] <- "No medal"

# Factorise the variables with chr datatype
dat$Sex <- factor(dat$Sex, levels = c("M", "F"))
dat$NOC <- factor(dat$NOC)
dat$Season <- factor(dat$Season)
dat$Sport <- factor(dat$Sport)
dat$Medal <- factor(dat$Medal, 
                    levels = c("No medal", "Bronze", "Silver", "Gold"),
                    ordered = TRUE)

# Number of missing values in each feature
count_miss_vals <- function(x) sum(is.na(x))
apply(dat, MARGIN = 2, FUN = count_miss_vals)

# Number of records not affected by missing values
dim(na.omit(dat)[1])

# Proportion of all observations by Medal
summarise(group_by(dat, Medal), Count = n()) %>% 
  mutate(percent = Count / sum(Count))

# Proportion of missing values by Medal
Missing_vals <- dat %>% 
                filter(!complete.cases(Age, Height, Weight))

summarise(group_by(Missing_vals, Medal), Count = n()) %>% 
                                mutate(percent = Count / sum(Count))

# Impute missing values of Age, Height and Weight with mean by subgroup of sport
dat$Age <-ave(dat$Age, dat$Sport, 
              FUN = function(x) ifelse(is.na(x), mean(x,na.rm = TRUE), x))
dat$Height <-ave(dat$Height, dat$Sport, 
                 FUN = function(x) ifelse(is.na(x), mean(x,na.rm = TRUE), x))
dat$Weight <-ave(dat$Weight, dat$Sport, 
                 FUN = function(x) ifelse(is.na(x), mean(x,na.rm = TRUE), x))

# Remove few remaining missing values
dat <- na.omit(dat)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------









# ------------------------------------------------------------------------------
# UNSUPERVISED OUTLIER DETECTION
# ------------------------------------------------------------------------------

# Subset data for numeric variables
datOutlier <- dplyr::select(dat, Age:Weight)

# KNN Outlier - distance to its 10th nearest neighbour
library(dbscan)
OutlierAge_Height <- kNNdist(datOutlier[, 1:2], k = 10, all = TRUE)[, 10]
OutlierAge_Weight <- kNNdist(datOutlier[, c(1,3)], k = 10, all = TRUE)[, 10]

top_nHeight <- 80
top_nWeight <- 80
rankAge_Height <- order(x = OutlierAge_Height, decreasing = TRUE) 
rankAge_Weight <- order(x = OutlierAge_Weight, decreasing = TRUE)

# Scatterplots with outliers
library(ggplot2)
Gr.a <- ggplot(data=datOutlier) + geom_point(mapping = aes(x = Age, y=Height))
Gr.a <- Gr.a +
  geom_point(data=datOutlier[rankAge_Height[1:top_nHeight],], 
             mapping = aes(x=Age,y=Height), shape=19, color="red", size=2) + 
             xlab("Age") + ylab("Height")
Gr.a

Gr.b <- ggplot(data=datOutlier) + geom_point(mapping = aes(x=Age, y=Weight))
Gr.b <- Gr.b +
  geom_point(data=datOutlier[rankAge_Weight[1:top_nWeight],], 
             mapping = aes(x=Age,y=Weight), shape=19, color="red", size=2) +
             xlab("Age") + ylab("Weight")
Gr.b

library(gridExtra)
grid.arrange(Gr.a, Gr.b, nrow = 1, ncol = 2)

# Index of outliers - union of two and remove outliers from 
# datasets for classification
OutlierIndex <- union(rankAge_Height[1:top_nHeight], 
                      rankAge_Weight[1:top_nHeight])

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------









# ------------------------------------------------------------------------------
# HIERARCHICAL BASED CLUSTERING
# ------------------------------------------------------------------------------

# Sample 10,000 observations (distance matrix won't work on full dataset)
set.seed(0)
datHcluster <- dat %>% 
  dplyr::select(Age:Weight) %>% 
  sample_n(10000)

# Distance matrix and hierarchical cluster using complete linkage
DistMatr <- dist(datHcluster, method = "euclidean")
H_cluster <- hclust(DistMatr, method = "complete") 

# Show the dendrogram
H <- plot(H_cluster, main = "Complete Linkage", xlab = "", sub = "", 
          hang = -1, labels = FALSE, xaxp = c(0,3,150))

# Cluster of interest - the 50 points on the left of the dendrogram
Cluster_pts <- datHcluster[H_cluster$order[1:50], ]

# Scatterplots of features with cluster of interest overlaid
Gr.c <- ggplot(data=datHcluster) + 
            geom_point(mapping = aes(x=Weight, y=Height, alpha=1/100), 
                       show.legend = FALSE)
Gr.c <- Gr.c +
  geom_point(data=Cluster_pts[, 2:3], 
             mapping = aes(x=Weight,y=Height), shape=19, color="red")
Gr.c

Gr.d <- ggplot(data=datHcluster) + 
              geom_point(mapping = aes(x=Age, y=Height, alpha=1/100), 
                         show.legend = FALSE)

Gr.d <- Gr.d +
  geom_point(data=Cluster_pts[, 1:2], 
             mapping = aes(x=Age,y=Height), shape=19, color="red")
Gr.d

Gr.e <- ggplot(data=datHcluster) + 
             geom_point(mapping = aes(x=Age, y=Weight, alpha=1/100), 
                        show.legend = FALSE)
Gr.e <- Gr.e +
  geom_point(data=Cluster_pts[, c(1,3)], 
             mapping = aes(x=Age,y=Weight), shape=19, color="red")
Gr.e

grid.arrange(Gr.c, Gr.e, Gr.d, nrow = 2, ncol = 2)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------









# ------------------------------------------------------------------------------
# K NEAREST NEIGHBOURS CLASSIFICATION
# ------------------------------------------------------------------------------

# Remove 138 outliers known from Unsupervised Outlier Detection
datKNN <- dat

# Sample 1,000 observations - KNN not scalable to high number of observations
datKNN <- sample_n(dat, 1000)

# Create dummy variables for categorical data
library(fastDummies)
datKNN <- dummy_cols(datKNN)

# Remove original categorical features
datKNN <- dplyr::select(datKNN, -c(Sex, Season, NOC, Sport))

# KNN using ALL VARIABLES
library(caret)
library(klaR)
trControl <- trainControl(method  = "cv",
                          number  = 10)

KN_crossval <- train(x = datKNN[, -4], y = datKNN$Medal,
                     method     = "knn",
                     tuneGrid   = expand.grid(k = 1:10),
                     preProcess = c("center", "scale"),
                     trControl  = trControl,
                     metric     = "Accuracy")
KN_crossval
    # k-Nearest Neighbors 
    # 
    # 1000 samples
    # 303 predictor
    # 4 classes: 'No medal', 'Bronze', 'Silver', 'Gold' 
    # 
    # Pre-processing: centered (303), scaled (303) 
    # Resampling: Cross-Validated (10 fold) 
    # Summary of sample sizes: 900, 900, 900, 900, 900, 900, ... 
    # Resampling results across tuning parameters:
    #   
    #   k   Accuracy   Kappa       
    # 1  0.7560055   0.050101756
    # 2  0.7620154   0.067259662
    # 3  0.8329764   0.038403977
    # 4  0.8329667   0.024707802
    # 5  0.8450467  -0.002548230
    # 6  0.8570271   0.022786725
    # 7  0.8640071   0.023996356
    # 8  0.8630170   0.007746597
    # 9  0.8650172   0.004186361
    # 10  0.8660172   0.000000000
    # 
    # Accuracy was used to select the optimal model using the largest value.
    # The final value used for the model was k = 10.

confusionMatrix(KN_crossval)
    # Prediction No medal Bronze Silver Gold
    # No medal     86.6    4.5    4.8  4.1
    # Bronze        0.0    0.0    0.0  0.0
    # Silver        0.0    0.0    0.0  0.0
    # Gold          0.0    0.0    0.0  0.0
    # 
    # Accuracy (average) : 0.866


# KNN repeated 10 times with ALL VARIABLES
KN_sample_accuracy <- rep(0, 10)
KN_true_positive <- rep(0, 10)

for (i in 1:10){
  datKNN <- sample_n(dat, 1000)
  datKNN <- dummy_cols(datKNN)
  datKNN <- dplyr::select(datKNN, -c(Sex, Season, NOC, Sport))
  KN_crossval2 <- train(Medal ~ .,
                       method     = "knn",
                       tuneGrid   = expand.grid(k = 10),
                       preProcess = c("center", "scale"),
                       trControl  = trControl,
                       metric     = "Accuracy",
                       data       = datKNN)
  KN_sample_accuracy[i] <- KN_crossval2$results$Accuracy
  KN_true_positive[i] <- sum(diag(confusionMatrix(KN_crossval)$table[-1, -1]))
}

# Average accuracy over 10 resamples
mean(KN_sample_accuracy)
    # 0.8511521
sd(KN_sample_accuracy)
    # 0.01123035

# True positve rates over 10 resamples
sum(KN_true_positive)
  # 0

# Repeat the KNN for NUMERIC VARIABLES ONLY
KN_crossval <- train(x = datKNN[, 2:4], y = datKNN$Medal,
                     method     = "knn",
                     tuneGrid   = expand.grid(k = 1:10),
                     preProcess = c("center", "scale"),
                     trControl  = trControl,
                     metric     = "Accuracy")
KN_crossval
    # k-Nearest Neighbors 
    # 
    # 1000 samples
    # 3 predictor
    # 4 classes: 'No medal', 'Bronze', 'Silver', 'Gold' 
    # 
    # Pre-processing: centered (3), scaled (3) 
    # Resampling: Cross-Validated (10 fold) 
    # Summary of sample sizes: 900, 899, 899, 900, 901, 900, ... 
    # Resampling results across tuning parameters:
    #   
    #   k   Accuracy   Kappa       
    # 1  0.7459793   0.015367495
    # 2  0.7659710   0.021959892
    # 3  0.8309627   0.069525806
    # 4  0.8379934   0.018293653
    # 5  0.8520241   0.031672708
    # 6  0.8520041   0.011019175
    # 7  0.8520041   0.006256155
    # 8  0.8540041  -0.002172866
    # 9  0.8550142  -0.006333223
    # 10  0.8590243   0.004347826
    # 
    # Accuracy was used to select the optimal model using the largest value.
    # The final value used for the model was k = 10.

confusionMatrix(KN_crossval)
    # Prediction No medal Bronze Silver Gold
    # No medal     85.9    5.7    4.8  3.4
    # Bronze        0.1    0.0    0.1  0.0
    # Silver        0.0    0.0    0.0  0.0
    # Gold          0.0    0.0    0.0  0.0

# Repeat the resample for NUMERIC VARIABLES only
for (i in 1:10){
  datKNN <- sample_n(dat, 1000)
  KN_crossval2 <- train(x = datKNN[, 2:4], y = datKNN$Medal,
                        method     = "knn",
                        tuneGrid   = expand.grid(k = 10),
                        preProcess = c("center", "scale"),
                        trControl  = trControl,
                        metric     = "Accuracy")
  
  KN_sample_accuracy[i] <- KN_crossval2$results$Accuracy
  KN_true_positive[i] <- sum(diag(confusionMatrix(KN_crossval)$table[-1, -1]))
}

# Average accuracy over 10 resamples
mean(KN_sample_accuracy)
    # 0.8511421
sd(KN_sample_accuracy)
    # 0.01223035

# True positve rates over 10 resamples
sum(KN_true_positive)
    # 0

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------









# ------------------------------------------------------------------------------
# ANOVA
# ------------------------------------------------------------------------------

# Stratified sample
datAV <- dat %>%
  group_by(Medal) %>%
  sample_n(size=1000)

# Subs each class label
datAV_Gold <- filter(datAV, Medal == "Gold")
datAV_Silver <- filter(datAV, Medal == "Silver")
datAV_Bronze <- filter(datAV, Medal == "Bronze")
datAV_NoMedal <- filter(datAV, Medal == "No medal")

# QQ Plots of each class
par(mfrow = c(3,4))
par(mar = c(2,1.5,1.5,1))

qqnorm(datAV_Gold$Age, main = "Age - Gold")
qqline(datAV_Gold$Age)
qqnorm(datAV_Gold$Height, main = "Height - Gold")
qqline(datAV_Gold$Height)
qqnorm(datAV_Gold$Weight, main = "Weight - Gold")
qqline(datAV_Gold$Weight)

qqnorm(datAV_Silver$Age, main = "Age - Silver")
qqline(datAV_Silver$Age)
qqnorm(datAV_Silver$Height, main = "Height - Silver")
qqline(datAV_Silver$Height)
qqnorm(datAV_Silver$Weight, main = "Weight - Silver")
qqline(datAV_Silver$Weight)

qqnorm(datAV_Bronze$Age, main = "Age - Bronze")
qqline(datAV_Bronze$Age)
qqnorm(datAV_Bronze$Height, main = "Height - Bronze")
qqline(datAV_Bronze$Height)
qqnorm(datAV_Bronze$Weight, main = "Weight - Bronze")
qqline(datAV_Bronze$Weight)

qqnorm(datAV_NoMedal$Age, main = "Age - No Medal")
qqline(datAV_NoMedal$Age)
qqnorm(datAV_NoMedal$Height, main = "Height - No Medal")
qqline(datAV_NoMedal$Height)
qqnorm(datAV_NoMedal$Weight, main = "Weight - No Medal")
qqline(datAV_NoMedal$Weight)

# Check the standard deviations of class level samples
summarise(group_by(datAV, Medal), stdDev = sd(Age))
summarise(group_by(datAV, Medal), stdDev = sd(Height))
summarise(group_by(datAV, Medal), stdDev = sd(Weight))

# Anova tests
AOV_Age <- aov(Age ~ Medal, data = datAV)
summary(AOV_Age)
    #               Df Sum Sq Mean Sq F value Pr(>F)  
    # Medal          3    273   91.13   2.549  0.054 .
    # Residuals   3996 142846   35.75                 
    # ---
    #   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

AOV_Height <- aov(Height ~ Medal, data = datAV)
summary(AOV_Height)
    #               Df Sum Sq Mean Sq F value   Pr(>F)    
    # Medal          3   3471  1156.8   12.14 6.64e-08 ***
    #   Residuals   3996 380908    95.3    

AOV_Weight <- aov(Weight ~ Medal, data = datAV)                            
summary(AOV_Weight)
    #               Df Sum Sq Mean Sq F value   Pr(>F)    
    # Medal          3   7094  2364.5   13.25 1.32e-08 ***
    #   Residuals   3996 713112   178.5   

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------









# ------------------------------------------------------------------------------
# NAIVE BAYES CLASSIFIER
# ------------------------------------------------------------------------------

# Remove outliers detected from Unsupervised Outlier Detection
datNB <- dat[-OutlierIndex, ]

# Naive Bayes Classifier USING CATEGORICAL FEATURES
library(caret)
trControl <- trainControl(method = "cv", number = 10)

NB_crossval <- train(x = datNB[, -c(2:4)], y = datNB$Medal, 
                     method = "nb", 
                     trControl = trControl,
                     tuneGrid = data.frame(fL = 1, 
                                           usekernel = FALSE, adjust = 0))
NB_crossval
    # Naive Bayes 
    # 
    # 270899 samples
    # 5 predictor
    # 4 classes: 'No medal', 'Bronze', 'Silver', 'Gold' 
    # 
    # No pre-processing
    # Resampling: Cross-Validated (10 fold) 
    # Summary of sample sizes: 243686, 243684, 243684, 243683, 243685, 243685, ... 
    # Resampling results:
    #   
    #   Accuracy  Kappa
    # 1         1    
    # 
    # Tuning parameter 'fL' was held constant at a value of 1
    # Tuning parameter 'usekernel'
    # was held constant at a value of FALSE
    # Tuning parameter 'adjust' was held constant at
    # a value of 0  

confusionMatrix(NB_crossval)
    # Cross-Validated (10 fold) Confusion Matrix 
    # 
    # (entries are percentual average cell counts across resamples)
    # 
    # Reference
    # Prediction No medal Bronze Silver Gold
    # No medal     85.4    0.0    0.0  0.0
    # Bronze        0.0    4.9    0.0  0.0
    # Silver        0.0    0.0    4.8  0.0
    # Gold          0.0    0.0    0.0  4.9
    # 
    # Accuracy (average) : 1

# Naive Bayes Classifier USING ALL FEATURES
NB_crossval <- train(x = datNB[, -8], y = datNB$Medal, method = "nb", 
                     trControl = trainControl(method = "cv", number = 10),
                     tuneGrid = data.frame(fL = 1, usekernel = FALSE, adjust = 0))
NB_crossval
    # Naive Bayes 
    # 
    # 270899 samples
    # 7 predictor
    # 4 classes: 'No medal', 'Bronze', 'Silver', 'Gold' 
    # 
    # No pre-processing
    # Resampling: Cross-Validated (10 fold) 
    # Summary of sample sizes: 243685, 243685, 243685, 243683, 243686, 243684, ... 
    # Resampling results:
    #   
    #   Accuracy   Kappa     
    # 0.8500486  0.04755011
    # 
    # Tuning parameter 'fL' was held constant at a value of 1
    # Tuning parameter 'usekernel' was held constant at
    # a value of FALSE
    # Tuning parameter 'adjust' was held constant at a value of 0

confusionMatrix(NB_crossval)
    # Prediction No medal Bronze Silver Gold
    # No medal     84.7    4.7    4.6  4.6
    # Bronze        0.1    0.0    0.0  0.0
    # Silver        0.0    0.0    0.0  0.0
    # Gold          0.6    0.1    0.2  0.3

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------




