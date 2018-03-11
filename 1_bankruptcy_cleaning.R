#----------------------------------------------------------------------------
# Data Source:
#    https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data
#----------------------------------------------------------------------------
myseed = 2018

#-----------------------------------------------------
# Step 1: Loading of database
#-----------------------------------------------------
df_1year <- read.csv("data/1year.csv", header=TRUE, na.strings="?")
df_2year <- read.csv("data/2year.csv", header=TRUE, na.strings="?")
df_3year <- read.csv("data/3year.csv", header=TRUE, na.strings="?")
df_4year <- read.csv("data/4year.csv", header=TRUE, na.strings="?")
df_5year <- read.csv("data/5year.csv", header=TRUE, na.strings="?")

totalrows_train = nrow(df_1year) + nrow(df_2year) + nrow(df_3year)
totalrows_train   # 27703

totalrows_test = nrow(df_4year) + nrow(df_5year)
totalrows_test    # 15702


#------------------------------------
# Step 2: Explore data 
#       : Change data type
#------------------------------------
df_combined <- rbind(df_1year, df_2year, df_3year, df_4year, df_5year) 
colnames(df_combined) # Attr1 - Attr64, class
dim(df_combined)   # 43405 66
str(df_combined)

df_combined$class <- as.factor(df_combined$class)


#-----------------------------------------------------------------------------
# Step 3: Checking total missing data for each column and total missing rows
#-----------------------------------------------------------------------------
apply(df_combined, 2, function(x) { sum(is.na(x)) })

cat("Row with no missing data:", sum(complete.cases(df_combined)))   # 19967 rows with no missing data
cat("Row with missing data: ", sum(!complete.cases(df_combined)))    # 23438 rows with missing data

# install.packages("VIM")
library("VIM")
aggr(df_combined)  # shows percentage of missing values in each column
# Note: Can investigate removing Attr21, Attr27, Attr37, Attr45, Attr60
#       due to many missing values in columns
#
# After consulting Subject Matter Expert, derive the following conclusion: 
# - Attr21: Changed missing value to 0. This refers to previous year's sales, because may be a new company.
# - Attr27: Changed missing value to 0, This shows how much return incurred through interest and other misc expenses. Some companies might not have loans from other institutions.
# - Attr37: Remove, because too much missing values. 
# - Attr45: Remove, because it is same as Attr60. Attr60 is more generalized form, as it does not includes Operating expenses.
# - Attr60: Don't remove, because this shows how companies managed to convert products to sales.

# Replace missing values from Attr21 and Attr27 with 0
df_combined$Attr21[is.na(df_combined$Attr21)] <- 0
df_combined$Attr27[is.na(df_combined$Attr27)] <- 0

# Remove column Attr37 and Attr45
df_combined <- subset(df_combined, select=-c(Attr37,Attr45))
colnames(df_combined)


#-----------------------------------------------------------
# Step 4: Check correlation table to see if some columns
#         can be removed
#-----------------------------------------------------------
df_cor <- cor(df_combined[1:(ncol(df_combined)-1)], use = "complete.obs")
df_cor
#install.packages("corrplot")
library(corrplot)
corrplot(df_cor, type="upper")

#install.packages("caret")
library(caret)
hc <- findCorrelation(df_cor, cutoff=0.8, verbose = T)
hc = sort(hc)
hc
# - Due to high correlation, these <<columns>> are recommended to be removed:
#   2  3  4  7  9 10 11 12 13 14 16 17 19 20 22 23 25 30 31 33 34 36 37
#   42 43 44 45 46 47 50 51 52 54 56 61 62
#
# - According to Subject Matter Expert, these are the <<attribute number>> to be removed:
#   4 9 11 12 13 14 22 24 25 26 28 31 33 35 39 40 42 48 49 50 51 53 54 56 64
# 
# - We will clean up the dataset using these 2 different ways


#-----------------------------------------------------------
# Step 5: Remove some columns using
#         - correlation
#         - subject matter expert view
# Note: PCA is not allowed in this assignment.
#-----------------------------------------------------------
# Case 1: Remove columns based on high correlation
df_reduced_using_cor <- df_combined[, -hc]
colnames(df_reduced_using_cor)

# Case 2: Remove columns based on subject matter expert view
df_reduced_using_expert <- subset(df_combined, select=-c(Attr4 , Attr9 , Attr11, Attr12, Attr13, Attr14, Attr22,
                                                         Attr24, Attr25, Attr26, Attr28, Attr31, Attr33, Attr35,
                                                         Attr39, Attr40, Attr42, Attr48, Attr49, Attr50, Attr51,
                                                         Attr53, Attr54, Attr56, Attr64))
colnames(df_reduced_using_expert)


#--------------------------------------------------
# Step 6: Impute missing data, then write to file.
#       : Note that it will take some time to run
#--------------------------------------------------
# Case 1
set.seed(myseed)
df_imputed_using_cor <- kNN(df_reduced_using_cor, imp_var = FALSE)
df_train_cor <- df_imputed_using_cor[1:totalrows_train, ]
df_test_cor  <- df_imputed_using_cor[(totalrows_train+1):nrow(df_imputed_using_cor) , ]

write.csv(x = df_train_cor, file = "cleaned_data/train_corr.csv", row.names = FALSE)
write.csv(x = df_test_cor,  file = "cleaned_data/test_corr.csv" , row.names = FALSE)

# Case 2
set.seed(myseed)
df_imputed_using_expert <- kNN(df_reduced_using_cor, imp_var = FALSE)
df_train_expert <- df_imputed_using_expert[1:totalrows_train, ]
df_test_expert  <- df_imputed_using_expert[(totalrows_train+1):nrow(df_imputed_using_expert) , ]

write.csv(x = df_train_expert, file = "cleaned_data/train_expert.csv", row.names = FALSE)
write.csv(x = df_test_expert, file = "cleaned_data/test_expert.csv", row.names = FALSE)
