myseed = 2018

#-------------------------------
# Step 1: Load database
#-------------------------------
df_train <- read.csv(file = "cleaned_data/train_corr.csv")
df_test  <- read.csv(file = "cleaned_data/test_corr.csv")
head(df_train)
colnames(df_train)
str(df_train)

# Training set
dim(df_train) # 27703    27
table(df_train$class)  # [class 0]:26537  [class 1]:1166 

# Testing set
dim(df_test)  # 15702    27
table(df_test$class)   # [class 0]:14777  [class 1]:925 

# Change datatype
df_train$class <- as.factor(df_train$class)
df_test$class  <- as.factor(df_test$class)



#--------------------------------------------
# Step 2: Build model with undersampling
#--------------------------------------------
library(randomForest)

samp_size = nrow(df_train[df_train$class==1,])
samp_size # 1166

# Tuning
set.seed(myseed)
model.rf <- randomForest(class~.,  data = df_train, ntree=150, sampsize=c(samp_size,samp_size), set.seed(myseed)) # try with different num of ntree
model.rf
plot(model.rf)  # Seems like 50 trees is good

# Variable Importance Plot
varImpPlot(model.rf, sort = T, main="Variable Importance")

# Change model to 50 trees
model.rf <- randomForest(class~.,  data = df_train, ntree=50, sampsize=c(samp_size,samp_size), set.seed(myseed))


#----------------------------------
# Step 3: Predict and Evaluation
#----------------------------------
set.seed(myseed)
result_predicted <- predict(model.rf, newdata = df_test)
result_confusionmatrix = table(result_predicted, df_test$class)
result_confusionmatrix
#  result_predicted     0     1
#                 0 13247   284
#                 1  1530   641


cm <- confusionMatrix(result_confusionmatrix)  
cm
cm$overall[1] # Accuracy: 0.8844733 


#----------------------
# Step 4: Evaluation
#----------------------
library(caret)
recall(result_confusionmatrix)       # Recall: 0.8964607
precision(result_confusionmatrix)    # Precision: 0.9790112
specificity(result_confusionmatrix)  # Specificity: 0.692973
F_meas(result_confusionmatrix)       # F_measure: 0.9359192


#-------------------------------------------------------------
# Step 5: Tuning for mtry after fixing ntree in Random Forest
#         mtry: Number of variables randomly sampled as
#               candidates at each split
#-------------------------------------------------------------
set.seed(myseed)
bestmtry <- tuneRF(x=df_train[,-ncol(df_train)], y=df_train$class, stepFactor=1.5, improve=1e-5, ntree=50)
print(bestmtry)   # chosen mtry=5


#--------------------------
# Step 6: Rebuild model 
#--------------------------
model.rf <- randomForest(class~., data = df_train, ntree=50, sampsize=c(samp_size,samp_size), mtry=5, set.seed(myseed)) 
model.rf

set.seed(myseed)
result_predicted <- predict(model.rf, newdata = df_test)
result_confusionmatrix = table(result_predicted, df_test$class)
result_confusionmatrix
# result_predicted     0     1
#                0 13247   284
#                1  1530   641

recall(result_confusionmatrix)       # Recall: 0.8964607
precision(result_confusionmatrix)    # Precision: 0.9790112
specificity(result_confusionmatrix)  # Specificity: 0.692973
F_meas(result_confusionmatrix)       # F_measure: 0.9359192


#----------------------------------------------------------------------
# Step 7: Conclusion
# Since the target of this assignment is that among those companies
# that become bankrupt, how many we can predict correctly. Therefore,
# we have chosen our performance metrics to be specificity.
#
# We acheived a Specificity of 0.69. We can try improving the 
# result with other algorithms.
#-----------------------------------------------------------------------
