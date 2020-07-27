#Loading libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Importing the data from GitHub
url_test = 'https://raw.githubusercontent.com/davidblumenstiel/Kaggle/master/titanic/test.csv'
url_train = 'https://raw.githubusercontent.com/davidblumenstiel/Kaggle/master/titanic/train.csv'

test = pd.read_csv(url_test)
train = pd.read_csv(url_train)

#Adding lables to differentiate the data later
test['label'] = "test"
train['label'] = "train"

# Exploratory data analysis
print(test.shape)    #Test is similar to train but without a survived category
print(train.head())
print(train.shape)
print(train.columns)
print(train.info)
print(test.describe())



## Random Forest model (inspired by the tutorial)


survival = train["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch", "Cabin", 'label', 'Age','Ticket','Fare']

#Need to process the data as one dataset to get the same number of dummy variables
df_combined = pd.concat([train, test]) 
df_combined_processed = pd.get_dummies(df_combined[features])
df_combined_processed = df_combined_processed.fillna(-999)   #Sets NaN values to -999 to avoid errors

#Splits the data again into train and test sets, and removes the lables
predictors_train = df_combined_processed[df_combined_processed["label_train"] == 1]
predictors_train = predictors_train.drop(columns = ["label_train","label_test"])
predictors_test = df_combined_processed[df_combined_processed["label_test"] == 1]
predictors_test = predictors_test.drop(columns = ["label_train","label_test"])


#Makes the model, and uses it to generate survival predictions
model = RandomForestClassifier(n_estimators = 1500, random_state=1) 
model.fit(predictors_train, survival) #Trains the model with training data

predictions = model.predict(predictors_test) #Makes predictions on the test data with the model

#Note: can't verify the model without survival information on the test set

#Saves the predictions as a csv for submission (copied from the tutorial)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('predictions.csv', index=False)
print("Your submission was successfully saved!")