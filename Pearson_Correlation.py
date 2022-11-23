from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt


# LOAD BOSTON 
data = load_boston()
print(data.DESCR)  # it prints all the details about the dataset 

# CREATE DATAFRAME
df = pd.DataFrame(data.data, columns = data.feature_names)
df["MEDV"] = data.target
print(df.head())

# CREATE FEATURE MATRIX I.E. X AND Y VALUES 
X = df.drop("MEDV", axis = 1)  # if we change something in the original dataset then the values in the derived datasets also changes
y = df["MEDV"]

# SEPARATE DATASET INTO TEST AND TRAIN
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)


# USING PEARSON COEFFICIENT
cor  = X_train.corr()   # range is from -1 to 1

# MAKING A HEATMAP
# import seaborn as sns
# plt.figure(figsize=(12,10))  # creates a figure or activates an existing one
# sns.heatmap(cor, annot = True, cmap=plt.cm.Pastel1_r)  # annot = true shows the values inside the matrix boxes
# plt.show()


# FINDING THE COLUMNS TO REMOVE
def correlation( dataset , threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
corr_features = correlation(X_train, 0.7)
print(len(corr_features))

# DROPPING THE CORRELATION FEATURES
X_train.drop(corr_features, inplace = True, axis = 1)
X_test.drop(corr_features, inplace = True, axis = 1)
print(X_train.head())

