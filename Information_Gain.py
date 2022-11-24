
# Mutual information measures the amount of information one can obtain from one variable given
# other i.e. dependency between two variables

import pandas as pd
from sklearn.datasets import load_wine
data = load_wine()
print(data.DESCR)
print(data.keys())   


df = pd.DataFrame(data.data, columns = data.feature_names)
df["wine"] = data.target   # i can use any name for the target column


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(labels = ["wine"], axis = 1), df["wine"], test_size=0.3, random_state=0)


# DETERMINE THE MUTUAL INFORMATION
from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X_train, y_train)
print(mutual_info)

# MAKING A SERIES DATATYPE 
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
print(mutual_info.sort_values(ascending = False, inplace = True))



# PLOT THE BAR GRAPH FOR TH MUTUAL INFO
# import matplotlib.pyplot as plt
# g = mutual_info.plot.bar(figsize = (20,8))
# plt.show()


# SELECTING K BEST FEATURES
from sklearn.feature_selection import SelectKBest
sel_best = SelectKBest(mutual_info_classif, k = 5)  # making object with the selection criteria 
sel_best.fit(X_train.fillna(0), y_train)
x = X_train.columns[sel_best.get_support()]

columns_remove = [column for column in X_train.columns if column not in x]
X_train.drop(columns_remove, axis = 1, inplace = True)
X_test.drop(columns_remove, axis = 1, inplace = True)

