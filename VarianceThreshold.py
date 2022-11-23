import pandas as pd
# Make DataFrame of the given data 
data = pd.DataFrame({"A":[1,2,4,1,2,4],
"B":[4,5,6,7,8,9],
"C":[-0.2,0,0,0,0,0],
"D":[1,1,1,1,1,1]})


data = pd.read_csv("train.csv", nrows = 1000)
print(data.head())   # using head we can define how many rows we need
X = data.drop(labels= "TARGET", axis = 1)
y = data["TARGET"]


from sklearn.model_selection import train_test_split
# random_state is same as random.seed(0)
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.3, random_state= 0)


from sklearn.feature_selection import VarianceThreshold
var = VarianceThreshold(threshold = 0)  # making an object of the class 
var.fit(data)
print(sum(var.get_support())) # it gives us the total no of true values 
x = data.columns[var.get_support()]  # returns true false for the columns included or not 
constant_columns = [column for column in data.columns if column not in x]
data.drop(constant_columns, axis = 1, inplace = True)  # inplace changes the value in the original dataframe rather than returning
print(data)

data.to_csv("train.csv" , index = False)


