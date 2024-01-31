import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



c =['Sepal length','Sepal width','Petal length','Petal width','Class_labels']
df=pd.read_csv('iris.data',names=c)
df.head(150)



from pandas.api.types import is_numeric_dtype

for col in df.columns:
    if is_numeric_dtype(df[col]):
        print('%s:' % (col))
        print('\t Mean = %.2f' % df[col].mean())
        print('\t Standard deviation = %.2f' % df[col].std())
        print('\t Minimum = %.2f' % df[col].min())
        print('\t Maximum = %.2f' % df[col].max())





class_correlation = {
    "Iris-setosa - Iris-versicolor": df[df["Class_labels"].isin(["Iris-setosa", "Iris-versicolor"])][["Sepal length", "Petal length"]].corr().iloc[0, 1],
    "Iris-versicolor - Iris-virginica": df[df["Class_labels"].isin(["Iris-versicolor", "Iris-virginica"])][["Sepal length", "Petal length"]].corr().iloc[0, 1],
    "Iris-virginica - Iris-setosa": df[df["Class_labels"].isin(["Iris-virgina", "Iris-virginica"])][["Sepal length", "Petal length"]].corr().iloc[0, 1]
}

# Print class correlations
print("Class Correlations:")
for correlation, value in class_correlation.items():
    print(f"{correlation}:{value}")
