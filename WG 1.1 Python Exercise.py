#importeren
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#inlezen van de dataset
df = pd.read_csv("Datasets.csv")
print(df)

#aantal dataset
aantal_datasets = df["dataset"].count()
print(aantal_datasets)

#print names van datasets
names_dataset = set(df["dataset"])
print(names_dataset)

#statistieken per groep (count, mean, variance, std dev)
count_per_groep = df.groupby("dataset").count()
mean_per_groep = df.groupby("dataset").mean()
variance_per_groep = df.groupby("dataset").var()
std_dev_per_groep = df.groupby("dataset").std()
print("count_per_groep:")
print(count_per_groep)
print("mean_per_groep:")    
print(mean_per_groep)
print("variance_per_groep:")
print(variance_per_groep)
print("std_dev_per_groep:")
print(std_dev_per_groep)

#my observations: count per group is allemaal gelijk, mean en var en std zit heeel dicht bij elkaar

# Violin plot x coordinaten
sns.violinplot(x="dataset", y="x", data=df)

plt.title("Violin plot of x-coordinates per dataset")
plt.show()

# Violin plot y coordinaten
sns.violinplot(x="dataset", y="y", data=df)

plt.title("Violin plot of y-coordinates per dataset")
plt.show()

# print correlaties
correlations = df.groupby("dataset")[["x", "y"]].corr()
print(correlations)

#covariantie matrix for each dataset
covariance_matrices = df.groupby("dataset")[["x", "y"]].cov()
print(covariance_matrices)

#lineaire regressie voor elke dataset

for dataset in names_dataset:
    subset = df[df["dataset"] == dataset]
    x = subset[["x"]]
    y = subset["y"]  

    model = LinearRegression()
    model.fit(x, y)
    
    print(f"Linear regression for {dataset}:")
    print(f"Slope: {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")
    print(f"R^2 score: {model.score(x, y)}")

    #create scatter plot with regression line
    #met facet grid maak je een aparte plot voor elke waarde in kolom "datasets"
    g = sns.FacetGrid(df, col="dataset", col_wrap=2)

    # met map_dataframe kun je een functie toepassen op elke subset van de data die overeenkomt met de facet grid
    g.map_dataframe(sns.scatterplot, x="x", y="y")

    g.set_axis_labels("X values", "Y values")
    g.set_titles("Dataset {col_name}")  

    plt.show()

    #scatter plot with regression line met Implot
    sns.lmplot(
    data=df,
    x="x",
    y="y",
    col="dataset",
    col_wrap=2
    )

    plt.show()