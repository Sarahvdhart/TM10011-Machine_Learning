#importeren
import pandas as pd

#inlezen van de dataset
df = pd.read_csv("Datasets.csv")
print(df)

#aantal dataset
aantal_datasets = df["dataset"].count()
print(aantal_datasets)

#print names van datasets
names_dataset = set(df["dataset"])
print(names_dataset)