#Import required Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
df = pd.read_csv('AusApparalSales4thQrt2020.csv')

#Data Wrangling Section
print("\nMissing values: ", df.isna().sum())

#Fill missing values (forward fill method)
df.fillna(method='ffill', inplace=True)

#Normalize numerical columns
df[['Sales', 'Unit']] = (df[['Sales', 'Unit']] - df[['Sales', 'Unit']].min()) / (df[['Sales', 'Unit']].max() - df[['Sales', 'Unit']].min())

#Grouping for analysis
state_sales = df.groupby('State')['Sales'].sum().sort_values(ascending=False)
group_sales = df.groupby('Group')['Sales'].sum().sort_values(ascending=False)

#Descriptive Statistics
print("\nDescriptive Statistics:")
print("\nSales Mean:", np.mean(df['Sales']))
print("\nSales Median:", np.median(df['Sales']))
print("\nSales Mode:", df['Sales'].mode().values[0])
print("\nSales Std Dev:", np.std(df['Sales']))

#Stats
print("\nState with the Highest Sales:", state_sales.idxmax(), "—", state_sales.max())
print("\nState with the Lowest Sales:", state_sales.idxmin(), "—", state_sales.min())
print("\nTop Demographic:", group_sales.idxmax(), "—", group_sales.max())
print("\nLowest Demographic:", group_sales.idxmin(), "—", group_sales.min())

#Convert Date column to datetime for time analysis
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

#Time Series Reports
weekly_report = df.resample('W').sum()
monthly_report = df.resample('M').sum()
quarterly_report = df.resample('Q').sum()

#Visualization Section

#Box plot for descriptive stats
plt.figure(figsize=(8,6))
sns.boxplot(x='Group', y='Sales', data=df)
plt.title("Sales Distribution by Demographic Group")
plt.tight_layout()
plt.show()

#Distribution plot
plt.figure(figsize=(8,6))
sns.histplot(df['Sales'], kde=True)
plt.title("Sales Distribution Across All Groups")
plt.tight_layout()
plt.show()

#State-wise sales by group
plt.figure(figsize=(12,6))
sns.barplot(x='State', y='Sales', hue='Group', data=df)
plt.xticks(rotation=45)
plt.title("State-wise Sales Performance by Group")
plt.tight_layout()
plt.show()

#Time-of-day analysis
df['Hour'] = df.index.hour
plt.figure(figsize=(10,5))
sns.countplot(x='Hour', data=df)
plt.title("Sales Volume by Hour of Day")
plt.tight_layout()
plt.show()