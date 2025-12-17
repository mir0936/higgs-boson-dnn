import pandas as pd
import matplotlib.pyplot as plt

# Just checking what the raw data looks like before building the DNN
df = pd.read_csv('training.csv')
print(df.head())
print(df.describe())

# Plot signal vs background ratio
df['Label'].value_counts().plot(kind='bar', color=['red', 'blue'])
plt.title('CERN Dataset: Signal vs Background Noise')
plt.savefig('data_exploration.png')
