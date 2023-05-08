import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('result.monitor.csv', skiprows=1)
#print(df)

plt.scatter(range(len(df['l'])), df['r'])
#plt.scatter(df['t'], df['r'])
plt.show()
