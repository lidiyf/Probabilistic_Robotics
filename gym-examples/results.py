import matplotlib.pyplot as plt
import pandas as pd

#f = 'result.monitor.csv'
f = 'mon/9.monitor.csv'
df = pd.read_csv(f, skiprows=1)
#df = pd.read_csv('reward.csv')
#print(df)

plt.plot(df['r'])
#plt.plot(df. iloc[:,0])
#lt.ylim(0, 500)
#plt.scatter(df['t'], df['r'])
#plt.savefig("a2cr")
plt.show()
