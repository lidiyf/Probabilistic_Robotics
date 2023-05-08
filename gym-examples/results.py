import matplotlib.pyplot as plt
import pandas as pd

#df = pd.read_csv('result.monitor.csv', skiprows=1)
df = pd.read_csv('follow.csv')
#print(df)

#plt.plot(df['r'])
plt.plot(df. iloc[:,0])
#plt.ylim(-500, 1000)
#plt.scatter(df['t'], df['r'])
plt.show()
