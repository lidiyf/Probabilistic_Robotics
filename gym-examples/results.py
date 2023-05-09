import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

f = 'mon/0.monitor.csv'
#f = 'num_steps.csv'
#f2 = 'num_steps_bl.csv'
df = pd.read_csv(f, skiprows=1)
#df2 = pd.read_csv(f2) #, skiprows=1)
#df = pd.read_csv('reward.csv')
#print(df)

x = np.array(list(range(len(df['r']))))
y = np.array(df['r'])
#plt.scatter(x, y)
plt.title("Reward Function")
plt.plot(df['r'])
plt.xlabel("episode")
plt.ylabel("reward")
a, b = np.polyfit(x, y, 1)
plt.plot(x, a*x+b)  
#plt.plot(df. iloc[:,0])
#plt.plot(df2. iloc[:,0])
#plt.plot(df. iloc[:,0])
#plt.ylim(-500, 500)
#plt.scatter(df['t'], df['r'])
#plt.savefig("a2cr")
plt.show()
