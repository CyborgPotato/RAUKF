import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('raukf_sweep.csv')

RMSEs = [f'RMSE_{v}' for v in ['Je','Jee','Ji','Jii']]

df_piv = df.pivot_table(index='b',columns=['justEI','Je_scale','sim_b'],values=RMSEs)

for oRMSE in RMSEs:
    plt.figure()
    ax = plt.gca()
    sns.heatmap(df_piv[RMSE][True][1],ax=ax)
    ax.invert_yaxis()
plt.show()


df_stim_piv = df.pivot_table(
    index='Je_scale',
    columns=['b','dbs_times','Ji_scale'],
    values=RMSEs
)


bs = df['b'].unique()
dbses = df['dbs_times'].unique()

for b in bs:
  f,axs = plt.subplots(2,2)
  axs = axs.ravel()
  i=0
  for dbs in dbses:
    for RMSE in RMSEs:
      if 'Jii' in RMSE or 'Jee' in RMSE:
        continue
      ax = axs[i]
      i+=1
      sns.heatmap(df_stim_piv[RMSE][b][dbs],ax=ax,cmap='coolwarm')
      ax.invert_yaxis()
      ax.set_title(str(b) + ' ' + RMSE + (' DBS' if len(dbs)==4 else ''))
plt.show()
