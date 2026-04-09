import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('raukf_sweep.csv')

RMSEs = [f'RMSE_{v}' for v in ['Je','Jee','Ji','Jii']]

df_piv = df.pivot(index='b',columns='sim_b',values=RMSEs)

for oRMSE in RMSEs:
    plt.figure()
    ax = plt.gca()
    sns.heatmap(df_piv[RMSE],ax=ax)
    ax.invert_yaxis()
