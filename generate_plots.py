import pandas as pd
import matplotlib.pyplot as plt

# Read the string data into pandas DataFrames
df_pruned = pd.read_csv("pruned_results.csv")
df_not_pruned = pd.read_csv("dense_results.csv")

# --- Data Preparation ---
df_pruned.sort_values(by=['client', 'round'], inplace=True)
df_not_pruned.sort_values(by=['client', 'round'], inplace=True)
df_pruned.reset_index(drop=True, inplace=True)
df_not_pruned.reset_index(drop=True, inplace=True)

df_compare = pd.DataFrame({
    'client': df_pruned['client'],
    'round': df_pruned['round'],
    'acc_pruned': df_pruned['acc'],
    'acc_not_pruned': df_not_pruned['acc'],
    'time_pruned': df_pruned['epoch_time'],
    'time_not_pruned': df_not_pruned['epoch_time']
})

df_compare['speedup'] = df_compare['time_not_pruned'] / df_compare['time_pruned']


df_c1 = df_compare[df_compare['client'] == 1].set_index('round')
df_c2 = df_compare[df_compare['client'] == 2].set_index('round')

# create df for mean speedup (sum of client 1 and client 2 speedups divided by 2)
mean_speedup = pd.DataFrame({
    'client': df_c1.index,
    'speedup': (df_c1['speedup'] + df_c2['speedup']) / 2
})

print("Mean Speedup for all rounds:", mean_speedup.sum().speedup / len(mean_speedup))

# --- Plot 1: Accuracy Comparison ---
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(df_c1.index, df_c1['acc_pruned'], 'o-', label='Cliente 1 (Podado)')
ax1.plot(df_c1.index, df_c1['acc_not_pruned'], 'o--', label='Cliente 1 (Não Podado)')
ax1.plot(df_c2.index, df_c2['acc_pruned'], 's-', label='Cliente 2 (Podado)')
ax1.plot(df_c2.index, df_c2['acc_not_pruned'], 's--', label='Cliente 2 (Não Podado)')

ax1.set_title('Acurácia por round (Podado vs. Não-Podado)', fontsize=16)
ax1.set_ylabel('Acurácia (%)')
ax1.set_xlabel('Round')
ax1.legend()
ax1.grid(True, linestyle=':')
ax1.set_xticks([1, 2, 3])
plt.tight_layout()

# Save the accuracy plot
plt.savefig('accuracy_comparison.png', dpi=300)
plt.show()
print("Saved 'accuracy_comparison.png'")

# --- Plot 2: Speedup Comparison ---
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(df_c1.index, df_c1['speedup'], 'o-', label='Cliente 1 Speedup')
ax2.plot(df_c2.index, df_c2['speedup'], 's-', label='Cliente 2 Speedup')
ax2.plot(mean_speedup['client'], mean_speedup['speedup'], 'd--', label='Speedup Médio', color='green')

ax2.axhline(1.0, color='r', linestyle=':', label='Baseline (No Speedup)')

ax2.set_title('Speedup (Tempo(Não Podado) / Tempo(Podado))', fontsize=16)
ax2.set_ylabel('Fator de Speedup (> 1 significa que Podado é mais rápido)')
ax2.set_xlabel('Round')
ax2.legend()
ax2.grid(True, linestyle=':')
ax2.set_xticks([1, 2, 3])
plt.tight_layout()

# Save the speedup plot
plt.savefig('speedup_comparison.png', dpi=300)
plt.show()
print("Saved 'speedup_comparison.png'")
