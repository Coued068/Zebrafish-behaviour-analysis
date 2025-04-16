import pandas as pd 
from scipy.stats import kruskal
import matplotlib.pyplot as plt
from pathlib import Path
import scikit_posthocs as sp

# CSV paths here for one age group 
csv_paths = [
    r"C:\Users\Laris\OneDrive\Desktop\New_csvfiles_newdata\XE991\XE-991_3DPF_JAN_13_2025.csv",
    r"C:\Users\Laris\OneDrive\Desktop\New_csvfiles_newdata\TCB-2_2ndcon\TCB-2_3DPF_2ndcon-27_01_2025.csv",
    r"C:\Users\Laris\OneDrive\Desktop\New_csvfiles_newdata\TCB-2_1stcon\TCB-2_3DPF_1STCON.csv",
    r"C:\Users\Laris\OneDrive\Desktop\New_csvfiles_newdata\Muscarine_2nd_con\Muscarine_3dpf_2nd_con.csv",
    r"C:\Users\Laris\OneDrive\Desktop\New_csvfiles_newdata\Muscarine_1stcon\Muscarine_3dpf_1st_con.csv",
    r"C:\Users\Laris\OneDrive\Desktop\New_csvfiles_newdata\ICA_069673\ICA-069673_3DPF_20_01_2025.csv",
    r"C:\Users\Laris\OneDrive\Desktop\New_csvfiles_newdata\Dopamine_2ndcon\Dopamine_3DPF_dopamine_2ndcon.csv",
    r"C:\Users\Laris\OneDrive\Desktop\New_csvfiles_newdata\Dopamine_1stcon\Dopamine_3DPF_1STCON_temp.vtr 3dpf 21 oct 2024 dopamine 1st con.csv",
    r"C:\Users\Laris\OneDrive\Desktop\New_csvfiles_newdata\8-OH-DPAT\8-OH-DPAT_3DPF_Carole.csv",
    r"C:\Users\Laris\OneDrive\Desktop\New_csvfiles_newdata\5-HT_2ndcon\5-HT_3DPF_2NDCON_300S.csv",
    r"C:\Users\Laris\OneDrive\Desktop\New_csvfiles_newdata\5-HT_1stcon\5-HT_3DPF_1STCON_300S.csv",
]

age_label = "3dpf"
metrics = ['smldur', 'lardur', 'smldist', 'lardist']
control_data = []

for path in csv_paths:
    df = pd.read_csv(path)
    df = df.head(193)
    
    summed_blocks = []
    for block_start in [0, 48, 96, 144]:
        block = df.iloc[block_start:block_start+24].copy()
        block = block[block['animal'].isin([f'WT{i:03}' for i in range(1, 13)])]
        summed_blocks.append(block.groupby('animal')[metrics].sum())

    summed_total = summed_blocks[0].copy()
    for block in summed_blocks[1:]:
        summed_total += block

    summed_total = summed_total.reset_index()
    summed_total['Group'] = Path(path).stem
    control_data.append(summed_total)

combined = pd.concat(control_data, ignore_index=True)

# Kruskal-Wallis test
results = []
for metric in metrics:
    data_by_group = [group[metric].dropna().values for _, group in combined.groupby('Group')]
    stat, p = kruskal(*data_by_group)
    results.append({
        'Metric': metric,
        'H-statistic': round(stat, 4),
        'p-value': '%.3e' % p,
        'Significant (p<0.05)': 'Yes' if p < 0.05 else 'No'
    })

results_df = pd.DataFrame(results)
results_df.to_csv(f"kruskal_results_controls_only_{age_label}.csv", index=False)
print(f"Kruskal-Wallis analysis complete for {age_label}.")

# Post hoc Dunn (non-parametric, multiple comparisons)
for metric in metrics:
    print(f"\n--- DUNN POST HOC: {metric.upper()} ---")
    dunn = sp.posthoc_dunn(combined, val_col=metric, group_col='Group', p_adjust='holm')
    print(dunn)
    dunn.to_csv(f"dunn_posthoc_{metric}_{age_label}.csv")

print("Dunn post hoc tests completed and saved.")
