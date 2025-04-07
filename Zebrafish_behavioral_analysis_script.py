import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from scipy import stats
import os

# Configuration globale
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Chemins directs vers les fichiers CSV
file_paths = {
    "3dpf": r"C:\Users\Laris\OneDrive\Desktop\New_csvfiles_newdata\TCB-2_2ndcon\TCB-2_3DPF_2ndcon-27_01_2025.csv",
    "4dpf": r"C:\Users\Laris\OneDrive\Desktop\New_csvfiles_newdata\TCB-2_2ndcon\TCB-2_4DPF_2ndcon_27_01_2025.csv",
    "5dpf": r"C:\Users\Laris\OneDrive\Desktop\New_csvfiles_newdata\TCB-2_2ndcon\TCB-2_5DPF_2ndcon_29_01_2025.csv"
}
columns = ["smldur", "smldist", "lardur", "lardist"]

# Extraction et structuration des données
def extract_totals(filepath):
    df = pd.read_csv(filepath, nrows=192)
    df = df.iloc[::2].sort_values('animal').reset_index(drop=True)
    totals = {"age": [], "mode": [], "group": [], "value": [], "parameter": [], "drugname": []}
    animals = sorted(df['animal'].unique())
    drugname = os.path.basename(filepath).split('_')[0]
    for col in columns:
        mode = 'Slow' if 'sml' in col else 'Fast'
        for i, animal in enumerate(animals):
            values = df[df['animal'] == animal][col].tolist()
            if not values:
                continue
            val = sum(values[:4])
            group = 'Control' if i < 12 else 'Drug'
            age_label = os.path.basename(filepath).split('_')[1].upper()
            totals["age"].append(age_label)
            totals["mode"].append(mode)
            totals["group"].append(group)
            totals["value"].append(val)
            totals["parameter"].append("Duration" if "dur" in col else "Distance")
            totals["drugname"].append(drugname)
    return pd.DataFrame(totals)

# Chargement global
def load_all_data(file_paths_dict):
    all_dfs = [extract_totals(path) for path in file_paths_dict.values()]
    return pd.concat(all_dfs, ignore_index=True)

# Couleurs personnalisées selon l'âge et le mode (uniquement pour Drug)
slow_swim_colors = {
    "3DPF": "#92B4F9",  # Light blue
    "4DPF": "#5A87F0",  # Mid blue
    "5DPF": "#1267E9"   # Dark blue
}
fast_swim_colors = {
    "3DPF": "#F9A3A9",  # Light red
    "4DPF": "#D94B55",  # Mid red
    "5DPF": "#E63946"   # Dark red
}

# Graphique structuré par jour pour poster
def plot_combined_blocked(df):
    drug_title = df["drugname"].iloc[0]
    output_prefix = f"grouped_comparison_blocked_{drug_title}"

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharey=False, frameon=False)
    parameters = ["Duration", "Distance"]
    age_order = ["3DPF", "4DPF", "5DPF"]
    mode_order = ["Slow", "Fast"]
    group_order = ["Control", "Drug"]

    age_markers = {
        "3DPF": 'o',
        "4DPF": '^',
        "5DPF": 's'
    }

    age_gap = 3.0
    mode_gap = 1.0
    group_gap = 0.4

        
    for i, param in enumerate(parameters):
        ax = axs[i]
        sub = df[df["parameter"] == param].copy()
        sub["age"] = pd.Categorical(sub["age"], categories=age_order, ordered=True)
        sub = sub.sort_values("age")

        if param == "Distance":
            sub["value"] = sub["value"] / 1000
            ax.set_yscale('log')
            ax.set_ylim(1e-1, sub["value"].max() * 1)  # Restore linear scale with space above

        tick_positions = []
        tick_labels = []
        tick_map = {}

        for age_idx, age in enumerate(age_order):
            base_pos = age_idx * age_gap
            for mode_idx, mode in enumerate(mode_order):
                for group_idx, group in enumerate(group_order):
                    values = sub[(sub["age"] == age) & (sub["mode"] == mode) & (sub["group"] == group)]["value"].tolist()
                    xpos = base_pos + mode_idx * mode_gap + group_idx * group_gap
                    tick_map[(age, mode, group)] = xpos

                    parts = ax.violinplot(values, positions=[xpos], widths=0.35, showmeans=True)

                    if group == "Control":
                        color = "#AAAAAA"
                    else:
                        color = slow_swim_colors.get(age) if mode == "Slow" else fast_swim_colors.get(age)

                    for pc in parts['bodies']:
                        pc.set_facecolor(color)
                        pc.set_edgecolor('black')
                        pc.set_alpha(1.0)
                        pc.set_linewidth(0.8)

                    for partname in ['cmeans', 'cbars', 'cmins', 'cmaxes']:
                        if partname in parts:
                            parts[partname].set_color('black')
                            parts[partname].set_linewidth(1.0)

                    ax.scatter([xpos]*len(values), values, alpha=0.6, color='black', s=30,
                               edgecolor='black', marker=age_markers.get(age, 'o'))

            center_pos = base_pos + 0.5 * ((len(mode_order) * mode_gap) + group_gap)
            tick_positions.append(center_pos)
            tick_labels.append(age.replace("DPF", " dpf"))
            

        unit = ' (s)' if param == 'Duration' else ' (m)'
        ax.set_ylabel(param + unit, fontsize=30, fontweight='bold')
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=30, fontweight='bold')
        ax.tick_params(axis='y', labelsize=14)

        for age in age_order:
            for mode in mode_order:
                vals1 = sub[(sub["age"] == age) & (sub["mode"] == mode) & (sub["group"] == "Control")]["value"].tolist()
                vals2 = sub[(sub["age"] == age) & (sub["mode"] == mode) & (sub["group"] == "Drug")]["value"].tolist()
                if not vals1 or not vals2:
                    continue
                stat1, p1 = stats.shapiro(vals1)
                stat2, p2 = stats.shapiro(vals2)
                if p1 > 0.05 and p2 > 0.05:
                    _, pval = stats.ttest_ind(vals1, vals2)
                else:
                    _, pval = stats.mannwhitneyu(vals1, vals2)
                if pval < 0.05:
                    start = tick_map.get((age, mode, "Control"))
                    end = tick_map.get((age, mode, "Drug"))
                    y = max(max(vals1), max(vals2)) * 1.05
                    ax.plot([start, start, end, end], [y, y + 0.5, y + 0.5, y], lw=1.2, c='black')
                    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*'
                    ax.text((start + end)/2, y + 0.5, sig, ha='center', va='bottom', fontsize=12)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf_path = f"{output_prefix}.pdf"
    jpeg_path = f"{output_prefix}.jpeg"
    PdfPages(pdf_path).savefig(fig)
    plt.savefig(jpeg_path, dpi=300)

    return pdf_path, jpeg_path

# Utilisation
if __name__ == "__main__":
    combined_df = load_all_data(file_paths)
    plot_combined_blocked(combined_df)
    plt.show()
