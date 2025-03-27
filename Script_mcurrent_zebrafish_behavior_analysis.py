#Script complet avec:
# - Violin plots
# - Couleur du contrôle en gris, couleur de la drogue personnalisable
# - Page statistique à la fin du PDF
# - Export Excel
# - Pas de p-value affichée sous les graphes

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

# Configuration de police
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# À personnaliser par drogue
drug_color = "#DC143C"  #  Cette ligne change en fonction de la couleur de la drogue

# Fichier CSV
file_path = r"C:\Users\Laris\OneDrive\Desktop\New_csvfiles_newdata\SKF\SKF-38393_3dpf.csv"
df = pd.read_csv(file_path, nrows=192)
drug_name = file_path.split("\\")[-1].split('_')[0]

# Nettoyage et tri
df_trimmed = df.iloc[::2]
df_trimmed = df_trimmed.sort_values('animal').reset_index(drop=True)

columns_to_analyze = ["smldur", "smldist", "lardur", "lardist"]
animals = [f'WT{str(i).zfill(3)}' for i in range(1, 25)]
intervals = ["0-5 minutes", "5-10 minutes", "10-15 minutes", "15-20 minutes"]

print("WT001 smldur values:", df_trimmed[df_trimmed['animal'] == 'WT001']['smldur'].tolist())

data_by_animal = {}
for animal in animals:
    data_by_animal[animal] = {}
    for col in columns_to_analyze:
        values = df_trimmed[df_trimmed['animal'] == animal][col].tolist()
        data_by_animal[animal][col] = values

data_dict = {col: {interval: [] for interval in intervals} for col in columns_to_analyze}
data_dict["total"] = {col: [] for col in columns_to_analyze}

for col in columns_to_analyze:
    for i, interval in enumerate(intervals):
        for animal in animals:
            vals = data_by_animal[animal][col]
            if len(vals) > i:
                data_dict[col][interval].append(vals[i])
            else:
                print(f"Pas de donnée pour {animal} dans la colonne {col} à l'intervalle {interval}")
    for animal in animals:
        vals = data_by_animal[animal][col]
        total_value = np.sum(vals[:len(intervals)]) if len(vals) >= len(intervals) else vals[0] if len(vals) > 0 else None
        data_dict["total"][col].append(total_value)

def create_individual_graph(ax, column, interval, label=None, label_color='black', fig=None):
    if interval == "total":
        interval_data = data_dict["total"][column]
        interval_label = "0-20 minutes"
    else:
        interval_data = data_dict[column][interval]
        interval_label = interval

    first_group = interval_data[:12]
    second_group = interval_data[12:]

    mean_1 = np.mean(first_group)
    mean_2 = np.mean(second_group)
    std_1 = np.std(first_group)
    std_2 = np.std(second_group)

    stat1, p1 = stats.shapiro(first_group)
    stat2, p2 = stats.shapiro(second_group)

    if p1 > 0.05 and p2 > 0.05:
        t_stat, p_value = stats.ttest_ind(first_group, second_group, equal_var=False)
        test_name = "Unpaired t-test"
    else:
        u_stat, p_value = stats.mannwhitneyu(first_group, second_group)
        test_name = "Mann-Whitney U test"

    data = [first_group, second_group]
    parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=False)

    bodies = parts['bodies']
    if len(bodies) >= 2:
        bodies[0].set_facecolor('#AAAAAA')  # Control = gris
        bodies[1].set_facecolor(drug_color)  # Drug = couleur personnalisée
        for b in bodies:
            b.set_edgecolor('black')
            b.set_alpha(0.8)
    if 'cmeans' in parts:
        parts['cmeans'].set_color('black')

    for i, group in enumerate(data):
        ax.scatter([i] * len(group), group, color='#000000', alpha=0.6, s=60, marker='o')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control', drug_name], fontsize=20)

    if column in ["lardist", "smldist"]:
        ax.set_ylabel("Distance (mm)", fontsize=20)
    elif column in ["lardur", "smldur"]:
        ax.set_ylabel("Duration (s)", fontsize=20)

    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)

    y_max_point = max(np.max(first_group), np.max(second_group))
    buffer = (std_1 + std_2) * 0.5
    sig_y = y_max_point + buffer
    y_min_point = min(np.min(first_group), np.min(second_group))
    bottom_padding = (std_1 + std_2) * 0.1  # marge basse proportionnelle aux variations
    # Début d’axe selon le type de paramètre
    if column in ["smldur", "lardur"]:
        lower_bound = -50
    else:
        lower_bound = -1500

    ax.set_ylim([lower_bound, sig_y * 1.15])
    
    if p_value < 0.001:
        sig_symbol = '***'
    elif p_value < 0.01:
        sig_symbol = '**'
    elif p_value < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = ''

    if sig_symbol:
        ax.plot([0, 1], [sig_y, sig_y], color='black', linewidth=0.8)
        ax.text(0.5, sig_y + buffer * 0.3, sig_symbol, ha='center', va='bottom', fontsize=30)
        

    if label:
        y_max = ax.get_ylim()[1]
        ax.text(-1.0, y_max * 0.99, label,
                fontsize=30, fontweight='bold', va='center', ha='right',
                color=label_color, transform=ax.transData)

    ax.set_title("")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return {
        'interval': interval_label,
        'column': column,
        'control_mean': mean_1,
        'drug_mean': mean_2,
        'control_std': std_1,
        'drug_std': std_2,
        'p_value': p_value,
        'test': test_name
    }

def create_combined_graph():
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    columns_to_plot = ["smldur", "smldist", "lardur", "lardist"]
    letters = ['A', 'D', 'G', 'J']
    label_color = "#000000"

    for i, column in enumerate(columns_to_plot):
        create_individual_graph(axs[i], column, "total", label=letters[i], label_color=label_color, fig=fig)
        axs[i].set_title("")
        if column in ["smldur", "lardur"]:
            axs[i].set_ylabel("Duration (s)", fontsize=20)
        else:
            axs[i].set_ylabel("Distance (mm)", fontsize=20)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

def add_stats_summary_page(pdf, stats_list):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    df_stats = pd.DataFrame(stats_list)
    table_data = df_stats[["interval", "column", "test", "p_value"]]

    columns = ["Interval", "Parameter", "Test Used", "p-value"]
    cell_text = [[
        row["interval"],
        row["column"],
        row["test"],
        f"{row['p_value']:.4f}"
    ] for _, row in table_data.iterrows()]

    table = ax.table(cellText=cell_text, colLabels=columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    ax.set_title("Statistical Summary", fontsize=16, fontweight='bold')
    pdf.savefig(fig)
    plt.close(fig)
    

def save_graphs_to_pdf_and_excel():
    file_name_base = file_path.split("\\")[-1].replace(".csv", "")
    pdf_file_name = f'graphiques_combines_{file_name_base}_Version2.pdf'
    excel_file_name = f'statistiques_{file_name_base}_Version2.xlsx'

    stats_list = []

    with PdfPages(pdf_file_name) as pdf:
        for column in columns_to_analyze:
            for interval in intervals + ["total"]:
                fig, ax = plt.subplots(figsize=(8, 4))
                stats_result = create_individual_graph(ax, column, interval)
                pdf.savefig(fig)
                plt.close(fig)
                if stats_result:
                    stats_list.append(stats_result)

        combined_fig = create_combined_graph()
        pdf.savefig(combined_fig)
        plt.close(combined_fig)

        add_stats_summary_page(pdf, stats_list)

    df_stats = pd.DataFrame(stats_list)
    df_stats.to_excel(excel_file_name, index=False)

    print(f"Tous les graphiques ont été créés et sauvegardés dans {pdf_file_name}.")
    print(f"Les statistiques ont été sauvegardées dans {excel_file_name}.")

# Exécution
if __name__ == "__main__":
    save_graphs_to_pdf_and_excel()
