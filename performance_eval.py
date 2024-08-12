import os
import json
import glob
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from modules import IrisMetrics
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.use('Agg')

# Customize matplotlib settings
plt.rcParams['figure.constrained_layout.use'] = True
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 15
plt.rcParams["figure.figsize"] = [6, 4.5]

# Define input and output directories
input_dir = './matching-scores-final/scores/'
output_dir = './figures-final/'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize dictionary to store results
matcher_results = {}

# Get a list of all .txt files in the input directory
files = sorted(glob.glob(os.path.join(input_dir, '*.txt')), reverse=True)

# Process each file individually
for file in files:
    # Read the data from the file
    data = pd.read_csv(file)

    # Drop rows where 'score' is NaN and filter out rows where 'score' is -1
    data = data.dropna(subset=['score'])
    data = data[data['score'] != -1]
    data['score'] = data['score'].round(2)

    # Toggle the 'label' values: 0 becomes 1, and 1 becomes 0
    data['label'] = data['label'] ^ 1

    # Create a list of (label, score) tuples
    observations = list(zip(data['label'], data['score']))

    # Extract the filename and matcher type from the file path
    matcher = os.path.basename(file).split('-')[1]
    matcher_type = os.path.basename(file).split('-')[2]
    matcher_name = f'{matcher}-{matcher_type}'

    # Determine comparison type based on matcher type
    comparison_type = 'similarity' if matcher == 'DGR' else 'distance'

    # Compute d-prime
    dprime = IrisMetrics.compute_d_prime(observations)

    # Compute metrics
    results = IrisMetrics.compute_fmr_fnmr_eer_auc(observations, comparison_type)

    # Compute Equal Error Rate (EER)
    EER = (results['FNMR'] + results['FMR']) / 2.0

    # Store results in the dictionary
    matcher_results[matcher_name] = {
        "dprime": dprime,
        "FNMR": results['FNMR'],
        "FMR": results['FMR'],
        "EER Threshold": results['EER_threshold'],
        "EER": EER,
        "AUC": results['AUC'],
        "FMRS": results['FMRS'],
        "TMRS": results['TMRS']
    }

    # Output results
    print(f"Filename: {file}")
    print(f"d-prime: {dprime:.2f}")
    print(f"FNMR at EER: {results['FNMR']:.2f}")
    print(f"FMR at EER: {results['FMR']:.2f}")
    print(f"EER Threshold: {results['EER_threshold']:.2f}")
    print(f"EER: {EER:.2f}")
    print(f"AUC: {results['AUC']:.2f}")

    # Data visualization
    data.loc[data["label"] == 1, "label"] = "Genuine"
    data.loc[data["label"] == 0, "label"] = "Impostor"
    data = data.sort_values(by='label')

    if matcher_name == 'HDBIF-Norm':
        w = 0.02
        x_right = 0.6
        y_top = 0.30
    elif matcher_name == 'HDBIF-Vanilla':
        w = 0.02
        x_right = 0.8
        y_top = 0.40
    elif matcher_name == 'OSIRIS-Norm':
        w = 0.022
        x_right = 0.6
        y_top = 0.5
    elif matcher_name == 'OSIRIS-Vanilla':
        w = 0.03
        x_right = 0.8
        y_top = 0.25
    elif matcher_name == 'USIT-Norm':
        w = 0.02
        x_right = 0.6
        y_top = 0.6
    elif matcher_name == 'USIT-Vanilla':
        w = 0.02
        x_right = 0.6
        y_top = 0.35
    elif matcher_name == 'WIRIS-Vanilla':
        w = 0.02
        x_right = 0.6
        y_top = 0.40
    else:
        w = 0.03
        x_right = 1.0
        y_top = 0.25

    lower = data['score'].min()
    upper = data['score'].max()
    bins = np.arange(lower, upper + w, w)

    # Plot distribution
    palette = sns.color_palette('Set2')
    colors = {"Genuine": palette[0], "Impostor": palette[1]}
    plt.figure() #figsize=(3.54, 3.54*0.8)
    ax = sns.histplot(
        data, x="score", hue="label", element="bars",
        bins=bins, stat='probability', fill=True, palette=colors,
        legend=True, common_norm=False, linewidth=1.5
    )

    # Set colors and styles
    linestyles = {"Genuine": "-", "Impostor": "-"}
    for line, label in zip(ax.lines, ["Impostor", "Genuine"]):
        line.set_color(colors[label])
        line.set_linestyle(linestyles[label])

    # Add legend with d-prime value
    legend_elements = [
        Line2D([0], [0], color=colors['Genuine'], linestyle='-', lw=1.5, label='Genuine'),
        Line2D([0], [0], color=colors['Impostor'], linestyle='-', lw=1.5, label='Impostor'),
        Line2D([0], [0], color='none', linestyle='', label=f"d'={dprime:.2f}"),
        Line2D([0], [0], color='none', linestyle='', label=f"τ={results['EER_threshold']:.2f}"),
        Line2D([0], [0], color='none', linestyle='', label=f"EER={EER:.2f}"),
        Line2D([0], [0], color='none', linestyle='', label=f"AUC={results['AUC']:.2f}")
    ]

    ax.legend(handles=legend_elements,
              bbox_to_anchor=(0, 1.02, 1, 0.225),
              loc='upper center',
              mode='expand',
              borderaxespad=0,
              ncol=3,
              fontsize=15,
              edgecolor="black",
              handletextpad=0.3,
              frameon=True)

    # Set titles and labels
    plt.xlabel('Score')
    plt.ylabel('Probability')
    plt.gca().set_xlim(0, x_right)
    plt.gca().set_ylim(0, y_top)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, matcher_name + '-dist.pdf'), format='pdf', dpi=600)
    plt.close()

# Save the dictionary as JSON
with open(os.path.join(output_dir, 'matcher_results.json'), 'w') as json_file:
    json.dump(matcher_results, json_file, indent=4)

# Plot ROC curve
plt.figure() # figsize=(5.54, 5.54*0.8)
for matcher, results in matcher_results.items():
    plt.plot(results['FMRS'], results['TMRS'], label=f"{matcher} (EER={results['EER']:.2f}, AUC={results['AUC']:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('FMR')
plt.ylabel('TMR')
plt.grid(True)
plt.legend(title=None, loc='lower right')
plt.savefig(os.path.join(output_dir, 'roc_curve.pdf'), format='pdf', dpi=600)
