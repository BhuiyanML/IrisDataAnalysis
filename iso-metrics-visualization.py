import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Customize matplotlib settings
matplotlib.use('Agg')
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['font.size'] = 15
plt.rcParams["figure.figsize"] = [6, 4.5]


def create_probability_histogram(data, column, figName):
    if column == "USABLE_IRIS_AREA":
        w = 2
        xlim = (40, 105)
        ylim = (0, 0.25)
    elif column == "IRIS_SCLERA_CONTRAST":
        w = 2
        xlim = (0, 100)
        ylim = (0, 0.14)
    elif column == "IRIS_PUPIL_CONTRAST":
        w = 5
        xlim = (0, 100)
        ylim = (0, 1.0)
    elif column == "PUPIL_BOUNDARY_CIRCULARITY":
        w = 1
        xlim = (98.5, 100)
        ylim = (0, 0.04)
    elif column == "GREY_SCALE_UTILISATION":
        w = 0.05
        xlim = (4.5, 7)
        ylim = (0, 0.12)
    elif column == "IRIS_RADIUS":
        w = 2
        xlim = (90, 210)
        ylim = (0, 0.16)
    elif column == "PUPIL_IRIS_RATIO":
        w = 1
        xlim = (18, 75)
        ylim = (0, 0.14)
    elif column == "IRIS_PUPIL_CONCENTRICITY":
        w = 3
        xlim = (85, 100)
        ylim = (0, 0.2)
    elif column == "MARGIN_ADEQUACY":
        w = 5
        xlim = (5, 120)
        ylim = (0, 1)
    elif column == "SHARPNESS":
        w = 2
        xlim = (0, 100)
        ylim = (0, 0.14)
    elif column == "MOTION_BLUR":
        w = 0.03
        xlim = (0.9, 2.5)
        ylim = (0, 0.10)
    elif column == "OVERALL_QUALITY":
        w = 2
        xlim = (0, 100)
        ylim = (0, 0.12)

    # Create bins based on the defined width
    mn = data[column].min()
    mx = data[column].max()
    bins = np.arange(mn, mx + w, w)

    # Define transparency and color
    alpha = 0.5
    palette = sns.color_palette('Set2')
    colors = {"Adult": palette[0], "Newborn": palette[1]}

    # Create the histogram plot
    ax = sns.histplot(
        data,
        x=column,
        hue="image_type",
        element="bars",
        bins=bins,
        palette=colors,
        legend=True,
        stat="probability",
        common_norm=False,
        edgecolors='none',
        alpha=alpha
    )

    sns.move_legend(ax, "upper left", title=None)
    plt.xlabel('Score')
    plt.ylabel('Probability')

    # Set specific limits for the x-axis and y-axis
    plt.gca().set_xlim(left=xlim[0], right=xlim[1])
    plt.gca().set_ylim(bottom=ylim[0], top=ylim[1])
    plt.grid(True)

    # Save the figure with the provided figName
    plt.savefig(figName + f'{column}.pdf', format='pdf', dpi=600)

    # Clear the plot for the next iteration
    plt.clf()

    print(f'{column} figure generated!!')


# Directories and class list
input_dir = './matching-scores-final/iso-scores/'
output_dir = './figures/'

data_list = []
# Read pairs of dataframes
files = os.path.join(input_dir, '*.txt')
files = sorted(glob.glob(files))

for file in files:
    df = pd.read_csv(file, low_memory=False)

    # Identify if it's an Authentic or Synthetic image
    if file.split('/')[-1].split('-')[0] == 'Adult':
        df['image_type'] = 'Adult'
    else:
        df['image_type'] = 'Newborn'

    # Filter rows where "OVERALL_QUALITY" or "USABLE_IRIS_AREA" is not 255 or 0
    df = df[(df['OVERALL_QUALITY'] != 255) | (df['OVERALL_QUALITY'] != 0)]
    df = df[(df['USABLE_IRIS_AREA'] != 255) | (df['USABLE_IRIS_AREA'] != 0)]

    data_list.append(df)

# Concatenate all dataframes into a single dataframe
df = pd.concat(data_list, ignore_index=True)

# Create probability distribution histograms for various columns
create_probability_histogram(df, "USABLE_IRIS_AREA", output_dir)
create_probability_histogram(df, "IRIS_SCLERA_CONTRAST", output_dir)
create_probability_histogram(df, "IRIS_PUPIL_CONTRAST", output_dir)
create_probability_histogram(df, "PUPIL_BOUNDARY_CIRCULARITY", output_dir)
create_probability_histogram(df, "GREY_SCALE_UTILISATION", output_dir)
create_probability_histogram(df, "IRIS_RADIUS", output_dir)
create_probability_histogram(df, "PUPIL_IRIS_RATIO", output_dir)
create_probability_histogram(df, "IRIS_PUPIL_CONCENTRICITY", output_dir)
create_probability_histogram(df, "MARGIN_ADEQUACY", output_dir)
create_probability_histogram(df, "SHARPNESS", output_dir)
create_probability_histogram(df, "MOTION_BLUR", output_dir)
create_probability_histogram(df, "OVERALL_QUALITY", output_dir)

print('Generating probability distribution figures complete!')
