import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Set a style for improved aesthetics
plt.style.use('seaborn-v0_8-darkgrid')

# Load the Iris Dataset
iris = load_iris()

# Convert to a Pandas DataFrame for easier plotting
iris_dataframe = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_dataframe['species'] = iris.target_names[iris.target]

# Create Ridge Plots for each Feature using Seaborn & Matplotlib
features = iris_dataframe.columns[:-1]
species_names = iris_dataframe['species'].unique()
species_names.sort()

# Define a color palette
palette = sns.color_palette("viridis", n_colors=len(species_names))

# Create a figure and a set of subplots
fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(10, 2 * len(features)), sharex=True)

# Add space between subplots
fig.subplots_adjust(hspace=-0.5)

# Loop through each feature and create a KDE plot for each species
for i, feature in enumerate(features):
    ax = axes[i]
    for j, species in enumerate(species_names):
        subset = iris_dataframe[iris_dataframe['species'] == species]
        sns.kdeplot(
            data=subset,
            x=feature,
            fill=True,
            alpha=0.7,
            lw=1.5,
            color=palette[j],
            ax=ax,
            label=species if i == 0 else None
        )

    y_tick_positions = []
    y_tick_labels = []
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Set x-axis label only for the bottom plot
    if i == len(features) - 1:
        ax.set_xlabel(f"Measurement (cm)", fontsize=12)
    else:
        ax.set_xlabel("")

    # Set title for each row (feature name)
    ax.set_title(f'{feature.replace("_", " ").title()}', loc='left', fontsize=12)
    ax.spines[['left', 'right', 'top']].set_visible(False)


# Adjust legend and overall layout
fig.suptitle('Distribution of Iris Features by Species (Ridge Plot Style)', y=0.98, fontsize=16)
fig.legend(title='Species', bbox_to_anchor=(0.895, 0.89), loc='upper right', frameon=False)
plt.tight_layout(rect=[0, 0, 0.9, 1])

plt.show()