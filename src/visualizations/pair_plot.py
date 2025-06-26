import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import os

def create_iris_pair_plot(save_path="figures/pair_plot.png"):
    """
    Generate and save a pair plot for the Iris dataset, with points colored by species.

    Parameters:
    - save_path (str): Path to save the pair plot image (default: 'figures/pair_plot.png')
    Returns:
    - None: Displays the plot and saves it as a PNG
    """
    # Close all existing figures to prevent duplicates
    plt.close('all')

    # Load the Iris dataset
    iris = load_iris()
    iris_dataframe = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_dataframe['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # Create pair plot with seaborn
    sns.set(style="ticks", context="talk")
    pair_plot = sns.pairplot(iris_dataframe, hue="species", diag_kind="hist", 
                             markers=["o", "s", "D"], palette="husl",
                             height=2.0, aspect=1.2, 
                             plot_kws={'s': 30},
                             diag_kws={'alpha': 0.6})

    # Adjust axis label font size
    for ax in pair_plot.axes.flatten():
        ax.tick_params(axis='both', labelsize=8)
        ax.set_xlabel(ax.get_xlabel(), fontsize=10)
        ax.set_ylabel(ax.get_ylabel(), fontsize=10)

    # Reposition and modify legend (to top-right)
    pair_plot._legend.set_bbox_to_anchor((0.85, 0.95))
    pair_plot._legend.set_frame_on(True)
    pair_plot._legend.get_frame().set_alpha(0.9)
    pair_plot._legend.set_title("Species", prop={'size': 8})

    # Adjust plot aesthetics to ensure title and legend visibility
    pair_plot.figure.suptitle("Pair Plot of Iris Dataset Features by Species", y=0.98, fontsize=14)
    pair_plot.figure.subplots_adjust(top=0.82, right=0.75)

    # Ensure the 'figures' directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the plot as a PNG before displaying
    pair_plot.figure.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    print(f"Pair plot saved to {save_path}")

    # Display the pair plot
    plt.show()

if __name__ == "__main__":
    create_iris_pair_plot()