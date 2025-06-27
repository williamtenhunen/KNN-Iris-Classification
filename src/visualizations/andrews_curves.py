import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns

# Set the style for aesthetics
plt.style.use('seaborn-v0_8-darkgrid')

# Load the Iris Dataset
iris = load_iris()

# Convert to Pandas DataFrame (easier plotting)
iris_dataframe = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_dataframe['species'] = iris.target_names[iris.target]

# Generate Andrews Curves
pd.plotting.andrews_curves(
    iris_dataframe,
    class_column='species',
    ax=plt.gca(),
    colormap='viridis'
)

plt.title('Andrews Curves of Iris Features by Species', fontsize=16)
plt.xlabel('Fourier Series Coefficient', fontsize=12)
plt.ylabel('f(t) value', fontsize=12)

# Add a legend outside of the plot (for clarity)
plt.legend(title='Species', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.show()