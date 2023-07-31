import numpy as np
import matplotlib.pyplot as plt

# Define the data
data = np.array([[100, 80, 85],
                 [75, 100, 84],
                 [53, 57, 100],
                 [95, 94, 89],
                 [96, 82, 95],
                 [84, 94, 85],
                 [83, 96, 95],
                 [71, 71, 85],
                 [68, 75, 86]])

# Create the heatmap
fig, ax = plt.subplots()
heatmap = ax.imshow(data, cmap='coolwarm')

# Add colorbar
cbar = ax.figure.colorbar(heatmap, ax=ax)

# Set the tick labels and axis labels
ax.set_xticks(np.arange(data.shape[1]))
ax.set_yticks(np.arange(data.shape[0]))
ax.set_xticklabels(['True class 0', 'True class 1', 'True class 2'])
ax.set_yticklabels(['True class 0', 'True class 1', 'True class 2', 'Class 0 predicted as 1',
                    'Class 0 predicted as 2', 'Class 1 predicted as 0', 'Class 1 predicted as 2',
                    'Class 2 predicted as 0', 'Class 2 predicted as 1'])


# Loop over data dimensions and create text annotations
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        text = ax.text(j, i, f'{data[i, j]}%',
                       ha='center', va='center', color='black')

# Show the plot
plt.title('Heatmap GO terms overlap, multiclass Sent2Vec Transformer')
plt.tight_layout()
plt.show()