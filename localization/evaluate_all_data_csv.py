import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

##read csv generated by all_data_csv, plot error map with angle aggregation, fit polynomial to represent loss in space

df = pd.read_csv("/home/tamirshor/EARS_project/EARS/grid.csv")
df = df[df['dataset'] == 'validation']

#df1 = df.copy()
result = df.groupby(['x_gt', 'y_gt'])['loss'].mean().reset_index()
pivot_df = result.pivot(index='y_gt', columns='x_gt', values='loss')

# Convert the pivot DataFrame to a NumPy array
image_array = pivot_df.to_numpy()

# If you want to handle missing values (NaN) differently, you can use fillna
# For example, fill NaN values with a specific number (e.g., 0)
image_array = pivot_df.fillna(0).to_numpy()
plt.imshow(image_array, cmap='viridis', origin='lower')
plt.title(f"mean {image_array.mean()}")

# Add a colorbar
cbar = plt.colorbar()

# Set the label for the colorbar
cbar.set_label('Loss Value')

# Show the plot
#plt.show()
plt.close()


#create polynomial
img = image_array
x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
z = img.flatten()

# Fit a 2D polynomial using numpy.linalg.lstsq
degree = 2
A = np.vstack([np.ones_like(x.flatten()), x.flatten(), y.flatten(), x.flatten()**2, x.flatten()*y.flatten(), y.flatten()**2]).T
coefficients, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

# Create a meshgrid for visualization
X, Y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
Z_fit = (
    coefficients[0] +
    coefficients[1] * X.flatten() +
    coefficients[2] * Y.flatten() +
    coefficients[3] * X.flatten() ** 2 +
    coefficients[4] * X.flatten() * Y.flatten() +
    coefficients[5] * Y.flatten() ** 2
)

# Reshape the fitted values back to 2D
Z_fit = Z_fit.reshape(img.shape)

# Plot the original and fitted surfaces
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the original surface
ax1.imshow(img, cmap='viridis', origin='lower')
ax1.set_title('Original Image')

# Plot the fitted surface
ax2.imshow(Z_fit, cmap='viridis', origin='lower')
ax2.set_title('Fitted Surface')

plt.show()