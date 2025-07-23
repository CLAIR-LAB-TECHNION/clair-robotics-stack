# import os
# print("DISPLAY =", os.environ.get("DISPLAY"))

# import glfw

# if not glfw.init():
#     raise Exception("GLFW init failed")

import numpy as np
import matplotlib.pyplot as plt  
  
data = np.load('./')
print(data.files) # This will print the keys of the arrays within the NPZ file
# Example: If your images are stored under the key 'images_array'

# images = data['images_array']
# plt.imshow(images[0])
# plt.title("First Image from NPZ")
# plt.axis('off') # Turn off axis labels for cleaner image display
# plt.show()

# fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 6))
# for i, ax in enumerate(axes.flatten()):
#     if i < len(images): # Ensure you don't go out of bounds
#         ax.imshow(images[i])
#         ax.set_title(f"Image {i+1}")
#         ax.set_axis_off()
# plt.tight_layout()
# plt.show()