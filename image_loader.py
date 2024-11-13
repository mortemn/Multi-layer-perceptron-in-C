from PIL import Image
import numpy as np

# Step 1: Read the pixel values from the file
file_path = "input.txt"
with open(file_path, "r") as f:
    # Read each line, strip newline characters, convert to float, then scale to 0-255 and cast to int
    pixel_values = [int(float(line.strip()) * 255) for line in f.readlines()]

# Step 2: Verify and reshape to 28x28
if len(pixel_values) != 784:
    raise ValueError(
        "File does not contain 784 pixels required for a 28x28 image")

# Convert to numpy array and reshape
pixel_array = np.array(pixel_values, dtype=np.uint8).reshape((28, 28))

# Step 3: Create and save/display the image
image = Image.fromarray(pixel_array, mode="L")  # 'L' mode is for grayscale
image.show()  # Display the image
image.save("output_image.png")  # Save to a file if needed
