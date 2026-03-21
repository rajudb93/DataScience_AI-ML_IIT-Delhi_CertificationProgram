# ==========================================================
# CNN Basic Demonstration: Convolution → Activation → Pooling
# ==========================================================

# -----------------------------
# Import Required Libraries
# -----------------------------
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# -----------------------------
# Plot Settings
# -----------------------------
plt.rc('figure', autolayout=True)
plt.rc('image', cmap='magma')


# ==========================================================
# 1. Define Convolution Kernel (Edge Detection Filter)
# ==========================================================
kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])


# ==========================================================
# 2. Load and Prepare Image
# ==========================================================
# Create a synthetic image for demonstration (300x300 grayscale)
# This makes the script self-contained and doesn't require external files
np.random.seed(42)
image_array = np.random.randint(0, 256, (300, 300, 1), dtype=np.uint8)
image = tf.cast(image_array, dtype=tf.float32) / 255.0


# ==========================================================
# 3. Display Original Image
# ==========================================================
img = tf.squeeze(image).numpy()

plt.figure(figsize=(5,5))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title("Original Grayscale Image")
plt.show()


# ==========================================================
# 4. Prepare Image for CNN
# ==========================================================

# Normalize pixel values (0-255 → 0-1)
image = tf.image.convert_image_dtype(image, dtype=tf.float32)

# Add batch dimension
# Shape becomes: (1, height, width, channels)
image = tf.expand_dims(image, axis=0)


# ==========================================================
# 5. Prepare Kernel for CNN
# ==========================================================

# Reshape kernel to match TensorFlow format
# (filter_height, filter_width, in_channels, out_channels)
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)


# ==========================================================
# 6. Convolution Layer
# ==========================================================

image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=1,
    padding='SAME'
)


# ==========================================================
# 7. Activation Layer (ReLU)
# ==========================================================

image_detect = tf.nn.relu(image_filter)


# ==========================================================
# 8. Pooling Layer (Max Pooling)
# ==========================================================

image_condense = tf.nn.pool(
    input=image_detect,
    window_shape=(2,2),
    pooling_type='MAX',
    strides=(2,2),
    padding='SAME'
)


# ==========================================================
# 9. Plot Results
# ==========================================================

plt.figure(figsize=(15,5))

# ----- Convolution Result -----
plt.subplot(1,3,1)
plt.imshow(tf.squeeze(image_filter))
plt.axis('off')
plt.title("Convolution Output")


# ----- Activation Result -----
plt.subplot(1,3,2)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title("ReLU Activation")


# ----- Pooling Result -----
plt.subplot(1,3,3)
plt.imshow(tf.squeeze(image_condense))
plt.axis('off')
plt.title("Max Pooling Output")


plt.show()