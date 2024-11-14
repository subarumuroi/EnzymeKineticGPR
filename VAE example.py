import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0  # Normalize images to [0, 1]
x_train = np.reshape(x_train, [-1, 28, 28, 1])  # Reshape for the CNN model
x_test = x_test.astype("float32") / 255.0
x_test = np.reshape(x_test, [-1, 28, 28, 1])

# Parameters for VAE
img_shape = (28, 28, 1)
latent_dim = 2  # Latent space dimensionality

# Encoder Model
inputs = layers.Input(shape=img_shape)

# Conv layers for feature extraction
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu")(x)

# Latent space: we output both mean and log-variance for the Gaussian distribution
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# Latent space sampling using reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder Model
latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

# Instantiate the encoder and decoder models
encoder = models.Model(inputs, [z_mean, z_log_var, z])
decoder = models.Model(latent_inputs, outputs)

# VAE Model
vae_outputs = decoder(encoder(inputs)[2])  # Use the sampled latent vector
vae = models.Model(inputs, vae_outputs)

# Custom VAE Loss Function (Reconstruction Loss + KL Divergence)
def vae_loss(x, x_decoded_mean, z_mean, z_log_var):
    # Reconstruction loss (binary crossentropy)
    xent_loss = tf.reduce_mean(
        tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, x_decoded_mean), axis=(1, 2))
    )

    # KL divergence loss
    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
    )

    return xent_loss + kl_loss

# Optimizer
optimizer = tf.keras.optimizers.Adam()

# Custom Training Loop
@tf.function
def train_step(x_batch):
    with tf.GradientTape() as tape:
        z_mean, z_log_var, z = encoder(x_batch)
        x_decoded_mean = decoder(z)
        
        # Compute the VAE loss
        loss = vae_loss(x_batch, x_decoded_mean, z_mean, z_log_var)
    
    # Compute gradients and apply them
    grads = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(grads, vae.trainable_variables))
    
    return loss

# Training Loop
epochs = 30
batch_size = 128
num_batches = x_train.shape[0] // batch_size

for epoch in range(epochs):
    total_loss = 0
    for i in range(num_batches):
        # Select a batch of images
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        x_batch = x_train[batch_start:batch_end]
        
        # Perform one training step
        loss = train_step(x_batch)
        total_loss += loss
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / num_batches:.4f}")


# Sampling from the latent space
def plot_latent_space(encoder, data):
    z_mean, _, _ = encoder.predict(data, batch_size=128)
    plt.figure(figsize=(10, 8))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=np.arange(len(z_mean)), cmap='viridis')
    plt.colorbar()  # To show the color scale
    plt.title("Latent Space Visualization")
    plt.show()

# Plot latent space of the test set
plot_latent_space(encoder, x_test)


# Function to plot both original and reconstructed images side by side
def plot_comparison(original_images, reconstructed_images, num_samples=10):
    plt.figure(figsize=(20, 4))
    
    for i in range(num_samples):
        # Plot original image
        ax = plt.subplot(2, num_samples, i + 1)
        plt.imshow(original_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        
        # Plot reconstructed image
        ax = plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    
    plt.show()

# Select a random batch of images from the test set for comparison
random_indices = np.random.choice(x_test.shape[0], 10, replace=False)
original_images = x_test[random_indices]

# Use the VAE to reconstruct the selected images
z_mean, z_log_var, z = encoder.predict(original_images)
reconstructed_images = decoder.predict(z)

# Plot the comparison between original and reconstructed images
plot_comparison(original_images, reconstructed_images, num_samples=10)