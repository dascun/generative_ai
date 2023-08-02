import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the VAE model
latent_dim = 64

# Encoder
encoder_inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(
    encoder_inputs
)
x = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
z_mean = tf.keras.layers.Dense(latent_dim)(x)  # Latent mean
z_log_var = tf.keras.layers.Dense(latent_dim)(x)  # Latent log variance
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")


# Sampling function for reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name="z")(
    [z_mean, z_log_var]
)

# Decoder
latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = tf.keras.layers.Dense(128, activation="relu")(latent_inputs)
x = tf.keras.layers.Dense(7 * 7 * 64, activation="relu")(x)
x = tf.keras.layers.Reshape((7, 7, 64))(x)
x = tf.keras.layers.Conv2DTranspose(
    64, 3, activation="relu", strides=2, padding="same"
)(x)
x = tf.keras.layers.Conv2DTranspose(
    32, 3, activation="relu", strides=2, padding="same"
)(x)
decoder_outputs = tf.keras.layers.Conv2DTranspose(
    1, 3, activation="sigmoid", padding="same"
)(x)
decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

# VAE
vae_inputs = encoder_inputs
vae_outputs = decoder(z)
vae = tf.keras.Model(vae_inputs, vae_outputs, name="vae")  # doesn't work when this vae model weights are saved in .h5 format and loaded later

# Function to preprocess the image
def preprocess_image(image):
    img_array = image.astype("float32") / 255.0  # Normalize the pixel values to [0, 1]
    img_array = np.expand_dims(
        img_array, axis=-1
    )  # Add channel dimension for grayscale
    return img_array


# Function to generate an image using the VAE model
def generate_image(sample_image):
    # Preprocess the sample image to make it compatible with the VAE model
    sample_image = np.expand_dims(sample_image, axis=0)  # Add batch dimension

    # Pass the sample image through the encoder to get the mean and log variance of the latent space
    z_mean, z_log_var = encoder.predict(sample_image)  # Fix here

    # Use the mean and log variance to sample a latent vector
    epsilon = np.random.normal(size=(1, latent_dim))
    sampled_z = z_mean + np.exp(0.5 * z_log_var) * epsilon

    # Pass the sampled latent vector through the decoder to generate the output image
    generated_image = decoder.predict(sampled_z)  # Fix here

    return generated_image[0, :, :, 0]


# Load the MNIST dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Select a random image from the dataset
sample_image = x_train[np.random.randint(0, x_train.shape[0])]

# Preprocess the sample image to make it compatible with the VAE model
preprocessed_sample_image = preprocess_image(sample_image)

# Generate the output image
generated_image = generate_image(preprocessed_sample_image)

# Display the sample and generated images side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(sample_image, cmap="gray")
axes[0].axis("off")
axes[0].set_title("Sample Image")

axes[1].imshow(generated_image, cmap="gray")
axes[1].axis("off")
axes[1].set_title("Generated Image")

plt.show()
