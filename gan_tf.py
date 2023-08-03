# Import necessary libraries: TensorFlow for building and training the GAN, NumPy for numerical computations, and Matplotlib for image visualization.
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load the MNIST dataset, which contains images of handwritten digits along with their corresponding labels. In this case, we are only interested in the images (train_images).
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values of the images to be in the range [-1, 1]. This is a common practice for training GANs, as it helps in convergence. The images are also reshaped
# to have a single channel (grayscale) and converted to a float32 data type.
train_images = (train_images - 127.5) / 127.5
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(
    "float32"
)

# Set random seeds for TensorFlow and NumPy to ensure reproducibility of results.
tf.random.set_seed(42)
np.random.seed(42)


# Define the generator model as a function called build_generator(). The generator takes random noise of dimension 100 as input and generates a 28x28 grayscale image as
# output. The generator consists of several dense layers with leaky ReLU activation functions and batch normalization. The final layer uses the hyperbolic tangent (tanh)
# activation to squash the output values between -1 and 1, which matches the range of the normalized images.
def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(256, input_dim=100))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(28 * 28 * 1, activation="tanh"))
    model.add(layers.Reshape((28, 28, 1)))
    return model


# Define the discriminator model as a function called build_discriminator(). The discriminator takes a 28x28 grayscale image as input and produces a single scalar value as
# output, representing the probability that the input image is real (1) or fake (0). The discriminator is also constructed using dense layers with leaky ReLU activation
# functions.
def build_discriminator():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


# Define the GAN model as a function called build_gan(generator, discriminator). The GAN combines the generator and discriminator models. The discriminator is set to
# non-trainable during GAN training so that only the generator's weights are updated. This prevents the discriminator from getting too good early in training and dominating
# the process.
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


# Create instances of the generator and discriminator using their respective functions.
generator = build_generator()
discriminator = build_discriminator()

# Compile the discriminator model with binary cross-entropy loss and an Adam optimizer. The binary cross-entropy loss is appropriate for binary classification tasks like
# determining if an image is real or fake.
discriminator.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
)

# Create the GAN by combining the generator and discriminator using the build_gan() function.
gan = build_gan(generator, discriminator)

# Compile the GAN model with binary cross-entropy loss and an Adam optimizer. The GAN's loss is calculated during training when updating the generator's weights.
gan.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
)

# Set the number of training epochs and the batch size for training.
epochs = 20000
batch_size = 128

# Training loop
for epoch in range(epochs):
    # In each epoch, generate random noise samples of shape (batch_size, 100). These samples will be used as input to the generator to create fake images.
    noise = np.random.normal(0, 1, size=(batch_size, 100))

    # Use the generator to produce fake images from the generated noise.
    generated_images = generator.predict(noise)

    # Select a random batch of real images from the MNIST dataset.
    real_images = train_images[
        np.random.randint(0, train_images.shape[0], batch_size)
    ]

    # Combine the real and generated images to create a batch of size (2 * batch_size).
    batch_images = np.concatenate([real_images, generated_images])

    # Create labels for the real images (labeled as 1) and generated images (labeled as 0).
    batch_labels = np.concatenate(
        [np.ones((batch_size, 1)), np.zeros((batch_size, 1))]
    )

    # Apply label smoothing by adding a small random noise to the labels. This is a regularization technique that helps prevent the discriminator from becoming too confident.
    batch_labels += 0.05 * np.random.random(batch_labels.shape)

    # Train the discriminator using the batch of real and generated images along with their labels. Calculate the discriminator's loss (binary cross-entropy) during this step.
    d_loss = discriminator.train_on_batch(batch_images, batch_labels)

    # Generate new random noise samples to be used as input to the generator.
    noise = np.random.normal(0, 1, size=(batch_size, 100))

    # Create labels for the generated images, pretending they are all real (labeled as 1).
    valid_labels = np.ones((batch_size, 1))

    # Train the GAN by updating the generator's weights to minimize the binary cross-entropy loss between the generated images and the "real" labels.
    g_loss = gan.train_on_batch(noise, valid_labels)

    # Print the discriminator and generator losses at regular intervals to track the progress of training.
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - D loss: {d_loss:.4f}, G loss: {g_loss:.4f}")

    # At certain intervals, generate 10 fake images using the generator and save them for visualization. The generated images are rescaled from the range [-1, 1] back to [0,
    # 1] before plotting and saving them as PNG files.
    if epoch % 1000 == 0:
        # Generate images using the generator
        generated_images = generator.predict(
            np.random.normal(0, 1, size=(10, 100))
        )

        # Rescale the images 0 - 1
        generated_images = 0.5 * generated_images + 0.5

        # Plot the generated images
        plt.figure(figsize=(10, 1))
        for i in range(10):
            plt.subplot(1, 10, i + 1)
            plt.imshow(generated_images[i].reshape(28, 28), cmap="gray")
            plt.axis("off")
        plt.savefig(f"generated_images_epoch_{epoch}.png")
        plt.close()
