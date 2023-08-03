# Generative AI
This project helps to understand simple implementations of Generative AI from scratch

## Variational Auto-Encoder (VAE)
Variational Autoencoders were proposed to address the limitation of traditional autoencoders by introducing a probabilistic approach to the latent space. Instead of mapping data points to a fixed point in the latent space, VAEs map them to a probability distribution. This allows us to sample new points from the latent space and generate diverse and novel data points.

### Key Components of VAEs

**Encoder** : The encoder network in a VAE takes an input data point and outputs the parameters of the probability distribution in the latent space. Typically, the encoder produces the mean and variance of a Gaussian distribution.

**Reparameterization Trick** : To enable backpropagation through the sampling process in VAEs, the reparameterization trick is used. Instead of sampling directly from the Gaussian distribution, the model samples from a standard Gaussian distribution (with mean 0 and variance 1) and then scales and shifts the sampled values using the mean and variance produced by the encoder.

**Decoder** : The decoder network takes a point from the latent space and reconstructs the original data point. The decoder is trained to minimize the difference between the reconstructed data and the original data.

During training, VAEs aim to minimize the reconstruction error (how well the decoder can reconstruct the original input) and the Kullback-Leibler (KL) divergence between the learned latent distribution and the standard Gaussian distribution. The KL divergence term encourages the latent space to be close to a standard Gaussian, ensuring that the latent representations are meaningful and smooth.

**Code implementation references** :

[VAE Model](https://github.com/dascun/generative_ai/edit/main/vae_tf.py)

[Generate image for MNIST sample](https://github.com/dascun/generative_ai/edit/main/mnist_image_generator_vae_tf.py)

## Generative Adversarial Networks (GANs)

Generative Adversarial Networks (GANs) consist of two neural networks, the generator, and the discriminator, which are trained simultaneously in a competitive manner.

### Key Components of GANs

**Generator**: The generator network takes random noise as input and generates synthetic data samples. Initially, the generator produces random and meaningless output, but as it is trained, it learns to generate data that resembles the real data distribution.

**Discriminator**: The discriminator network acts as a binary classifier and takes both real data samples and synthetic data samples as input. Its goal is to distinguish between real and fake data. The discriminator is trained to maximize its ability to correctly identify real data and minimize the probability of incorrectly classifying fake data.

**Training Process**: The training of GANs involves a minimax game between the generator and the discriminator. The generator aims to produce data that is indistinguishable from real data to fool the discriminator, while the discriminator aims to correctly classify the real and fake data.

During training, the generator and discriminator are updated alternately. First, a batch of real data is fed to the discriminator, and it is trained to maximize the probability of correctly classifying real data and minimize the probability of incorrectly classifying fake data. Then, a batch of random noise is fed to the generator, and the discriminator's loss is used to update the generator's weights to minimize the probability of the discriminator classifying the generated data as fake.

As training progresses, the generator becomes more skilled at generating realistic data, while the discriminator becomes better at distinguishing between real and fake data. Ideally, the generator will learn to generate data that is so realistic that the discriminator cannot differentiate it from real data.

GAN training can be challenging and sensitive to hyperparameters, and there are various improvements and modifications, such as Wasserstein GANs (WGANs) and Deep Convolutional GANs (DCGANs), that address stability and convergence issues.

**Code implementation references**:

[GAN Model](https://github.com/dascun/generative_ai/edit/main/gan_tf.py)

This implementation uses TensorFlow to build and train a simple GAN to generate images of handwritten digits from the MNIST dataset. The generator and discriminator networks are constructed using fully connected layers, and the GAN is trained using a binary cross-entropy loss function. The training process involves updating the discriminator and generator iteratively to improve their performance in generating realistic images. The generated images can be saved for visualization at specific intervals during training.
