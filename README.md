# generative_ai
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
