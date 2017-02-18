import tensorflow as tf
import numpy as np


def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out)) 
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)


def zero_init(fan_out):
    return tf.zeros([fan_out], dtype=tf.float32)


def _construct_layer(input_layer, (in_dim, out_dim), activation_func, _init_weights=xavier_init, 
                     _init_biases=zero_init, _name_template='{}_{:03d}', _iter_lbl=0):
    weights_name = _name_template.format('weights', _iter_lbl)
    biases_name = _name_template.format('biases', _iter_lbl)

    weights = tf.Variable(_init_weights(in_dim, out_dim), name=weights_name)
    biases = tf.Variable(_init_biases(out_dim), name=biases_name)
    output_layer = activation_func(tf.add(tf.matmul(input_layer, weights), biases))

    return output_layer, weights, biases


def rmse_loss(actual, predicted):
    return tf.reduce_mean(tf.square(tf.subtract(actual, predicted)))


def _construct_network(input_layer, size_of_each_layer, activation_func, prefix=''):
    _name_template = (prefix + '_{}_{:03d}') if prefix else '{}_{:03d}'
    weight_dims = zip(size_of_each_layer, size_of_each_layer[1:])
    layers = [input_layer]
    weights = []

    for i, weight_dim in enumerate(weight_dims):
        _activations, _weights, _biases = _construct_layer(layers[-1], weight_dim, activation_func, _name_template=_name_template, _iter_lbl=i)
        layers.append(_activations)
        weights.append((_weights, _biases))

    return (layers, weights)


def _gaussian_log_likelihood(observations, mean, log_squared_sigma):
    '''CHECK THIS MATH'''
    squared_sigma = np.exp(log_squared_sigma)

    first_term = -0.5 * N * np.reduce_sum(log_squared_sigma)
    second_term = -0.5 * tf.divide(tf.squared_difference(observations, mean), 2 * squared_sigma)

    return tf.add(first_term, second_term)


def _kl_zero_mean_identity_cov_gaussian(mean, log_squared_sigma):
    squared_sigma = np.exp(log_squared_sigma)
    return -0.5 * tf.reduce_sum(1 + log_squared_sigma - mean - squared_sigma)


def _kl_zero_mean_identity_cov_gaussian(mean, log_squared_sigma):
    squared_sigma = np.exp(log_squared_sigma)
    return -0.5 * tf.reduce_sum(1 + log_squared_sigma - mean - squared_sigma)


class Autoencoder(object):

    def __init__(self, num_input_dim, num_latent_dim, optimizer, encoding_network_hidden_layers, loss_func_dict={'reconstruction_loss': rmse_loss}):
        ### TODO: Figure out where to initalize session
        '''Simple class to wrap code constructing an autoencoder (fully connected)

        Args:
            num_input_dim: (int) the dimension of the inputs
            num_latent_dim: (int) the dimension of the innermost latent layer
            encoding_network_hidden_layers: (iter of int) dimensions of in-between layers
        '''
        self._N = num_input_dim
        self._M = num_latent_dim
        self._hidden_layer_dims = encoding_network_hidden_layers

        self._network = self._construct_network(self._N, self._M, self._hidden_layer_dims) # B/c of demands of subclass, return dictionary?
        self._loss = self._create_loss_func(self._network, **loss_func_dict)
        self._optimizer = optimizer.minimize(self._loss)

        self.input_layer = self._network['input_layer']
        self.latent_layer = self._network['latent_layer']

    def _construct_network(N, M, hidden_layer_dims, activation_func):
        size_of_each_layer = [N] + hidden_layer_dims + [M] + hidden_layer_dims + [N]
        input_layer = tf.placeholder(tf.float32, [None, N])

        layers, weights = _construct_network(input_layer, size_of_each_layer)

        input_layer = layers[0]
        latent_layer = layers[1 + len(hidden_layer_dims) + 1]

        return {'all_layers': layers, 'all_weights': weights, 'input_layer': input_layer, 'latent_layer': latent_layer}

    def _create_loss_func(network, reconstruction_loss):
        input_layer = network['input_layer']
        output_layer = network['prediction_layer']
        return reconstruction_loss(input_layer, prediction_layer)
           
    def partial_fit(self, X, y=None):
        feed_dict = {self.input_layer: X}
        opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict=feed_dict)
        return opt, cost

    def transform(self, X, y=None):
        feed_dict = {self.input_layer: X}
        transformed = self.sess.run((self.latent_layer, ), feed_dict=feed_dict)
        return transformed


class VariationalAutoencoder(object):

    def __init__(self, num_input_dim, num_latent_dim, encoding_network_hidden_layers, loss_func_dict={'reconstruction_loss': _gaussian_log_likelihood, 'kl_loss': _kl_zero_mean_identity_cov_gaussian}):
        super(VariationalAutoencoder, self).__init__(**params)

    def _construct_network(N, M, hidden_layer_dims, activation_func):
        encoder_structure = [N] + hidden_layer_dims + [M]
        decoder_structure = [M] + hidden_layer_dims + [N]

        input_layer = tf.placeholder(tf.float32, [None, N])
        epsilon_noise = tf.placeholder(tf.float32, [None, M])

        # Create encoder networks
        encoder_mean_layers, encoder_mean_weights = _construct_network(input_layer, encoder_structure, activation_func, prefix='encoder_mean')
        encoder_log_squared_sigma_layers, encoder_log_squared_sigma_weights = _construct_network(input_layer, encoder_structure, activation_func, prefix='encoder_log_squared_sigma')

        # Define how latent samples are generated
        latent_mean_layer = encoder_mean_layers[-1]
        latent_log_squared_sigma_layer = encoder_log_squared_sigma_layers[-1]
        latent_samples = tf.add(latent_mean_layer, tf.multiply(tf.exp(latent_log_squared_sigma_layer), epsilon_noise))

        # Create decoder networks
        decoder_mean_layers, decoder_mean_weights = _construct_network(latent_samples, decoder_structure, activation_func, prefix='decoder_mean')
        decoder_log_squared_sigma_layers, decoder_log_squared_sigma_weights = _construct_network(latent_samples, decoder_structure, activation_func, prefix='decoder_log_squared_sigma')

        return {'input_layer': input_layer, 'epsilon_noise': epsilon_noise, 'latent_mean_layer': latent_mean_layer, 'latent_log_squared_sigma_layer': latent_log_squared_sigma_layer,
                'latent_samples': latent_samples, 'prediction_mean_layer': decoder_mean_layers[-1],
                'prediction_log_squared_sigma_layer': decoder_log_squared_sigma_layers[-1],
                'all_layers': (encoder_mean_layers, encoder_log_squared_sigma_layers, decoder_mean_layers, decoder_log_squared_sigma_layers),
                'all_weights': (encoder_mean_weights, encoder_log_squared_sigma_weights, decoder_mean_weights, decoder_log_squared_sigma_weights)}

    def _create_loss_func(network, reconstruction_loss, kl_loss):
        # First compute reconstruction log likelihood of x given the generated z
        observations = network['input_layer']
        posterior_mean = network['prediction_mean_layer']
        posterior_log_squared_sigma = network['prediction_log_squared_sigma_layer']
        reconstruction_cost = reconstruction_loss(observations, posterior_mean, posterior_log_squared_sigma)

        # Then, construct the divergence of variational distribution against prior of latent space
        latent_mean = network['latent_mean_layer']
        latent_log_squared_sigma = network['latent_log_squared_sigma_layer']
        kl_cost = kl_loss(latent_mean, latent_log_squared_sigma)

        # Sum them together
        overall_cost = tf.add(reconstruction_cost, kl_cost)
        return overall_cost

    def _generate_epsilon_noise(self, X):
        return np.random.normal(size=X.shape)

    def partial_fit(self, X, y=None):
        epsilon_noise = self._generate_epsilon_noise(X)
        feed_dict = {self.input_layer: X, self._network['epsilon_noise']: epsilon_noise}
        opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict=feed_dict)
        return opt, cost

    def transform(self, X, y=None):
        epsilon_noise = self._generate_epsilon_noise(X)
        feed_dict = {self.input_layer: X, self._network['epsilon_noise']: epsilon_noise}
        transformed, samples = self.sess.run((self._network['latent_mean_layer'], self._network['latent_samples']), feed_dict=feed_dict)
        return transformed, samples


class GenerativeSemiSupervisedAutoencoder(object):

    def __init__(self, num_input_dim, num_latent_dim, encoding_network_hidden_layers, loss_func_dict={'reconstruction_loss': _gaussian_log_likelihood, 'kl_loss': _kl_zero_mean_identity_cov_gaussian}):
        super(VariationalAutoencoder, self).__init__(**params)

    def _construct_network(N_in, N_out, M, hidden_layer_dims, activation_func):
        encoder_structure = [N_in] + hidden_layer_dims + [M]
        decoder_structure = [M] + hidden_layer_dims + [N_out]

        input_layer = tf.placeholder(tf.float32, [None, N_in])
        epsilon_noise = tf.placeholder(tf.float32, [None, M])

        # Create encoder networks
        encoder_mean_layers, encoder_mean_weights = _construct_network(input_layer, encoder_structure, activation_func, prefix='encoder_mean')
        encoder_log_squared_sigma_layers, encoder_log_squared_sigma_weights = _construct_network(input_layer, encoder_structure, activation_func, prefix='encoder_log_squared_sigma')

        # Define how latent samples are generated
        latent_mean_layer = encoder_mean_layers[-1]
        latent_log_squared_sigma_layer = encoder_log_squared_sigma_layers[-1]
        latent_samples = tf.add(latent_mean_layer, tf.multiply(tf.exp(latent_log_squared_sigma_layer), epsilon_noise))

        # Create decoder networks
        decoder_mean_layers, decoder_mean_weights = _construct_network(latent_samples, decoder_structure, activation_func, prefix='decoder_mean')
        decoder_log_squared_sigma_layers, decoder_log_squared_sigma_weights = _construct_network(latent_samples, decoder_structure, activation_func, prefix='decoder_log_squared_sigma')

        return {'input_layer': input_layer, 'epsilon_noise': epsilon_noise, 'latent_mean_layer': latent_mean_layer, 'latent_log_squared_sigma_layer': latent_log_squared_sigma_layer,
                'latent_samples': latent_samples, 'prediction_mean_layer': decoder_mean_layers[-1],
                'prediction_log_squared_sigma_layer': decoder_log_squared_sigma_layers[-1],
                'all_layers': (encoder_mean_layers, encoder_log_squared_sigma_layers, decoder_mean_layers, decoder_log_squared_sigma_layers),
                'all_weights': (encoder_mean_weights, encoder_log_squared_sigma_weights, decoder_mean_weights, decoder_log_squared_sigma_weights)}

    def _create_loss_func(network, reconstruction_loss, kl_loss):
        # First compute reconstruction log likelihood of x given the generated z
        observations = network['input_layer']
        posterior_mean = network['prediction_mean_layer']
        posterior_log_squared_sigma = network['prediction_log_squared_sigma_layer']
        reconstruction_cost = reconstruction_loss(observations, posterior_mean, posterior_log_squared_sigma)

        # Then, construct the divergence of variational distribution against prior of latent space
        latent_mean = network['latent_mean_layer']
        latent_log_squared_sigma = network['latent_log_squared_sigma_layer']
        kl_cost = kl_loss(latent_mean, latent_log_squared_sigma)

        # Sum them together
        overall_cost = tf.add(reconstruction_cost, kl_cost)
        return overall_cost

    def _generate_epsilon_noise(self, X):
        return np.random.normal(size=X.shape)

    def partial_fit(self, X, y=None):
        epsilon_noise = self._generate_epsilon_noise(X)
        feed_dict = {self.input_layer: X, self._network['epsilon_noise']: epsilon_noise}
        opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict=feed_dict)
        return opt, cost

    def transform(self, X, y=None):
        epsilon_noise = self._generate_epsilon_noise(X)
        feed_dict = {self.input_layer: X, self._network['epsilon_noise']: epsilon_noise}
        transformed, samples = self.sess.run((self._network['latent_mean_layer'], self._network['latent_samples']), feed_dict=feed_dict)
        return transformed, samples
