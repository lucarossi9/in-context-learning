import math
import torch
import numpy as np


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    """
    Return the data sampler
    :param data_name: Name of the data (gaussian for linear regression, tensor for tensor PCA, mixtures for mixtures of Bernoulli)
    :param n_dims: number of dims of the problems
    :param kwargs: additional kwargs
    :return: the sampler
    """
    names_to_classes = {
        "gaussian": GaussianSampler,
        "tensor": TensorSampler,
        "mixtures": MixturesSampler,
    }
    if data_name in names_to_classes:  # the sampler exists
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)  # create data-sampler
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    """
    Used for the skewed covariance experiment.
    :param eigenvalues: the eigs of the covariance matrix.
    :param normalize: if True normalize by the norm.
    :return: the covariance matrix.
    """
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)  # construct covariance matrix
    if normalize:
        norm_subspace = torch.sum(eigenvalues ** 2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


def generate_corr(n_points, dims, b_size):
    """
    Generates the correlation between some of the columns of the design matrix X
    :param n_points: number of points.
    :param dims: number of dimensions.
    :param b_size: batch size
    :return: Randomly genearated points from a normal distribution with correlation.
    """
    # Generate a random matrix of dimension (m, n) with normally distributed entries

    corr = np.eye(dims)
    corr[0, 1] = 0.7
    corr[1, 0] = 0.7
    corr[5, 2] = 0.8
    corr[2, 5] = 0.8
    mean = np.zeros(dims)
    B = np.random.multivariate_normal(mean, corr, size=(b_size, n_points))

    return torch.from_numpy(B).float()


class GaussianSampler(DataSampler):
    """
    Class used to sample from gaussian the points
    """
    def __init__(self, n_dims, bias=None, scale=None):
        """
        Init method for gaussian sampler.
        :param n_dims: number of dimensions.
        :param bias: Bias if want to generate from a biased normal.
        :param scale: scale if want to generate from a scaled normal.
        """
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, flag_corr=False):
        """
        Sample xs
        :param n_points: number of points.
        :param b_size: batch size.
        :param n_dims_truncated: number of dimensions.
        :param seeds: seeds if want to fix them.
        :param flag_corr: if True add correlation between features.
        :return:
        """
        if flag_corr:
            # Correlation in the sampling
            if seeds is None:
                xs_b = generate_corr(n_points, self.n_dims, b_size)
        else:  # no correlation between columns
            if seeds is None:
                # simply generate using randn
                xs_b = torch.randn(b_size, n_points, self.n_dims)
            else:
                xs_b = torch.zeros(b_size, n_points, self.n_dims)
                generator = torch.Generator()  # Creates and returns a generator object that manages the state
                # of the algorithm which produces pseudo random numbers
                assert len(seeds) == b_size
                for i, seed in enumerate(seeds):
                    generator.manual_seed(seed)
                    xs_b[i] = torch.randn(n_points, self.n_dims,
                                          generator=generator)  # use a different seed for each batch
        if self.scale is not None:
            eigenvalues = 1 / (torch.arange(self.n_dims) + 1)
            scale = sample_transformation(eigenvalues, normalize=False)
            self.scale = scale
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            # truncate all points using the dimension
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class TensorSampler(DataSampler):
    """
    Class for sampling for Tensor PCA
    """
    def __init__(self, n_dims, bias=None, scale=None):
        """
        Init method for gaussian sampler.
        :param n_dims: number of dimensions.
        :param bias: Bias if want to generate from a biased normal.
        :param scale: scale if want to generate from a scaled normal.
        """
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_tensors(self, b_size, lambda_):
        """
        Sample tensors T generated from vector v
        :param b_size: batch size.
        :param lambda_: lambda param, controls noise to signal ratio.
        :return: T and v the tensor and the vector used to generate it.
        """
        # tensor shape has to be of size (b_size, d, d, ..., d)
        tensor_shape = (b_size, self.n_dims, self.n_dims, self.n_dims, self.n_dims)

        # build vector of random noise
        W = torch.randn(tensor_shape)

        # build signal v
        v = torch.randn((b_size, self.n_dims))

        # normalize such that its norm is ||v||_2 = \sqrt(N)
        for i in range(b_size):
            v[i, :] = v[i, :] / torch.norm(v[i, :]) * torch.sqrt(torch.tensor(self.n_dims))

        # signal is a tensor of shape (b_size, d, d, ..., d) containing signal[i,:] = v[i] \tensor v[i] \tensor ...,
        # v[i]
        signal = torch.zeros(tensor_shape)
        for i in range(b_size):
            signal_tmp = v[i, :].T
            signal[i, :, :, :] = torch.einsum('i,j,k,h->ijkh', signal_tmp, signal_tmp, signal_tmp, signal_tmp)
        # generate T
        T = W + lambda_ / (self.n_dims ** ((4 - 1) / 2)) * signal
        T = T.reshape(b_size, self.n_dims ** 3, -1)
        return T, v


class MixturesSampler(DataSampler):
    """
    Class of generating point for mixtures of Bernoulli
    """
    def sample_from_mixtures_bernoulli(self, b_size, n_clusters, n_points):
        """
        Generate points from mixtures of bernoulli
        :param b_size: batch size.
        :param n_clusters: number of clusters.
        :param n_points: number of points.
        :return: batch of points, bernoulli means for each element of the batch, probabilities for each element of the batch
        """

        # sample Bernoullis params uniformly at random
        bernoulli_params = torch.rand(b_size, self.n_dims, n_clusters)

        # generate probability parameters, each batch should have a n_clusters
        probas_params = torch.rand(b_size, n_clusters)
        # normalize
        probas_params = probas_params / torch.sum(probas_params, dim=1).unsqueeze(1)
        # probas_params has shape (b_size, n_clusters)

        # assign points to clusters according to probas_params --> cluster assign
        # is a n_points dimensional vector
        cluster_assign = torch.multinomial(probas_params, n_points, replacement=True)
        # cluster_assign is a matrix of shape (b_size, n_points)

        xs = torch.zeros(b_size, n_points, self.n_dims)

        sampling_matrix = torch.zeros((self.n_dims, n_points))
        for i in range(b_size):
            bernoulli_param = bernoulli_params[i, :, :]
            # find to which cluster the points are assigned
            batch_assign = cluster_assign[i, :]
            # sample now according to the bernoulli with that parameter.
            sampling_matrix[:, torch.arange(n_points)] = bernoulli_param[:, batch_assign]
            points_in_batch = torch.bernoulli(sampling_matrix)
            xs[i, :, :] = points_in_batch.T
        return xs, bernoulli_params, probas_params
