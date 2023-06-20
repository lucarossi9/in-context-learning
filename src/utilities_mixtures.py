import torch
import torch.nn as nn
import numpy as np


def sample_from_mixtures_bernoulli(b_size, n_clusters, n_points, n_dims):
    """
    Samples from a mixture of beroulli b_size batches, each with n_points each with n_dims
    :param b_size: Batch size.
    :param n_clusters: number of clusters.
    :param n_points: number of points.
    :param n_dims: number of dimensions.
    :return: xs, bernoulli_means, bernoulli_probas
    """
    # generate the parameters of the bernoulli, each batch has n_clusters
    # dimensional vectors
    bernoulli_params = torch.rand(b_size, n_dims, n_clusters)
    # bernoulli_params = bernoulli_params / torch.sum(bernoulli_params, dim=1).unsqueeze(1)

    # generate probability parameters, each batch should have a n_clusters
    # dimensional vector
    probas_params = torch.rand(b_size, n_clusters)
    probas_params = probas_params / torch.sum(probas_params, dim=1).unsqueeze(1)
    # probas_params has shape (b_size, n_clusters)

    # assign points to clusters according to probas_params --> cluster assign
    # is a n_points dimensional vector
    cluster_assign = torch.multinomial(probas_params, n_points, replacement=True)
    # cluster_assign is a matrix of shape (b_size, n_points)

    xs = torch.zeros(b_size, n_points, n_dims)

    sampling_matrix = torch.zeros((n_dims, n_points))
    for i in range(b_size):
        bernoulli_param = bernoulli_params[i, :, :]
        batch_assign = cluster_assign[i, :]
        sampling_matrix[:, torch.arange(n_points)] = bernoulli_param[:, batch_assign]
        points_in_batch = torch.bernoulli(sampling_matrix)
        xs[i, :, :] = points_in_batch.T
    return xs, bernoulli_params, probas_params


class LossLikelihoodBernoulli(nn.Module):
    """
    Computes the -loglikelihood Bernoulli mixtures problem
    """
    def __init__(self):
        super(LossLikelihoodBernoulli, self).__init__()

    def forward(self, xs, means, probas):
        """
        Forward
        :param xs: the points.
        :param means: The means of the Bernoulli.
        :param probas: The probabilities of the Bernoulli.
        :return: the loss function.
        """
        batch_size, n_points, dims = xs.shape
        _, _, n_clusters = means.shape

        xs_reshaped = xs.view(batch_size, n_points, dims, 1)

        batch_xs_transpose = xs_reshaped.permute(0, 2, 1, 3).reshape(batch_size, dims, -1)
        batch_mean_transpose = means.permute(0, 2, 1)

        log_likelihood = torch.log(torch.clamp(batch_mean_transpose, min=1e-8)
                                   ) @ batch_xs_transpose + torch.log(torch.clamp(1 - batch_mean_transpose, min=1e-8)
                                                                      ) @ (1 - batch_xs_transpose)

        likelihood_bernoulli = torch.exp(log_likelihood)

        weighted_likelihood = likelihood_bernoulli.transpose(1, 2) * probas.unsqueeze(1)
        log_likelihood_batch = torch.sum(torch.log(torch.clamp(
            torch.sum(weighted_likelihood, dim=2), min=1e-8)), dim=1)
        loss = -torch.mean(log_likelihood_batch)

        return loss


def random_init_mixtures(b_size, n_dims, n_clusters):
    """
    Random initialization bernoulli mixtures problem
    :param b_size: size of the batch.
    :param n_dims: number of dimensions.
    :param n_clusters: number of clusters.
    :return: random initialization of means and probabilities.
    """
    current_mean = torch.rand(b_size, n_dims, n_clusters, requires_grad=True)
    current_prob = torch.rand(b_size, n_clusters, requires_grad=True)
    current_prob = current_prob / torch.sum(current_prob, dim=1).unsqueeze(1)
    current_prob = current_prob.clone().detach().requires_grad_(True)
    return current_mean, current_prob


def projection_simplex_sort(v, z=1):
    """
    project v onto the simplex of radius z
    :param v: the param to project.
    :param z: the radius of the simplex.
    :return:
    """
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return torch.tensor(w)


def PGD_for_mixtures(current_prob, current_mean, lr, n_steps, xs, loss):
    """
    PGD for Bernoulli mixtures
    :param current_prob: initial iterate for the prob.
    :param current_mean: Initial iterate for the mean.
    :param lr: the learning rate.
    :param n_steps: the number of steps.
    :param xs: the inputs.
    :param loss: the loss function.
    :return: current_mean and current_prob: the updated means and probas after n_steps of PGD
    """
    for i in range(n_steps):
        loss_value = loss(xs, current_mean, current_prob)
        loss_value.backward(retain_graph=True)

        # step of GD for prob
        current_prob.data = current_prob.data - lr * current_prob.grad.data
        current_prob.data = torch.stack([projection_simplex_sort(row.numpy()) for row in current_prob.data])

        # step of projection
        current_mean.data = current_mean.data - lr * current_mean.grad.data
        current_mean.data = torch.clamp(current_mean.data, 0, 1)

        current_mean.grad.data.zero_()
        current_prob.grad.data.zero_()

    return current_mean, current_prob


def EM_for_mixtures(current_probas, current_means, n_steps, xs, loss):
    """
    EM for mixtures of Bernoulli problem.
    :param current_probas: The current iterate for the probabilities params.
    :param current_means: The current iterate for the means params.
    :param n_steps: The number of steps.
    :param xs: The inputs.
    :param loss: The loss function.
    :return: current_mean and current_probas which represent the updated params after n_steps of EM
    """
    # E-Step
    b_size, n_points, dims = xs.shape
    _, _, n_clusters = current_means.shape
    xs_reshaped = xs.view(b_size, n_points, dims, 1)
    for i in range(n_steps):
        batch_xs_transpose = xs_reshaped.permute(0, 2, 1, 3).reshape(b_size, dims, -1)
        batch_mean_transpose = current_means.permute(0, 2, 1)
        log_likelihood = torch.log(torch.clamp(batch_mean_transpose, min=1e-12)
                                   ) @ batch_xs_transpose + torch.log(torch.clamp(1 - batch_mean_transpose, min=1e-12)
                                                                      ) @ (1 - batch_xs_transpose)
        likelihood_bernoulli = torch.exp(log_likelihood)
        weighted_likelihood = likelihood_bernoulli.transpose(1, 2) * current_probas.unsqueeze(1)
        responsabilities = weighted_likelihood / torch.sum(weighted_likelihood, dim=2).unsqueeze(2)

        # M-step
        N_k = torch.sum(responsabilities, dim=1)
        current_means_copy = current_means
        current_means = (xs.transpose(1, 2) @ responsabilities) / N_k.unsqueeze(1)
        nan_mask = torch.isnan(current_means)
        # current_means[nan_mask] = current_means_copy[nan_mask]

        current_probas = N_k / (n_points * torch.ones(b_size, n_clusters))

    return current_means, current_probas


def compute_number_k_clusters(b_size, n_clusters, n_dims, n_points, model, lr, n_batches):
    """
    The funtion compute for each algo the number of k<K clusters solutions.
    :param b_size: The batch size.
    :param n_clusters: The number of clusters.
    :param n_dims: The dimension d.
    :param n_points: The number of points.
    :param model: The model (Transformer).
    :param lr: The lr for PGD.
    :param n_batches: The num of batches to use to evaluate.
    :return: dic of results.
    """
    k_clusters_GD = 0
    k_clusters_EM = 0
    k_clusters_transformers = 0
    average_loss_GD = 0
    average_loss_EM = 0
    average_loss_transformers = 0
    loss = LossLikelihoodBernoulli()
    for i in range(n_batches):
        # sample batch
        xs, _, _ = sample_from_mixtures_bernoulli(b_size, n_clusters, n_points, n_dims)

        # model statistics
        predicted_means, predicted_probas = model(xs, n_clusters)
        average_loss_transformers += float(loss(xs, predicted_means, predicted_probas))
        count = torch.sum(predicted_probas < 0.1/n_clusters, dim=1)
        k_clusters_transformers += torch.sum(count > 0).item()

        # GD statistics
        current_means, current_probas = random_init_mixtures(b_size, n_dims, n_clusters)
        predicted_means, predicted_probas = PGD_for_mixtures(current_probas, current_means, lr , 10, xs, loss)
        average_loss_GD += float(loss(xs, predicted_means, predicted_probas))
        count = torch.sum(predicted_probas < 0.1/n_clusters, dim=1)
        k_clusters_GD += torch.sum(count > 0).item()

        # EM statistics
        current_means, current_probas = random_init_mixtures(b_size, n_dims, n_clusters)
        predicted_means, predicted_probas = EM_for_mixtures(current_probas, current_means, 1000, xs, loss)
        average_loss_EM += float(loss(xs, predicted_means, predicted_probas))
        count = torch.sum(predicted_probas < 0.1/n_clusters, dim=1)
        k_clusters_EM += torch.sum(count > 0).item()

        del xs

    num_tasks = n_batches * b_size

    # compose dict of results
    dic = {}
    dic["proportions"] = {"transformers": k_clusters_transformers / num_tasks, "GD": k_clusters_GD / num_tasks,
                          "EM": k_clusters_EM / num_tasks}
    dic["losses"] = {"transformers": average_loss_transformers / num_tasks, "GD": average_loss_GD / num_tasks,
                          "EM": average_loss_EM / num_tasks}
    return dic
