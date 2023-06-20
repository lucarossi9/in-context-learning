import math
import torch
from torch.distributions.laplace import Laplace


def squared_error(ys_pred, ys):
    """
    Squared error loss
    :param ys_pred: predictions
    :param ys: labels
    :return: squared error loss
    """
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    """
    Means squared error loss
    :param ys_pred: predictions
    :param ys: labels
    :return: mean squared error loss
    """
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    """
    Accuracy metric
    :param ys_pred: predictions
    :param ys: labels
    :return: accuracy.
    """
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    """
    Cross entropy loss.
    :param ys_pred: predictions
    :param ys: labels
    :return: cross entropy loss.
    """
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    """
    Class representing a task
    """

    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, delta=1, mu=0):
        """
        Init for the task class.
        :param n_dims: number of dimensions.
        :param batch_size: batch size.
        :param pool_dict: if we already have the function parameters.
        :param seeds: if we want to fix the seeds.
        :param delta: scaling of the variance w \sim (mu, \deltaI)
        :param mu: mean w \sim (mu, \deltaI)
        """
        self.n_dims = n_dims  # dimension of the (parameter) of the function
        self.b_size = batch_size  # batch size of the prompt
        self.pool_dict = pool_dict  # dict containing the parameters of the function
        self.seeds = seeds  # vector of seeds
        self.delta = delta
        self.mu = mu
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, delta, mu):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs):
    """
    The function return the task sampler
    :param task_name: name of the task.
    :param n_dims: number of dimensions.
    :param batch_size: size of the batch.
    :param pool_dict: parameters of the function.
    :param num_tasks: number of tasks if fixed they will be sampled from a fixed number each time.
    :param kwargs: additional args.
    :return:
    """
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "laplace_linear_regression": NoisyLaplaceLinearRegression,
        "noisy_laplace_linear_regression": NoisyLaplaceLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
    }  # dict mapping task names to models' classes

    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]  # extract the task
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)  # dict with ground-truth params
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args)  # returns the tasks
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):
    """
    Linear regression task
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, delta=1, mu=0):
        """
        Init for the linear regression task
        :param n_dims: number of dimension.
        :param batch_size: batch size.
        :param pool_dict: parameters of the function.
        :param seeds: seeds to generate in a reproducible way.
        :param scale: scale for the params of the function.
        :param delta: w \sim N(mu, delta I)
        :param mu: w \sim N(mu, delta I)
        """
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds, delta, mu)
        self.scale = scale

        if pool_dict is None and seeds is None:  # if w not set, we set it randomly
            self.w_b = mu + delta * torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            # if we have seeds, we loop over the batches and generate one w at the time for each el in the batch
            # with the given seed
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = mu + delta * torch.randn(self.n_dims, 1, generator=generator)
        else:  # no seed is set
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            # in this case we permute and take first batch size
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        """
        Find predictions given xs_b
        :param xs_b: batch of inputs
        :return: ys_b
        """
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, delta, mu, **kwargs):  # ignore extra args
        """
        Generate params of the function
        :param n_dims: number of dims.
        :param num_tasks: numb of tasks.
        :param delta: w \sim N(mu, delta I)
        :param mu: w \sim N(mu, delta I)
        :param kwargs: additional args
        :return:
        """
        return {"w": mu + delta * torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearRegression(LinearRegression):
    """
    Sparse linear regression task
    """
    def __init__(
            self,
            n_dims,
            batch_size,
            pool_dict=None,
            delta=1,
            mu=0,
            seeds=None,
            scale=1,
            sparsity=3,
            valid_coords=None
    ):
        """
        Init for the sparse linear regression.
        :param n_dims: number of dims.
        :param batch_size: batch size
        :param pool_dict: parameters of the task.
        :param delta:  w \sim N(mu, delta I)
        :param mu:  w \sim N(mu, delta I)
        :param seeds: additional seeds for reproducibility
        :param scale: scale param.
        :param sparsity: number of non zero components in the weight vector.
        :param valid_coords: valid coordinates to zero out.
        """
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale, delta, mu
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()  # Mask of trues
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False  # mask is a mask of indices that don't have to be set to zero
            w[mask] = 0  # changed also in self.w_b since objects are passed by reference.

    def evaluate(self, xs_b):
        """
        Generate the predictions using the weights.
        :param xs_b: batch of inputs xs
        :return: the labels corresponding to ys.
        """
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LaplaceLinearRegression(Task):
    """
    Task where the laplace generates the weight
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, delta=1, mu=0):
        """
        Init for the sparse linear regression.
        :param n_dims: number of dims.
        :param batch_size: batch size
        :param pool_dict: parameters of the task.
        :param seeds: additional seeds for reproducibility
        :param scale: scale param.
        :param delta:  w \sim N(mu, delta I)
        :param mu:  w \sim N(mu, delta I)
        """
        super(LaplaceLinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds, delta, mu)
        self.scale = scale
        self.dist = Laplace(mu*torch.ones(self.n_dims), delta*torch.ones(self.n_dims))

        if pool_dict is None and seeds is None:  # if w not set, we set it randomly
            self.w_b = self.dist.sample(torch.Size([self.b_size])).view(self.b_size, self.n_dims, 1)  # delta is the
            # laplace param
        elif seeds is not None:
            # if we have seeds, we loop over the batches and generate one w at the time for each el in the batch
            # with the given seed
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            print("WARNING: Not using the generator")
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                self.w_b[i] = mu + delta * self.dist.sample().view(self.n_dims, 1)
        else:  # no seed is set
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            # in this case we permute and take first batch size
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        """
        Generate labels
        :param xs_b: input batch
        :return: ys_b
        """
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, delta, mu, **kwargs):  # ignore extra args
        return {"w": mu + delta * torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class NoisyLaplaceLinearRegression(LaplaceLinearRegression):  # inherits from linear regression
    """
    Class of noisy laplace linear regression
    """
    def __init__(
            self,
            n_dims,
            batch_size,
            pool_dict=None,
            seeds=None,
            scale=1,
            delta=1,
            mu=0,
            noise_std=0.1,
            renormalize_ys=False,
    ):
        """
        :param n_dims: number of dimensions.
        :param batch_size: batch size.
        :param pool_dict: parameter w
        :param seeds: can specify seeds for reproducibility.
        :param scale: scaling params.
        :param delta:  w \sim N(mu, delta I).
        :param mu:  w \sim N(mu, delta I).
        :param noise_std: std of label noise.
        :param renormalize_ys: renormalizing flag.
        """
        super(NoisyLaplaceLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale, delta, mu
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)  # access to method of the parent class
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std  # add noise
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()
        return ys_b_noisy


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):  # inherits from linear regression
    def __init__(
            self,
            n_dims,
            batch_size,
            pool_dict=None,
            seeds=None,
            scale=1,
            delta=1,
            mu=0,
            noise_std=0,
            renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale, delta, mu
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        """
        Generate labels
        :param xs_b: input batch
        :return: ys_b
        """
        ys_b = super().evaluate(xs_b)  # access to parent class
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std  # add noise
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class QuadraticRegression(LinearRegression):
    """
    Quadratic regression class
    """
    def evaluate(self, xs_b):
        """
        Generate predictions for quadratic regression
        :param xs_b: batch input.
        :return: pred.
        """
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b ** 2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegression(Task):
    def __init__(
            self,
            n_dims,
            batch_size,
            pool_dict=None,
            seeds=None,
            scale=1,
            hidden_layer_size=4,  # number of neurons of the hidden layer
            delta=1,
            mu=0,
    ):
        """
        :param n_dims: number of dimensions.
        :param batch_size: batch size.
        :param pool_dict: parameter w
        :param seeds: can specify seeds for reproducibility.
        :param scale: scaling params.
        :param hidden_layer_size:  dimension of hidden layer
        :param delta:  w \sim N(mu, delta I).
        :param mu:  w \sim N(mu, delta I).
        """
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds, delta, mu, False)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = mu + delta * torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = mu + delta * torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = mu + delta * torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = mu + delta * torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = mu + delta * torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = mu + delta * torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        """
        Evaluate labels of this parametric class
        :param xs_b: inputs
        :return: preds
        """
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)  # like sampling params of the network from N(0, 2/r)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, delta, mu, hidden_layer_size=4, **kwargs):
        """
        Generate params
        :param n_dims: number of dimensions
        :param num_tasks: number of tasks.
        :param delta: delta for normal.
        :param mu: mu for normal
        :param hidden_layer_size: n of neurons layer
        :param kwargs: additional args
        :return: pooldict
        """
        return {
            "W1": mu + delta * torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": mu + delta * torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


# did not change here for the weights
class DecisionTree(Task):
    """
    Class decision tree
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4, delta=1, mu=0):
        """
        init for decision tree
        :param n_dims: number of dimensions.
        :param batch_size: size of the batch.
        :param pool_dict: parameters.
        :param seeds: seeds for reproducibility.
        :param depth: depth of the tree.
        :param delta:
        :param mu:
        """
        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds, delta, mu, False)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)  # second dimension is the number of nodes
                # in the tree
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        """
        Create labels function.
        :param xs_b: input
        :return: labels.
        """
        # move to device
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)

        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):  # for each el in the batch
            xs_bool = xs_b[i] > 0  # coordinates of the element of the batch bigger than 0, xs_bool -> (n_points, dim)
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()  # zero vector of the same dimension as
            # the number of points in xs

            # doing it batch-wise
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]  # cur_coords for each batch
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions  # if cur_dec is true (1) go right

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
