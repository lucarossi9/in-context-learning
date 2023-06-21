import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.linear_model import Lasso
import warnings
from sklearn import tree
import xgboost as xgb
from base_models import NeuralNetwork, ParallelNetworks
import pymc as pm
import arviz as az
import math
import torch.nn.functional as F


def build_model(conf):
    """
    Build model from configuration.
    :param conf: The config information.
    :return: the model
    """
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,  # The maximum sequence length that this model might ever be used with
            n_embd=conf.n_embd,  # Dimensionality of the embeddings and hidden states.
            n_layer=conf.n_layer,  # Number of hidden layers in the Transformer encoder.
            n_head=conf.n_head,  # Number of attention heads for each attention layer in the Transformer encoder.
        )

    elif conf.family == "DNN":
        # Deep fully connected NN for different prompt len
        model = DeepNeuralNetwork(
            n_dims=conf.n_dims,
            hidden_size=conf.hidden_size,
            n_layers=conf.n_layers,
            max_len_prompt=conf.max_len_prompt
        )
    elif conf.family == "DNNSimplified":
        # Deep fully connected NN for fixed len prompt
        model = DeepNeuralNetworkSimplified(
            n_dims=conf.n_dims,
            hidden_size=conf.hidden_size,
            n_layers=conf.n_layers,
            max_len_prompt=conf.n_positions
        )
    elif conf.family == "transformer_tensor_pca":
        # Transformer for tensor PCA
        config = TransformerTensorPCAConfig(vocab_size=None,
                                            max_len=2 * conf.n_positions,
                                            num_heads=conf.n_head,
                                            num_blocks=conf.num_blocks,
                                            embed_dim=conf.n_embd,
                                            n_dims=conf.n_dims,
                                            initializer_range=conf.initializer_range)
        model = TransformerTensorPCA(config)
        # model.apply(model.initialize_weights)
    elif conf.family == "transformer_mixtures":
        # Transformer for mixtures
        config = TransformerMixtureConfig(vocab_size=None,
                                          max_len=2 * conf.n_positions,
                                          num_heads=conf.n_head,
                                          num_blocks=conf.num_blocks,
                                          embed_dim=conf.n_embd,
                                          n_dims=conf.n_dims,
                                          initializer_range=conf.initializer_range)
        model = TransformerMixture(config)
        # model.apply(model.initialize_weights)
    elif conf.family == "transformer_linear_regression":
        # Transformer for linear regression, not using GPT2 because not allowing for probing from hidden states.
        config = TransformerLinearRegressionConfig(vocab_size=None,
                                                   max_len=2 * conf.n_positions,
                                                   num_heads=conf.n_head,
                                                   num_blocks=conf.num_blocks,
                                                   embed_dim=conf.n_embd,
                                                   n_dims=conf.n_dims,
                                                   initializer_range=conf.initializer_range)
        model = TransformerLinearRegression(config)
    elif conf.family == "transformer_linear_regression_LSA":
        config = TransformerLinearRegressionLSAConfig(vocab_size=None,
                                                   max_len=2 * conf.n_positions,
                                                   num_heads=conf.n_head,
                                                   num_blocks=conf.num_blocks,
                                                   embed_dim=conf.n_embd,
                                                   n_dims=conf.n_dims,
                                                   initializer_range=conf.initializer_range)
        model = TransformerLinearRegression_LSA(config)
    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name, **task_kwargs):
    """
    For a given task, get the relevant baseline for the task.
    :param task_name: The name of the task.
    :param task_kwargs: The kwargs relative to the task.
    :return: The models.
    """
    task_to_baselines = {
        # the task is the key: linear_regression, sparse_linear_regression, relu_2nn_regression, decision_tree
        # for each of them, we compute the results of the baselines.

        "linear_regression": [
            (LeastSquaresModel, {"mu": task_kwargs["mu"], "delta": task_kwargs["delta"]}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
                                        (LeastSquaresModel, {}),
                                        (NNModel, {"n_neighbors": 3}),
                                        (AveragingModel, {}), ]
                                    + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
        "noisy_laplace_linear_regression": [
                                               (LeastSquaresModel, {}),
                                               (PosteriorMean, task_kwargs),
                                               (AdaptiveLasso, task_kwargs),
                                           ] + [(LassoModel, {"alpha": alpha}) for alpha in
                                                [1, 0.1, 0.01, 0.001, 0.0001]],
    }
    # **kwargs are the keywords arguments like {"n_neighbors": 3} for NNModel
    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class TransformerModel(nn.Module):
    """
    Transformer model from GPT2 family
    """

    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        """
        Init of the class
        :param n_dims: number of dimensions of the problem.
        :param n_positions: 2*n_positions = max len of the prompt.
        :param n_embd: hidden size of the transformer.
        :param n_layer: number of layers.
        :param n_head: number of heads.
        """
        super(TransformerModel, self).__init__()
        # create config of Transformer model
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            initializer_range=0.02,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,  # The dropout probability for all fully connected layers in the embeddings, encoder,
            # and pooler.
            embd_pdrop=0.0,  # dropout ratio for embedding
            attn_pdrop=0.0,  # The dropout ratio for the attention.
            use_cache=False,
        )
        # set name
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.n_dims = n_dims

        # set linear transformation from the input matrix to the input of the first decoder block
        self._read_in = nn.Linear(n_dims + 1, n_embd)
        self._backbone = GPT2Model(configuration)

        # linear transformation mapping the output of the structured Transformer to the predictions.
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod  # method that does not use any method of the class but it makes sense to add it to the class
    def _combine(xs_b, ys_b):
        """
        Interleaves the x's and the y's into a single sequence.
        :param xs_b: the batch of xs.
        :param ys_b: the batch of ys.
        :return: The input matrix M
        """
        bsize, points, dim = xs_b.shape  # bsize = batch_size, points = length of the prompt, dim = dimension of each x

        # my version, the one which does not require curriculum.
        xs_b_wide = torch.cat(
            (
                xs_b,
                torch.zeros(bsize, points, 1, device=ys_b.device)
            ),
            axis=2,
        )
        ys_b_wide = torch.cat(
            (
                xs_b,
                ys_b.view(bsize, points, 1),  # reshape ys_b as a tensor (bsize, points, 1)
            ),
            axis=2,
        )
        zs = torch.stack((xs_b_wide, ys_b_wide), dim=2)  # concatenate along third dimension: dim = d
        # zs is now of shape (bsize, points, dim*2)

        zs = zs.view(bsize, 2 * points, dim + 1)  # reshape like (bsize, 2 * points, dim)

        # # original version
        # ys_b_wide = torch.cat(
        #     (
        #         ys_b.view(bsize, points, 1),
        #         torch.zeros(bsize, points, dim - 1, device=ys_b.device),
        #     ),
        #     axis=2,
        # )
        # zs = torch.stack((xs_b, ys_b_wide), dim=2)
        # zs = zs.view(bsize, 2 * points, dim)

        # xs_b_wide = torch.cat(
        #     (
        #         xs_b,
        #         torch.zeros(bsize, points, 1, device=ys_b.device)
        #     ),
        #     axis=2,
        # )
        # ys_b_wide = torch.cat(
        #     (
        #         torch.zeros(bsize, points, dim, device=ys_b.device),
        #         ys_b.view(bsize, points, 1)
        #     ),
        #     axis=2,
        # )
        # zs = torch.stack((xs_b_wide, ys_b_wide), dim=2)
        # zs = zs.view(bsize, 2 * points, dim + 1)
        return zs

    def forward(self, xs, ys):
        """
        Compute the forward pass for the Transformer model.
        :param xs: the batch of xs.
        :param ys: the batch of ys.
        :return:
        """
        zs = self._combine(xs, ys)  # zs has shape (bsize, 2 * points, dim)
        embeds = self._read_in(zs)  # use a linear transformation to pass from dim to emb dimensions.
        output = self._backbone(inputs_embeds=embeds).last_hidden_state  # instead of passing input_ids you can choose
        # to directly pass an embedded representation. This is useful if you want more control over how to convert
        # input_ids indices into associated vectors than the model’s internal embedding lookup matrix.
        # basically we are computing the forward from the embedding (only decoder part).
        prediction = self._read_out(output)  # use a linear transformation to pass from emb dimensions to a scalar.
        # prediction has now shape (batch, n_points*2, 1)
        return prediction[:, ::2, 0]  # slice every two to get the predictions.


class NNModel:
    """
    N_neighbors model
    """

    def __init__(self, n_neighbors, weights="uniform"):
        """
        Init for the class of N nearest neighbors.
        :param n_neighbors: number of neighbors to average.
        :param weights: specify weights.
        """
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        """
        Forward method for this model.
        :param xs: batch of xs.
        :param ys: batch of ys.
        :param inds: indices of the points where we want to predict.
        :return: the predictions for the model.
        """
        # __call__ method will be called as --> name_model(xs, ys)
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i: i + 1]  # we try to infer point i in the batch
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()  # compute distances

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:  # more weights to closer points
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)  # because the first point does not have 3 neighbors
            ranks = dist.argsort()[:, :k]  # k smallest element for each batch
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]  # y and w are the y and the weights of the closest point
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank
# due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    """
    Implement the OLS estimator
    """

    def __init__(self, driver=None, **kwargs):
        """
        Init of the class.
        :param driver: The driver.
        :param kwargs: The kwargs for the model.
        """
        self.driver = driver
        self.name = f"OLS_driver={driver}"
        if "delta" in kwargs:
            self.delta = kwargs["delta"]
        else:
            self.delta = 1
        if "mu" in kwargs:
            self.mu = kwargs["mu"]
        else:
            self.mu = 0

    def __call__(self, xs, ys, inds=None):
        """
        Forward method for OLS
        :param xs: batch of points to be evaluated.
        :param ys: labels of points to be evaluated.
        :param inds: indices.
        :return: predictions of the OLS.
        """
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i: i + 1]  # we try to predict i-th from the previous i-th-1 elements
            batch_size, n_points, n_dims = train_xs.size()[0], train_xs.size()[1], train_xs.size()[2]

            if self.mu != 0 and n_points < n_dims:  # undetermined solution and mu!=0
                # if mu!=0 we compute different estimator: shifted OLS
                shift = torch.matmul(train_xs, (self.mu * torch.ones(batch_size, n_dims, 1)))
                # compute the weight using train_xs
                ws, _, _, _ = torch.linalg.lstsq(
                    train_xs, train_ys.unsqueeze(2) - shift, driver=self.driver
                )
                ws = ws + torch.ones_like(ws) * self.mu
            else:
                ws, _, _, _ = torch.linalg.lstsq(
                    train_xs, train_ys.unsqueeze(2), driver=self.driver
                )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)

    def add_mean_and_var(self, delta, mu):
        """
        Add the mean and variance to the model
        :param delta: the variance of the weight vector w_i
        :param mu: the mean of the weight vector w_i
        :return: None, simply add the attributes to the model.
        """
        self.delta = delta
        self.mu = mu


class AveragingModel:
    """
    Compute the baseline Averaging.
    """

    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        """
        Forward for the averaging estimator.
        :param xs: batch of xs.
        :param ys: batch of ys.
        :param inds: the indices of the points where we want the predictions.
        :return: the predictions.
        """
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i: i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            # compute weight vector
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            # use it to make predictions
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class LassoModel:
    """
    Class for the lasso model.
    """

    def __init__(self, alpha, max_iter=100000):
        """
        Init method for lasso model.
        :param alpha: the regularization parameter in sklearn
        :param max_iter: The max number of iters for lasso optimizatiion problem.
        """
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        """
        Forward method for lasso
        :param xs: batch of xs.
        :param ys: batch of ys.
        :param inds: The indices where we want the predictions.
        :return: the predictions made by Lasso.
        """
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):  # double for loop because difficult to vectorize
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )  # sklearn to initialize the lasso model

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise
                    # find weight w
                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i: i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1},
# opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    """
    Model implementing one step of GD for a NN
    """

    def __init__(
            self,
            model_class,  # to which model apply GD
            model_class_args,
            opt_alg="sgd",
            batch_size=1,
            num_steps=1000,
            lr=1e-3,
            loss_name="squared",
    ):
        """
        Init for GD model
        :param model_class: torch.nn model class
        :param model_class_args: a dict containing arguments for model_class
        :param opt_alg: can be 'sgd' or 'adam'
        :param batch_size: batch size
        :param num_steps: num steps
        :param lr: learning rate
        :param loss_name: name of the loss used
        """
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name} "

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        """
        Forward method for the class
        :param xs: batch of xs.
        :param ys: batch of ys.
        :param inds: indices where we want the predictions.
        :param verbose: verbose =True print additional informations
        :param print_step: every print_steps we print info
        :return:
        """
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args  # ys.shape[0] is the batch dimension
            )  # model parallel network because it can be parallelized
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i: i + 1], ys[:, i: i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()  # mask is a vector of false
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    # at each step we permute the points before i and we take a batch of them

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    """
    Decision Tree model class
    """

    def __init__(self, max_depth=None):
        """
        Init method for the class.
        :param max_depth: depth of the class.
        """
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        """
        Forward method for the class.
        :param xs: batch of xs.
        :param ys: batch of ys.
        :param inds: indices where we want the predictions.
        :return: the predictions.
        """
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i: i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    """
    Class XGBoost
    """

    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        """
        Forward method for the class.
        :param xs: batch of xs.
        :param ys: batch of ys.
        :param inds: indices where we want the predictions.
        :return: the predictions.
        """
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i: i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DeepNeuralNetwork(nn.Module):
    """
    The model implements the fully connected NN for in-context learning linear functions.
    """

    def __init__(self, n_dims, max_len_prompt, hidden_size=512, n_layers=8):
        """
        Init of the class
        :param n_dims: number of dimension d of the problem.
        :param max_len_prompt: Max length of the prompt.
        :param hidden_size: Hidden size of the hidden layers.
        :param n_layers: Number of layers.
        """
        super().__init__()
        self.n_dims = n_dims
        self.max_len_prompt = max_len_prompt
        self.flatten = nn.Flatten()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.block = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU())
        self.input_dim = 2 * self.n_dims * self.max_len_prompt
        # self.input_dim = (self.n_dims + 1)*self.max_len_prompt - 1
        self.linear_in = nn.Linear(self.input_dim, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, 1)
        self.name = f"DNN_layer={n_layers}_hidden_size={hidden_size}"

    def preprocess(self, xs_b, ys_b):
        """
        Implement the preprocessing for feeding the prompt to the model
        :param xs_b: Batch of inputs.
        :param ys_b: Batch of outputs.
        :return: Preprocessed inputs.
        """

        # xs is (batch_size, n_points, n_dims)
        # ys is (batch_size, n_points)

        batch_size, n_points, n_dims = xs_b.shape
        inputs = torch.zeros(batch_size, self.max_len_prompt, self.input_dim)  # second dimension one for each prompt
        # artificially created

        inputs[:, 0, :n_dims] = xs_b[:, 0, :]  # set the first prompt as the first x only
        for i in range(1, n_points):
            xs = xs_b[:, :i, :]  # select the xs before i
            ys = ys_b[:, :i]  # select the ys before i

            ys_wide = torch.cat(
                (
                    ys.view(batch_size, i, 1),  # reshape ys_b as a tensor (bsize, points_tmp, 1)
                    torch.zeros(batch_size, i, n_dims - 1, device=ys_b.device),
                ),
                axis=2,
            )  # ys_wide is now (batch_size, n_points, n_dims)
            zs = torch.stack((xs, ys_wide), dim=2)  # now zs = (batch_size, n_points = i, 2*n_dims)
            zs = zs.view(batch_size, 2 * i * n_dims)
            inputs[:, i, :(2 * i * n_dims)] = zs
            inputs[:, i, (2 * i * n_dims): (2 * i * n_dims + n_dims)] = xs_b[:, i:i + 1, :].view(batch_size, -1)

        # inputs[:, i ,:] is the v^{(i)} in the report
        return inputs.to(device=ys_b.device)

    def forward(self, xs, ys, inds=None):
        """
        Compute the forward for the fully connected neural network model.
        :param xs: Batch of xs.
        :param ys: Batch of ys.
        :param inds: indices where we want to make predictions.
        :return: predictions
        """
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        inputs = self.preprocess(xs, ys)  # input is now (batch_size, max_len_prompt, input_dim)
        inputs = self.linear_in(inputs)
        for i in range(self.n_layers):
            inputs = self.block(inputs)
        output = self.linear_out(inputs)

        return output[:, :, 0][:, inds]


class DeepNeuralNetworkSimplified(nn.Module):
    """
    Implement the class of fully connected deep NN for a fixed length
    """

    def __init__(self, n_dims, max_len_prompt, hidden_size=512, n_layers=8):
        """
        Init of the class
        :param n_dims: number of dimension d of the problem.
        :param max_len_prompt: Max length of the prompt.
        :param hidden_size: Hidden size of the hidden layers.
        :param n_layers: Number of layers.
        """
        super().__init__()
        self.n_dims = n_dims
        self.max_len_prompt = max_len_prompt  # length of the prompt: (x_1, y_1, x_2, ?) has len=2
        self.flatten = nn.Flatten()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.block = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU())
        self.input_dim = self.n_dims * self.max_len_prompt + (self.max_len_prompt - 1)
        # self.input_dim = (self.n_dims + 1)*self.max_len_prompt - 1
        self.linear_in = nn.Linear(self.input_dim, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, 1)
        self.name = f"DNN_layer={n_layers}_hidden_size={hidden_size}"

    def preprocess(self, xs_b, ys_b):
        """
        Combine inputs and outputs
        :param xs_b: Batch of xs.
        :param ys_b: Batch of ys.
        :return: a matrix of inputs, each row is a flattened vector of [(x_i, y_i)] pairs.
        """
        # xs is (batch_size, n_points, n_dims)
        # ys is (batch_size, n_points)
        batch_size, n_points, n_dims = xs_b.shape
        inputs = torch.zeros(batch_size, self.input_dim)  # second dimension one for each prompt of fixed len
        counter = 0
        for i in range(n_points - 1):
            inputs[:, counter:counter + self.n_dims] = xs_b[:, i, :].view(-1, self.n_dims)
            counter = counter + self.n_dims
            inputs[:, counter:counter + 1] = ys_b[:, i].view(-1, 1)
            counter += 1
        inputs[:, counter:counter + self.n_dims] = xs_b[:, n_points - 1, :].view(-1, self.n_dims)
        return inputs.to(device=ys_b.device)

    def forward(self, xs, ys, inds=None):
        """
        Compute the forward for the fully connected neural network model.
        :param xs: Batch of xs.
        :param ys: Batch of ys.
        :param inds: indices where we want to make predictions.
        :return: predictions
        """
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        inputs = self.preprocess(xs, ys)  # input is now (batch_size, max_len_prompt, input_dim)
        inputs = self.linear_in(inputs)
        for i in range(self.n_layers):
            inputs = self.block(inputs) + inputs
        output = self.linear_out(inputs)

        return output[:, 0]


class PosteriorMean:
    """
    The class implementing the posterior mean for Laplace prior
    """

    def __init__(self, delta, mu, sigma):
        """
        Init for the class.
        :param delta: The parameter controlling the variance of the laplace.
        :param mu: The mean of the laplace.
        :param sigma: The noise in the labels.
        """
        self.delta = delta
        self.mu = mu
        self.sigma = sigma
        self.name = f"posterior_mean_delta={delta}_sigma={sigma}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.

    def __call__(self, xs, ys, inds=None):
        """
        Forward for the class
        :param xs: Batch of xs.
        :param ys: Batch of ys.
        :param inds: indices of points where to make predictions.
        :return:
        """
        # inference done in cpu
        try:
            xs, ys = xs.cpu(), ys.cpu()
        except:
            print("evaluating on cpu")

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point
        n_dims = xs.shape[2]

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):  # double for loop because difficult to vectorize
                    train_xs, train_ys = xs[j, :i], ys[j, :i]  # points in batch j from 0 to i-1
                    train_xs = train_xs.detach().numpy()
                    train_ys = train_ys.detach().numpy()

                    # using Pymc to find the mean of the posterior distribution
                    basic_model = pm.Model()

                    with basic_model:
                        # Priors for unknown model parameters
                        beta = pm.Laplace("beta", mu=self.mu, b=self.delta, shape=n_dims)

                        # Expected value of outcome
                        mu = train_xs @ beta

                        # Likelihood (sampling distribution) of observations
                        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=self.sigma, observed=train_ys.reshape(-1))

                    with basic_model:
                        # draw 5000 posterior samples
                        trace = pm.sample(chains=2, cores=4, progressbar=False)
                        w_pred = az.summary(trace)["mean"].values

                    test_x = xs[j, i: i + 1]
                    y_pred = (test_x @ torch.tensor(w_pred, dtype=torch.float32))
                    pred[j] = y_pred[0]
                    print(f"batch {j}, point {i}")

            preds.append(pred)

        return torch.stack(preds, dim=1)


class AdaptiveLasso:
    """
    Class implementing AdaptiveLasso estimator.
    """

    def __init__(self, delta, mu, sigma, max_iter=100000):
        """
        Init method for the class
        :param delta: The parameter controlling the variance of the Laplace
        :param mu: The mean of the laplace.
        :param sigma: The noise of labels.
        :param max_iter: The max iters for lasso.
        """
        # the l1 regularizer gets multiplied by alpha.
        self.max_iter = max_iter
        self.delta = delta
        self.mu = mu
        self.sigma = sigma
        self.name = f"adaptive_lasso_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        """
        Forward for the class
        :param xs: Batch of xs.
        :param ys: Batch of ys.
        :param inds: indices of points where to make predictions.
        :return:
        """
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):  # double for loop because difficult to vectorize
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # compute optimum alpha for MAP
                    alpha = (self.sigma ** 2) / (self.delta * i)
                    clf = Lasso(
                        alpha=alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i: i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


####################################################################################################################
# custom Transformers

class Block(nn.Module):
    """
    Implement decoder block
    """

    def __init__(self, config):
        """
        Initialize decoder block
        :param config: Configurations
        """
        super().__init__()
        embed_dim = config.embed_dim
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(config)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(config.ff_dropout),
        )

    def forward(self, x):
        """
        Forward for decoder block (self-attention+feed_forward)
        :param x: inputs
        :return: predictions
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        # x = x + self.attn(self.ln1(x))
        return x


class MultiheadAttention(nn.Module):
    """
    The class implements the multihead self-attention layer.
    """

    def __init__(self, config):
        """
        Init for the class.
        :param config: the configurations.
        """
        super().__init__()
        embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        assert embed_dim % self.num_heads == 0, "invalid heads and embedding dimension configuration"

        # key components of SA layer
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.proj_dropout = nn.Dropout(config.ff_dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_len, config.max_len)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, zs):
        """
        Forward for multi-head self attention layer
        :param zs:
        :return:
        """
        batch_size = zs.size(0)
        seq_len = zs.size(1)
        # x.shape == (batch_size, seq_len, embed_dim)
        k_t = self.key(zs).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(zs).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(zs).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape == (batch_size, num_heads, seq_len, head_dim)

        # compute attention
        attn = torch.matmul(q, k_t) / math.sqrt(q.size(-1))
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        # apply masking
        mask = self.mask[:, :, :seq_len, :seq_len]
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.attn_dropout(attn)
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        # compute score
        attn = F.softmax(attn, dim=-1)
        y = torch.matmul(attn, v)
        # y.shape == (batch_size, num_heads, seq_len, head_dim)
        y = y.transpose(1, 2)
        # y.shape == (batch_size, seq_len, num_heads, head_dim)
        y = y.reshape(batch_size, seq_len, -1)
        # y.shape == (batch_size, seq_len, embed_dim)
        y = self.proj_dropout(self.proj(y))
        return y


class BlockNoMask(nn.Module):
    """
    Decoder block when no masking is applied used for mixture problem.
    """

    def __init__(self, config):
        """
        Initialize decoder block
        :param config: Configurations
        """
        super().__init__()
        embed_dim = config.embed_dim
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttentionNoMask(config)
        # self.ff = nn.Linear(config.embed_dim, config.embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(config.ff_dropout),
        )

    def forward(self, x):
        """
        Forward for decoder block (self-attention+feed_forward)
        :param x: inputs
        :return: predictions
        """
        x = x + self.attn(self.ln1(x))  # attn layer
        x = x + self.ff(self.ln2(x))  # feed forward layer
        return x


class MultiheadAttentionNoMask(nn.Module):
    """
    Multi-Head self-attention layer with no masking.
    """

    def __init__(self, config):
        super().__init__()
        embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        assert embed_dim % self.num_heads == 0, "invalid heads and embedding dimension configuration"
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.proj_dropout = nn.Dropout(config.ff_dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_len, config.max_len)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, zs):
        batch_size = zs.size(0)
        seq_len = zs.size(1)
        # x.shape == (batch_size, seq_len, embed_dim)
        k_t = self.key(zs).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(zs).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(zs).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape == (batch_size, num_heads, seq_len, head_dim)

        # no masking is now applied
        attn = torch.matmul(q, k_t) / math.sqrt(q.size(-1))
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        # mask = self.mask[:, :, :seq_len, :seq_len]
        # attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn = self.attn_dropout(attn)
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        y = torch.matmul(attn, v)
        # y.shape == (batch_size, num_heads, seq_len, head_dim)
        y = y.transpose(1, 2)
        # y.shape == (batch_size, seq_len, num_heads, head_dim)
        y = y.reshape(batch_size, seq_len, -1)
        # y.shape == (batch_size, seq_len, embed_dim)
        y = self.proj_dropout(self.proj(y))
        return y


class TransformerTensorPCAConfig:
    attn_dropout = 0.0
    embed_dropout = 0.0
    ff_dropout = 0.0

    def __init__(self, vocab_size, max_len, num_heads, num_blocks, embed_dim, **kwargs):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        # additional arguments if we want to set them
        for key, value in kwargs.items():
            setattr(self, key, value)


class TransformerTensorPCA(nn.Module):
    """
    Transformer used for tensor PCA
    """

    def __init__(self, config):
        """
        Init for the Transformer
        :param config: configuration for tensor PCA.
        """
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_dims = config.n_dims
        self.max_len = config.max_len
        self.initializer_range = config.initializer_range
        self.name = f"transformer_tensor_pca_n_layers={config.num_blocks}"

        # self.tok_embed = nn.Embedding(config.vocab_size, embed_dim)
        # self.pos_embed = nn.Embedding(config.max_len, embed_dim)
        self.proj = nn.Linear(self.n_dims, self.embed_dim)
        # add pos embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_len, self.embed_dim)
        )
        # apply non masked blocks
        self.blocks = nn.Sequential(
            *[BlockNoMask(config) for _ in range(config.num_blocks)]
        )
        self.ln = nn.LayerNorm(self.embed_dim)
        self._read_out = nn.Linear(self.embed_dim, self.n_dims)

    def initialize_weights(self, m):
        """
        Initialize the weights of the Transformers
        :param m: model
        :return: None
        """
        # initialize Linear layers with normal
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if m.bias is not None:
                m.bias.data.zero_()
        # same for embedding layers
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, Ts, vs):
        """
        Forward method for tensor PCA
        :param Ts:
        :param vs:
        :return:
        """
        # initial projection
        Ts = self.proj(Ts)
        # position embedding
        pos_embedding = self.pos_embed[:, :Ts.shape[1], :]

        output = self.blocks(Ts + pos_embedding)
        output = self.ln(output)
        prediction = self._read_out(output)
        # read the vector as the last row of each output matrix
        return prediction[:, -1, :]


class TransformerMixtureConfig:
    """
    Config for Transformer for mixtures
    """
    attn_dropout = 0.0
    embed_dropout = 0.0
    ff_dropout = 0.0

    def __init__(self, vocab_size, max_len, num_heads, num_blocks, embed_dim, **kwargs):
        """
        Init method for the Transformer model.
        :param vocab_size: size of the vocabulary.
        :param max_len: max length of the prompt.
        :param num_heads: number of heads in SA layer.
        :param num_blocks: number of decoder blocks.
        :param embed_dim: number of embedding dims.
        :param kwargs: further params.
        """
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        # additional arguments if we want to set them
        for key, value in kwargs.items():
            setattr(self, key, value)


class TransformerMixture(nn.Module):
    """
    Class of Transformers for mixtures
    """

    def __init__(self, config):
        """
        init of the Transformer
        :param config: configurations.
        """
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_dims = config.n_dims
        self.max_len = config.max_len
        self.initializer_range = config.initializer_range
        self.name = f"transformer_mixture_n_layers={config.num_blocks}"

        # initial projections
        self.proj = nn.Linear(self.n_dims, self.embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_len, self.embed_dim)
        )
        # apply blocks with no masking
        self.blocks = nn.Sequential(
            *[BlockNoMask(config) for _ in range(config.num_blocks)]
        )
        self.ln = nn.LayerNorm(self.embed_dim)
        # self.fc = nn.Linear(embed_dim, SimpleTransformerConfig.vocab_size)

        # the last hidden output will projected into a matrix of (n_points, dims) for each batch, used to read the k
        # means of the clusters
        self._read_out = nn.Linear(self.embed_dim, self.n_dims)

        # the last hidden output will projected into a matrix of (n_points, 1) for each batch, used to read the k
        # probabilities
        self.proj_probs = nn.Linear(self.embed_dim, 1)

        # to normalize probas and means.
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def initialize_weights(self, m):
        """
        Initialize the weights of the Transformers
        :param m: model
        :return: None
        """
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, xs, n_clusters):
        """
        Forward for mixture of Bernoullis problem.
        :param xs: The batch of inputs shape=(b_size, n_points, dims)
        :param n_clusters: number of clusters in mixture of bernoulli
        :return: the means and probabilities of the problem.
        """
        xs = self.proj(xs)
        # position embedding
        pos_embedding = self.pos_embed[:, :xs.shape[1], :]

        output = self.blocks(xs + pos_embedding)
        output = self.ln(output)
        # compute the means as the last n_cluster colums of the output matrix of each batch after a linear transform
        means = self._read_out(output)[:, -n_clusters:, :]
        means = self.sigmoid(means)  # to normalize to be in [0,1]^d
        means = means.transpose(1, 2)

        # compute the probas as the second to last n_clusters columns after a linear transformation.
        probas = self.proj_probs(output)[:, -2 * n_clusters:-n_clusters, -1]
        probas = self.softmax(probas)  # normalize so that they are positive and summing to 1

        return means, probas


class TransformerLinearRegressionConfig:
    """
    Config class for a custom Transformer used for linear regression.
    """
    attn_dropout = 0.0
    embed_dropout = 0.0
    ff_dropout = 0.0

    def __init__(self, vocab_size, max_len, num_heads, num_blocks, embed_dim, **kwargs):
        """
        Init method for the Transformer model.
        :param vocab_size: size of the vocabulary.
        :param max_len: max length of the prompt.
        :param num_heads: number of heads in SA layer.
        :param num_blocks: number of decoder blocks.
        :param embed_dim: number of embedding dims.
        :param kwargs: further params.
        """
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        # additional arguments if we want to set them
        for key, value in kwargs.items():
            setattr(self, key, value)


class TransformerLinearRegression(nn.Module):

    def __init__(self, config):
        """
        init of the Transformer
        :param config: configurations.
        """
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_dims = config.n_dims
        self.max_len = config.max_len
        self.initializer_range = config.initializer_range
        self.name = f"transformer_linear_regression_n_layers={config.num_blocks}"

        # self.tok_embed = nn.Embedding(config.vocab_size, embed_dim)
        # self.pos_embed = nn.Embedding(config.max_len, embed_dim)
        self.proj = nn.Linear(self.n_dims, self.embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_len, self.embed_dim)
        )
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.num_blocks)]
        )
        self.ln = nn.LayerNorm(self.embed_dim)
        # self.fc = nn.Linear(embed_dim, SimpleTransformerConfig.vocab_size)
        self._read_out = nn.Linear(self.embed_dim, 1)

    def initialize_weights(self, m):
        """
        Initialize the weights of the Transformers
        :param m: model
        :return: None
        """
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if m.bias is not None:
                m.bias.data.zero_()

    @staticmethod  # method that does not use any method of the class but it makes sense to add it to the class
    def _combine(xs_b, ys_b):
        """
        Interleaves xs and ys into a single input matrix for each batch.
        :param xs_b: Batch of xs.
        :param ys_b: Batch of ys.
        :return: the combine tensor (xs_b, ys_b)
        """
        bsize, points, dim = xs_b.shape  # bsize = batch_size, points = length of the prompt, dim = dimension of each x

        ys_b_wide = torch.cat(
            (
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
                ys_b.view(bsize, points, 1),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)

        return zs

    def forward(self, xs, ys):
        """
        Forward method for custom Transformer model
        :param xs: Batch of xs.
        :param ys: Batch of ys.
        :return: the predictions.
        """
        zs = self._combine(xs, ys)
        zs = self.proj(zs)
        # position embedding
        pos_embedding = self.pos_embed[:, :zs.shape[1], :]

        output = self.blocks(zs + pos_embedding)
        output = self.ln(output)
        predictions = self._read_out(output)  # linear projection on R.

        return predictions[:, ::2, 0]  # slice every two


#################################################################################################
class TransformerLinearRegressionLSAConfig:
    """
    Config class for a custom Transformer used for linear regression.
    """
    attn_dropout = 0.0
    embed_dropout = 0.0
    ff_dropout = 0.0

    def __init__(self, vocab_size, max_len, num_heads, num_blocks, embed_dim, **kwargs):
        """
        Init method for the Transformer model.
        :param vocab_size: size of the vocabulary.
        :param max_len: max length of the prompt.
        :param num_heads: number of heads in SA layer.
        :param num_blocks: number of decoder blocks.
        :param embed_dim: number of embedding dims.
        :param kwargs: further params.
        """
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        # additional arguments if we want to set them
        for key, value in kwargs.items():
            setattr(self, key, value)

class Block_LSA(nn.Module):
    """
    Implement decoder block
    """

    def __init__(self, config):
        """
        Initialize decoder block
        :param config: Configurations
        """
        super().__init__()
        embed_dim = config.embed_dim
        self.ln1 = nn.LayerNorm(embed_dim)
        # self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention_LSA(config)
        # self.ff = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim * 4),
        #     nn.GELU(),
        #     nn.Linear(embed_dim * 4, embed_dim),
        #     nn.Dropout(config.ff_dropout),
        # )

    def forward(self, x):
        """
        Forward for decoder block (self-attention+feed_forward)
        :param x: inputs
        :return: predictions
        """
        x = x + self.attn(self.ln1(x))
        # x = x + self.ff(self.ln2(x))
        # x = x + self.attn(self.ln1(x))
        return x


class MultiheadAttention_LSA(nn.Module):
    """
    The class implements the multihead self-attention layer.
    """

    def __init__(self, config):
        """
        Init for the class.
        :param config: the configurations.
        """
        super().__init__()
        embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        assert embed_dim % self.num_heads == 0, "invalid heads and embedding dimension configuration"

        # key components of SA layer
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.proj_dropout = nn.Dropout(config.ff_dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_len, config.max_len)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, zs):
        """
        Forward for multi-head self attention layer
        :param zs:
        :return:
        """
        batch_size = zs.size(0)
        seq_len = zs.size(1)
        # x.shape == (batch_size, seq_len, embed_dim)
        k_t = self.key(zs).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(zs).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(zs).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape == (batch_size, num_heads, seq_len, head_dim)

        # compute attention
        attn = torch.matmul(q, k_t) / math.sqrt(q.size(-1))
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        # apply masking
        mask = self.mask[:, :, :seq_len, :seq_len]
        attn = attn.masked_fill(mask == 0, float(0))
        attn = self.attn_dropout(attn)
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        # compute score
        # attn = F.softmax(attn, dim=-1)
        y = torch.matmul(attn, v)
        # y.shape == (batch_size, num_heads, seq_len, head_dim)
        y = y.transpose(1, 2)
        # y.shape == (batch_size, seq_len, num_heads, head_dim)
        y = y.reshape(batch_size, seq_len, -1)
        # y.shape == (batch_size, seq_len, embed_dim)
        y = self.proj_dropout(self.proj(y))
        return y


class TransformerLinearRegression_LSA(nn.Module):

    def __init__(self, config):
        """
        init of the Transformer
        :param config: configurations.
        """
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_dims = config.n_dims
        self.max_len = config.max_len
        self.initializer_range = config.initializer_range
        self.name = f"transformer_linear_regression_LSA_n_layers={config.num_blocks}"

        # self.tok_embed = nn.Embedding(config.vocab_size, embed_dim)
        # self.pos_embed = nn.Embedding(config.max_len, embed_dim)
        self.proj = nn.Linear(self.n_dims+1, self.embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_len, self.embed_dim)
        )
        self.blocks = nn.Sequential(
            *[Block_LSA(config) for _ in range(config.num_blocks)]
        )
        self.ln = nn.LayerNorm(self.embed_dim)
        self._read_out = nn.Linear(self.embed_dim, 1)

    def initialize_weights(self, m):
        """
        Initialize the weights of the Transformers
        :param m: model
        :return: None
        """
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if m.bias is not None:
                m.bias.data.zero_()

    @staticmethod  # method that does not use any method of the class but it makes sense to add it to the class
    def _combine(xs_b, ys_b):
        """
        Interleaves xs and ys into a single input matrix for each batch.
        :param xs_b: Batch of xs.
        :param ys_b: Batch of ys.
        :return: the combine tensor (xs_b, ys_b)
        """
        bsize, points, dim = xs_b.shape  # bsize = batch_size, points = length of the prompt, dim = dimension of each x

        xs_b_wide = torch.cat(
            (
                xs_b,
                torch.zeros(bsize, points, 1, device=ys_b.device)
            ),
            axis=2,
        )
        ys_b_wide = torch.cat(
            (
                xs_b,
                ys_b.view(bsize, points, 1),  # reshape ys_b as a tensor (bsize, points, 1)
            ),
            axis=2,
        )
        zs = torch.stack((xs_b_wide, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim + 1)

        return zs

    def forward(self, xs, ys):
        """
        Forward method for custom Transformer model
        :param xs: Batch of xs.
        :param ys: Batch of ys.
        :return: the predictions.
        """
        zs = self._combine(xs, ys)
        zs = self.proj(zs)
        # position embeddign
        pos_embedding = self.pos_embed[:, :zs.shape[1], :]

        output = self.blocks(zs + pos_embedding)
        output = self.ln(output)
        predictions = self._read_out(output)  # linear projection on R.

        return predictions[:, ::2, 0]  # slice every two
