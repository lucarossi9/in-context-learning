import torch
import torch.nn as nn
from models import Block
from models import TransformerLinearRegressionConfig
from eval import get_model_from_run
from train import train
import wandb
import shutil
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_submodel(model, depth, n_embd, n):
    """
    From a single Transformer model, it creates multiple probe submodels
    :param model: original model.
    :param depth: depth of the sub model.
    :param n_embd: embedding dimension.
    :param n: depth original model
    :return: the submodel
    """
    config = TransformerLinearRegressionConfig(None, model.max_len, 1, n, model.embed_dim)

    class SubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = n_embd
            self.n_dims = model.n_dims
            self.max_len = model.max_len
            self.initializer_range = model.initializer_range
            self.name = f"transformer_linear_regression_n_layers={depth}"

            # self.tok_embed = nn.Embedding(config.vocab_size, embed_dim)
            # self.pos_embed = nn.Embedding(config.max_len, embed_dim)
            self.proj = nn.Linear(self.n_dims, self.embed_dim)
            self.pos_embed = nn.Parameter(
                torch.zeros(1, model.max_len, self.embed_dim)
            )
            self.blocks = nn.Sequential(
                *[Block(config) for _ in range(depth)]
            )
            self.ln = nn.LayerNorm(self.embed_dim)
            # self.fc = nn.Linear(embed_dim, SimpleTransformerConfig.vocab_size)
            self._read_out = nn.Linear(self.embed_dim, 1)

        @staticmethod  # method that does not use any method of the class but it makes sense to add it to the class
        def _combine(xs_b, ys_b):
            """Interleaves the x's and the y's into a single sequence."""
            bsize, points, dim = xs_b.shape  # bsize = batch_size, points = length of the prompt, dim = dimension of
            # each x

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
            zs = self._combine(xs, ys)
            zs = self.proj(zs)
            # position embedding
            pos_embedding = self.pos_embed[:, :zs.shape[1], :]

            output = self.blocks(zs + pos_embedding)
            output = self.ln(output)
            predictions = self._read_out(output)

            return predictions[:, ::2, 0]

    return SubModel()


def test_errors_layer_by_layer(run_path):
    """
    Given a run path it computes the errors of the probe models.
    :param run_path: path of the run of the original model
    :return: None
    """
    model, conf = get_model_from_run(run_path)
    n = conf["model"]["num_blocks"]
    n_embd = conf["model"]["n_embd"]
    path = conf["out_dir"]
    submodels = []
    # creates all the submodels
    for depth in range(1, n + 1):
        submodel = create_submodel(model, depth, n_embd, n)

        state_dict_model = model.state_dict()
        state_dict_submodel = submodel.state_dict()
        for name, param in state_dict_model.items():
            if name.startswith('blocks') and int(name.split(".")[1]) < depth:
                state_dict_submodel[name].copy_(param)
            if name == "pos_embed" or name == "proj.weight" or name == "proj.bias":
                state_dict_submodel[name].copy_(param)

        for name, param in submodel.named_parameters():
            if '_read_out' not in name:
                param.requires_grad = False
        submodels.append(submodel)

    # train on wandb each submodel
    conf["training"]["train_steps"] = 50000
    os.environ['WANDB_START_METHOD'] = 'thread'
    for i, submodel in enumerate(submodels):
        try:
            shutil.rmtree(path)
            print("directory is deleted")
        except:
            print("Error occured")
        os.mkdir(path)
        run = wandb.init(
            dir=conf.out_dir,
            project=conf.wandb.project,
            entity=conf.wandb.entity,
            config=conf.__dict__,
            notes=conf.wandb.notes,
            name=f"run_{i + 1}",
            resume=True,
        )
        print(f"RUN MODEL {i + 1}")
        train(submodel.cuda(), conf)
        run.finish()

def read_predictions(folder_name, num_batches):
    """
    Reads the predictions from the run path and for a given estimator.
    :param folder_name: folder where results are stored.
    :param num_batches: number of batches.
    :return: predictions.
    """
    with open(f"{folder_name}/checkpoint_0.json") as f:
        data = json.load(f)
    preds = np.array(data["results"]["pred"])
    for i in range(1, num_batches):
        with open(f"{folder_name}/checkpoint_{i}.json") as f:
            data = json.load(f)
        batch_pred = np.array(data["results"]["pred"])
        preds = np.vstack((preds, batch_pred))
    return preds


def mean_similarity(preds_1, preds_2):
    """
    Compute SPD between predictions.
    :param preds_1: first outputs.
    :param preds_2: second outputs.
    :return: similarity between predictions for each number of in-context examples.
    """
    list_similarity = []
    for i in range(0, preds_1.shape[1]):
      list_similarity.append(np.mean((preds_1[:, i]-preds_2[:, i])**2))
    return list_similarity

def plot_SPD_laplace(folder, n_files):
    """
    :param folder: folder of the run.
    :param n_files: number of batch used for evaluation.
    :return:
    """
    sns.set_theme("notebook", "darkgrid")
    palette = sns.color_palette("colorblind")
    preds_OLS = read_predictions(folder+"/metric_folder_OLS_driver=None", n_files)
    preds_adaptive = read_predictions(folder+"/metric_folder_adaptive_lasso_max_iter=100000", n_files)
    preds_transformer = read_predictions(folder+"/metric_folder_gpt2_embd=256_layer=12_head=8", n_files)
    preds_lasso_01 = read_predictions(folder+"/metric_folder_lasso_alpha=0.01_max_iter=100000", n_files)
    preds_posterior = read_predictions(folder+"/metric_folder_posterior_mean_delta=1_sigma=0.1", n_files)

    similarity_with_OLS = mean_similarity(preds_transformer, preds_OLS)
    similarity_with_lasso_01 = mean_similarity(preds_transformer, preds_lasso_01)
    similarity_with_posterior_mean = mean_similarity(preds_transformer, preds_posterior)
    similarity_with_lasso_adaptive = mean_similarity(preds_transformer, preds_adaptive)

    plt.figure(figsize=(8, 5))
    fig, ax = plt.subplots(1, 1)

    color = 0
    metrics = {"OLS": similarity_with_OLS, "Lasso (alpha=0.1)": similarity_with_lasso_01,
              "Posterior mean": similarity_with_posterior_mean, "Adaptive Lasso": similarity_with_lasso_adaptive}
    for name, metric in metrics.items():
        ax.plot(metric, "-", label=name, color=palette[color % 10], lw=2)
        color += 1
    ax.set_xlabel("in-context examples")
    ax.set_ylabel("Nomalized SPD")
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.set_size_inches(8, 5)
    for line in legend.get_lines():
        line.set_linewidth(3)

    # Displaying the plot
    plt.show()
