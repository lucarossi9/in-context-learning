import os
from random import randint
import uuid
import torch.nn as nn
from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml
from eval import get_run_metrics
from samplers import get_data_sampler
from schema import schema
from models import build_model
import wandb

torch.backends.cudnn.benchmark = True


class LossLikelihoodBernoulli(nn.Module):
    """
    Implement custom loss function: -bernoulli loglikelihood
    """
    def __init__(self):
        super(LossLikelihoodBernoulli, self).__init__()

    def forward(self, xs, means, probas):
        """
        Forward method
        :param xs: observed points.
        :param means: means of the clusters.
        :param probas: probabilities of the clusters.
        :return: -log_likelihood mixture of Bernoulli
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


def train_step(model, xs, optimizer, loss_func, n_clusters):
    """
    One step of training.
    :param model: The model.
    :param xs: inputs.
    :param optimizer: the optimizer.
    :param loss_func: the loss function.
    :param n_clusters: the number of clusters.
    :return: loss
    """
    optimizer.zero_grad()

    means, probas = model(xs, n_clusters)
    loss = loss_func(xs, means, probas)  # average over batch size and over len_prompt

    loss.backward()

    optimizer.step()

    return loss.detach().item()
    # detach-># Returns a new Tensor, detached from the current graph.


def sample_seeds(total_seeds, count):
    """
    The function sample seeds from [0, total_seeds]
    :param total_seeds: total number of seeds
    :param count: number to sample
    :return: set of seeds
    """
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    """
    Train Transformer for mixture of bernoulli problem.
    :param model: model.
    :param args: params.
    :return:
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)  # set optimizer

    simplified_setting = args.training.simplified
    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)  # initial states before training
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]

    n_dims = model.n_dims
    bsize = args.training.batch_size

    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims, **args.training.data_kwargs)  # allows to get a

    pbar = tqdm(range(starting_step, args.training.train_steps))
    if "mu" in args.training.task_kwargs:
        rand_mu = args.training.task_kwargs["mu"]

    for i in pbar:  # equivalent to for i in range(starting_step, args.training.train_steps) but with the bar
        data_sampler_args = {}
        task_sampler_args = {}

        if "mu" in args.training.task_kwargs:
            task_sampler_args["mu"] = rand_mu
        else:
            task_sampler_args["mu"] = 0
        task_sampler_args["delta"] = 1

        # sample data
        xs, bernoulli_params, probas_params = data_sampler.sample_from_mixtures_bernoulli(bsize,
                                                                                          args.training.n_clusters,
                                                                                          args.training.n_points)
        loss_func = LossLikelihoodBernoulli()  # different loss function for each task

        assert n_dims == xs.shape[2]
        assert model.max_len >= xs.shape[0]

        # do one step of training
        loss = train_step(model, xs.cuda(), optimizer, loss_func, args.training.n_clusters)

        # monitor metrics
        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "lr": args.training.learning_rate,
                    "n_dims": n_dims
                },
                step=i,
            )

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)  # save intermediate results in state_path

        if (
                args.training.keep_every_steps > 0
                and i % args.training.keep_every_steps == 0
                and not args.test_run
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args)

    if not args.test_run and model.name.split("_")[0] not in ["simple"]:
        task_kwargs = args.training.task_kwargs
        _ = get_run_metrics(args.out_dir, **task_kwargs)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm", "DNN", "DNNSimplified", "transformer_mixtures",
                                 "transformer_tensor_PCA", "transformer_linear_regression"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
