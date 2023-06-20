import os
from random import randint
import uuid
import torch.nn as nn
from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml
from transformers import get_constant_schedule_with_warmup
from eval import get_run_metrics
from samplers import get_data_sampler
from schema import schema
from models import build_model
import wandb

torch.backends.cudnn.benchmark = True


def train_step(model, Ts, vs, optimizer, scheduler, loss_func):
    """
    One step of training.
    :param model: The model.
    :param Ts: batch of tensors.
    :param vs: parameters that generated the tensors.
    :param optimizer: the optimizer.
    :param scheduler: the scheduler.
    :param loss_func: the loss function.
    :return: loss
    """
    optimizer.zero_grad()

    output = model(Ts, vs)
    loss = loss_func(output, vs)  # average over batch size and over len_prompt

    loss.backward()

    optimizer.step()
    scheduler.step()
    return loss.detach().item(), output.detach()
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
    scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1)

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

    # sample datasampler
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims, **args.training.data_kwargs)  # allows to get a

    pbar = tqdm(range(starting_step, args.training.train_steps))
    if "mu" in args.training.task_kwargs:
        rand_mu = args.training.task_kwargs["mu"]

    # for each training step
    for i in pbar:  # equivalent to for i in range(starting_step, args.training.train_steps) but with the bar
        data_sampler_args = {}
        task_sampler_args = {}

        if "mu" in args.training.task_kwargs:
            task_sampler_args["mu"] = rand_mu
        else:
            task_sampler_args["mu"] = 0
        task_sampler_args["delta"] = 1

        Ts, vs = data_sampler.sample_tensors(bsize, lambda_=args.training.lambda_)

        loss_func = nn.MSELoss()  # different loss function for each task
        assert n_dims**2 == Ts.shape[2]
        assert model.max_len >= Ts.shape[1]

        loss, output = train_step(model, Ts.cuda(), vs.cuda(), optimizer, scheduler, loss_func)

        # monitor metrics of interest
        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            cosine = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
            wandb.log(
                {
                    "overall_loss": loss,
                    "lr": scheduler.get_last_lr()[0],
                    "n_dims": n_dims,
                    "mean_cosine": torch.mean(torch.tensor([cosine(vs[i, :].cuda(), output[i, :]/torch.norm(output[i, :]))
                                                            for i in range(vs.shape[0])]))
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
    assert args.model.family in ["gpt2", "lstm", "DNN", "DNNSimplified", "simple_transformer",
                                 "simple_transformer_simplified_setting"]
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
