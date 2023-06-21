import os
from random import randint
import uuid
from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml
from transformers import get_constant_schedule_with_warmup
from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model
import wandb

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, scheduler, loss_func, **kwargs):
    """
    Operation in the training step
    :param model: model to train.
    :param xs: input.
    :param ys: labels.
    :param optimizer: optimizer to use.
    :param scheduler: scheduler pytorch.
    :param loss_func: loss function.
    :param kwargs: additional args.
    :return: loss, output, grad
    """
    optimizer.zero_grad()
    output = model(xs, ys)  # compute output

    # compute loss
    if kwargs["simplified"]:
        loss = loss_func(output, ys[:, -1])  # average over batch size and over len_prompt
    else:
        loss = loss_func(output, ys)

    # backward pass
    loss.backward()

    # check the 2-norm of the gradients
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    grads = total_norm

    # optimizer step and scheduler step
    optimizer.step()
    scheduler.step()
    return loss.detach().item(), output.detach(), grads
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
    Train function
    :param model: model to train
    :param args: args of the training
    :return: None
    """
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)  # set optimizer
    scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1)

    # scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=50000,
    # num_training_steps=500001) scheduler = Scheduler(optimizer=optimizer, dim_embed=6, warmup_steps=80000)
    # print(model._backbone.wpe.weight)
    # with torch.no_grad():
    #   model._backbone.wpe.weight = nn.Parameter(torch.zeros_like(model._backbone.wpe.weight))
    # print(model._backbone.wpe.weight)
    # for p in model._backbone.wpe.parameters():
    #   p.requires_grad = False

    curriculum = Curriculum(args.training.curriculum)  # set curriculum

    simplified_setting = args.training.simplified
    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")

    # if resume id we resume the training of an older model by loading it
    if os.path.exists(state_path):
        state = torch.load(state_path)  # initial states before training
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]

        # update curriculum starting_step times since we are training from starting_step the loaded model
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    if "flag_corr" in args.training.data_kwargs:
        flag_corr = args.training.data_kwargs["flag_corr"]
        args.training.data_kwargs.pop("flag_corr")
    else:
        flag_corr = False

    # get samplers
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims, **args.training.data_kwargs)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,  # if specified tasks are sampled from a fixed number of weights vectors
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples
    if "mu" in args.training.task_kwargs:
        rand_mu = args.training.task_kwargs["mu"]

    for i in pbar:  # equivalent to for i in range(starting_step, args.training.train_steps) but with the bar
        data_sampler_args = {}
        task_sampler_args = {}

        # set arguments for samplers
        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples,
                                 bsize)  # generate bsize seeds random in (0, num_training_examples-1)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]
        if "mu" in args.training.task_kwargs:
            task_sampler_args["mu"] = rand_mu
        else:
            task_sampler_args["mu"] = 0
        if "delta" in args.training.task_kwargs:
            task_sampler_args["delta"] = curriculum.sigma
        else:
            task_sampler_args["delta"] = 1
        data_sampler_args["flag_corr"] = flag_corr

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)

        loss_func = task.get_training_metric()  # different loss function for each task
        dict_kwargs = {"simplified": simplified_setting}
        # one training step
        loss, output, grads = train_step(model, xs.cuda(), ys.cuda(),
                                                    optimizer, scheduler, loss_func, **dict_kwargs)
        # record metrics
        baseline_loss = (
                sum(
                    max(curriculum.n_dims_truncated - ii, 0)
                    for ii in range(curriculum.n_points)
                )
                / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "lr": scheduler.get_last_lr()[0],
                    "excess_loss": loss / baseline_loss,
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                    "sigma": curriculum.sigma,
                    "grads": grads,
                },
                step=i,
            )
        # update count of curriculum
        curriculum.update()

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
        os.environ['WANDB_START_METHOD'] = 'thread'
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
    # parse arguments
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm", "DNN", "DNNSimplified", "transformer_mixtures",
                                 "transformer_tensor_PCA", "transformer_linear_regression",
                                 "transformer_linear_regression_LSA"]
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
