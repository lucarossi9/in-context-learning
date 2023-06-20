import json
import os
import sys

from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml
from os import listdir
from os.path import isfile, join
import models
from samplers import get_data_sampler, sample_transformation
from tasks import get_task_sampler


def get_model_from_run(run_path, step=-1, only_conf=False):
    """
    The function returns the model and its configurations from the given run-path.
    :param run_path: the path where the information of the run are stored.
    :param step: the checkpoint corresponding to the training step that we want to load.
    :param only_conf: If True, we return only the configurations of the model.
    :return: a tuple (model, conf) containing the model and the configuration of the run.
    """
    config_path = os.path.join(run_path, "config.yaml")  # in the run directory we get the config path
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))  # it is a dict with the arguments in config.yaml file
    if only_conf:  # extract only the conf
        return None, conf

    # otherwise load the full model
    model = models.build_model(conf.model)

    if step == -1:  # we load the final model which is in the file state.pt
        state_path = os.path.join(run_path, "state.pt")
        try:
            state = torch.load(state_path)  # try to load on GPU
        except:
            state = torch.load(state_path, map_location=torch.device('cpu'))  # otherwise load the model on CPU
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")  # load the model from a given step
        try:
            state_dict = torch.load(model_path)
        except:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    return model, conf


# Functions for evaluation


def eval_batch(model, task_sampler, xs, xs_p=None, delta=1, mu=0, complete_save=True, seeds=None):
    """
    The function evaluates the model for a new batch of prompts.
    :param model: The model to be evaluated.
    :param task_sampler: The task sampler used to sample the task.
    :param xs: The inputs xs.
    :param xs_p: The inputs pre-transformation.
    :param delta: The parameter such that w \sim N(mu, delta I).
    :param mu: The parameter such that w \sim N(mu, delta I).
    :param complete_save: If complete_save==True we also save the labels and predictions.
    :param seeds: If specified we generate using fixed seeds for reproducibility.
    :return:
    """
    # generate task
    if seeds is None:
        task = task_sampler()
    else:
        task = task_sampler(seeds=seeds)

    # decide if evaluate on GPU or on CPU
    if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2", "lstm", "DNN", "simple", "transformer"]:
        device = "cuda"
    else:
        device = "cpu"

    # if we have mean different from zero we use the shifted OLS estimator
    if "OLS_driver" in model.name:
        model.add_mean_and_var(delta, mu)

    if xs_p is None:  # xs_p is post transformation
        ys = task.evaluate(xs)
        pred = model(xs.to(device), ys.to(device)).detach()
        metrics = task.get_metric()(pred.cpu(), ys)  # compute MSE
    else:
        b_size, n_points, _ = xs.shape
        metrics = torch.zeros(b_size, n_points)
        for i in range(n_points):
            xs_comb = torch.cat((xs[:, :i, :], xs_p[:, i:, :]), dim=1)  # meta-train on pre features test on new
            # features
            ys = task.evaluate(xs_comb)
            pred = model(xs_comb.to(device), ys.to(device), inds=[i]).detach()
            metrics[:, i] = task.get_metric()(pred.cpu(), ys)[:, i]  # compute MSE

    if complete_save:
        # save batch results
        batch_results = {"pred": pred.cpu().numpy().tolist(), "ys": ys.cpu().numpy().tolist()}
    else:
        batch_results = {}

    return metrics, batch_results


# Functions for generating different kinds of train/test data
def gen_standard(data_sampler, n_points, b_size, flag_corr, seeds):
    """
    Generate standard inputs xs using the sampler.
    :param data_sampler: Data sampler used to generate the inputs.
    :param n_points: Number of datapoints to be generated.
    :param b_size: The batch size.
    :param flag_corr: if flag_corr == True we add correlation between some of the features in X.
    :param seeds: if specified we test on specific seeds.
    :return: xs, None
    """
    # generate using the data sampler.
    xs = data_sampler.sample_xs(n_points, b_size, seeds=seeds, flag_corr=flag_corr)
    return xs, None


def gen_opposite_quadrants(data_sampler, n_points, b_size):
    """
    Generate points in opposite quadrants.
    :param data_sampler: the data sampler, instance of the class used to sample.
    :param n_points: the number of points to be generated.
    :param b_size: The batch size.
    :return: xs_pre_transformation, xs_post_transformation
    """
    # sample xs
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()  # different sign for each el in batch and each dim

    # pre will lie on the positive quadrant, post on the opposite one.
    xs_train_pre = xs.abs() * pattern
    xs_test_post = -xs_train_pre

    return xs_train_pre, xs_test_post


def gen_random_quadrants(data_sampler, n_points, b_size):
    """
    Generate points in random quadrants.
    :param data_sampler: The data sampler.
    :param n_points: The number of points.
    :param b_size: The batch size.
    :return: xs_pre_transformation, xs_post_transformation
    """
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = xs

    return xs_train_pre, xs_test_post


def gen_orthogonal_train_test(data_sampler, n_points, b_size):
    """
    Generate points in train and test such that these are orthogonal.
    :param data_sampler: The data sampler.
    :param n_points: The number of points.
    :param b_size: The batch size.
    :return: xs_pre_transformation, xs_post_transformation
    """
    xs = data_sampler.sample_xs(n_points, b_size)
    n_dim = xs.shape[2]
    n_points = min(n_points, n_dim)
    # raise ValueError("number of points should be at most the dimension.")
    xs_train_pre = xs
    xs_test_post = torch.zeros(xs.shape)
    for i in range(n_points):
        xs_test_post_i = xs[:, i: i + 1, :]
        xs_train_pre_i = xs[:, :i, :]

        # orthogonalize points
        _, _, Vt = torch.linalg.svd(xs_train_pre_i, full_matrices=False)
        xs_train_pre_i_projection = Vt.transpose(1, 2) @ Vt
        xs_test_post_i_orthogonalized = (
                xs_test_post_i - xs_test_post_i @ xs_train_pre_i_projection
        )  # remove the projection onto xs_train_pre_i_projection
        xs_test_post_i_normalized = (
                xs_test_post_i_orthogonalized
                * xs_test_post_i.norm(dim=2).unsqueeze(2)
                / xs_test_post_i_orthogonalized.norm(dim=2).unsqueeze(2)
        )

        xs_test_post[:, i: i + 1, :] = xs_test_post_i_normalized

    return xs_train_pre, xs_test_post


def gen_overlapping_train_test(data_sampler, n_points, b_size):
    """
    Generates points overlapped.
    :param data_sampler: The data sampler.
    :param n_points: The number of points.
    :param b_size: The batch size
    :return: xs_pre_transformation, xs_post_transformation
    """
    xs = data_sampler.sample_xs(n_points, b_size)
    xs_train_pre = xs
    xs_test_post = xs.clone()
    b_size = xs.shape[0]
    for i in range(1, n_points):
        xs_train_pre_i = xs[:, :i, :]
        perm = torch.stack([torch.randperm(i) for _ in range(b_size)]).unsqueeze(dim=1)  # randperm Returns a random
        # permutation of integers from 0 to n - 1.
        ind_mat = (perm == 0) + 0.0
        xs_test_post[:, i: i + 1, :] = ind_mat @ xs_train_pre_i

    return xs_train_pre, xs_test_post


def gen_orthogonal_design(data_sampler, n_points, b_size):
    """
    Generates orthogonal design matrix for evaluation.
    :param data_sampler: the sampler for the data.
    :param n_points: the number of points.
    :param b_size: the batch size.
    :return: xs_pre_transformation, xs_post_transformation
    """
    xs = data_sampler.sample_xs(n_points, b_size)
    dim = xs.size()[2]
    xs_train_pre = xs
    xs_test_post = xs.clone()
    for i in range(b_size):
        mat = xs_test_post[i, :, :]
        Q, R = torch.linalg.qr(mat)
        if n_points >= dim:  # normalize iff we have n_points > dim, otherwise we can't have orthogonal features
            scaling = torch.linalg.norm(xs_train_pre, axis=1).mean().item()
            xs_test_post[i, :, :] = Q * scaling
    return xs_train_pre, xs_test_post


def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Return aggregated metrics of the evaluation.
    :param metrics: The evaluation metrics to summarize.
    :param bootstrap_trials: The number of bootstrap trials for confidence intervals.
    :return: A list with std, means and bootstrap confidence intervals.
    """

    results = {}

    # compute mean and std
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, unbiased=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]

    # compute confidence intervals with bootstrap
    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}


def eval_model(
        model,
        save_path,
        task_name,
        data_name,
        n_dims,
        n_points,
        prompting_strategy,
        mu=0,
        delta=1,
        flag_corr=False,
        num_eval_examples=1280,
        batch_size=64,
        data_sampler_kwargs={},
        task_sampler_kwargs={},
):
    """
    Evaluate a model on a task with a variety of strategies.
    :param model: The model to be evaluated.
    :param save_path: The path in which we save the results.
    :param task_name: The name of the task.
    :param data_name: The name specifying how data has been generated.
    :param n_dims: The number of dimensions d.
    :param n_points: The number of points n.
    :param prompting_strategy: How the prompt data has to be generated  e.g., "random_quadrants".
    :param mu: The mean such that w_i \sim (mu, delta^2)
    :param delta: The std of the normal from which w is sampled.
    :param flag_corr: if True we add correlation between the columns of X.
    :param num_eval_examples: The number of prompts for the evaluation.
    :param batch_size: The batch size for the evaluation.
    :param data_sampler_kwargs: The kwargs relative to the sampling of the data.
    :param task_sampler_kwargs: The kwargs relative to the sampling of the weights.
    :return:
    """
    # add parameters mu and delta to the task sampler kwargs
    task_sampler_kwargs["delta"] = delta
    task_sampler_kwargs["mu"] = mu
    assert num_eval_examples % batch_size == 0

    # sample data and tasks
    data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)
    task_sampler = get_task_sampler(
        task_name, n_dims, batch_size, **task_sampler_kwargs
    )

    tmp_metrics = []
    tmp_results = []
    all_metrics = []
    all_results = []

    generating_func = globals()[f"gen_{prompting_strategy}"]  # The globals() method returns a dictionary with all the
    # global variables and symbols for the current program.

    # check if there are saved metrics
    save_at = 0
    load = False
    folder_save_path = save_path.replace("metrics.json", f"metric_folder_{model.name}")
    if os.path.exists(folder_save_path):
        # if the folder exists we will load the files
        files = [f for f in listdir(folder_save_path) if isfile(join(folder_save_path, f))]
        num_eval_examples = num_eval_examples - batch_size * len(files)
        save_at = len(files)
        load = True

    # fix the seeds
    seeds = list(range(0, batch_size))
    list_to_add = [save_at * batch_size for i in range(len(seeds))]
    seeds = [sum(x) for x in zip(seeds, list_to_add)]

    # evaluate the model and save the results
    for i in range(num_eval_examples // batch_size):
        xs, xs_p = generating_func(data_sampler, n_points, batch_size, flag_corr, seeds)
        metrics, batch_results = eval_batch(model, task_sampler, xs, xs_p, delta, mu, seeds=seeds)
        # save computed metrics
        folder_dir = save_path.replace("metrics.json", f"metric_folder_{model.name}")
        curr_save_path = save_path.replace("metrics.json", f"metric_folder_{model.name}/checkpoint_{i + save_at}.json")
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
        with open(curr_save_path, "w") as fp:
            metrics = metrics.numpy().tolist()
            dict = {"metrics": metrics, "results": batch_results}
            json.dump(dict, fp, indent=2)
            metrics = torch.tensor(np.array(metrics))
        tmp_metrics.append(metrics)
        tmp_results.append(batch_results)

    if load:
        # if there are already saved file we load them, we add the newly computed metrics and we save them.
        for file in files:
            with open(os.path.join(folder_save_path, file), "r") as fp:
                dict_res = json.load(fp)
                metrics = dict_res["metrics"]
                results = dict_res["results"]
                metrics = torch.tensor(np.array(metrics))
                all_metrics.append(metrics)
                all_results.append(results)

    all_metrics = all_metrics + tmp_metrics
    all_results = all_results + tmp_results
    metrics = torch.cat(all_metrics, dim=0)
    return aggregate_metrics(metrics)


def build_evals(conf):
    """
    The function builds the evaluation kwargs (the configuration) to pass to the evalutation task
    :param conf: The configuration of the run.
    :return: The evaluation kwargs of the model.
    """
    n_dims = conf.model.n_dims
    n_points = conf.training.curriculum.points.end
    batch_size = conf.training.batch_size
    task_name = conf.training.task

    if task_name == "noisy_laplace_linear_regression":
        task_name = "laplace_linear_regression"

    data_name = conf.training.data

    # set base kwargs
    base_kwargs = {
        "task_name": task_name,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
        "prompting_strategy": "standard",
        "flag_corr": False
    }

    # update kwargs
    if "mu" in conf.training.task_kwargs:
        base_kwargs["mu"] = conf.training.task_kwargs.mu
    if "delta" in conf.training.task_kwargs:
        base_kwargs["delta"] = conf.training.task_kwargs.delta
    if "flag_corr" in conf.training.data_kwargs:
        base_kwargs["flag_corr"] = conf.training.data_kwargs.flag_corr
    evaluation_kwargs = {}

    # build evaluation kwargs
    evaluation_kwargs["standard"] = {"prompting_strategy": "standard"}

    if task_name != "linear_regression":
        if task_name in ["relu_2nn_regression"]:
            evaluation_kwargs["linear_regression"] = {"task_name": "linear_regression"}
        for name, kwargs in evaluation_kwargs.items():
            # allow kwargs to override base_kwargs values
            evaluation_kwargs[name] = base_kwargs.copy()
            evaluation_kwargs[name].update(kwargs)
        return evaluation_kwargs

    # if task is linear regression, compute also all the scaling strategies
    for strategy in [
        "random_quadrants",
        "orthogonal_train_test",
        "overlapping_train_test",
    ]:
        evaluation_kwargs[strategy] = {"prompting_strategy": strategy}

    for method in ["half_subspace", "skewed"]:
        if "subspace" in method:
            eigenvals = torch.zeros(n_dims)
            eigenvals[: n_dims // 2] = 1
        else:
            eigenvals = 1 / (torch.arange(n_dims) + 1)

        scale = sample_transformation(eigenvals, normalize=True)
        evaluation_kwargs[f"{method}"] = {
            "data_sampler_kwargs": {"scale": scale},
        }

    for dim in ["x", "y"]:
        for scale in [0.333, 0.5, 2, 3]:
            if dim == "x":
                eigenvals = scale * torch.ones(n_dims)
                t = sample_transformation(eigenvals)
                scaling_args = {"data_sampler_kwargs": {"scale": t}}
            else:
                eigenvals = scale * torch.ones(n_dims)
                scaling_args = {"task_sampler_kwargs": {"scale": scale}}

            evaluation_kwargs[f"scale-{dim}={scale}"] = scaling_args

    evaluation_kwargs[f"noisyLR"] = {
        "task_sampler_kwargs": {"renormalize_ys": True, "noise_std": 1},
        "task_name": "noisy_linear_regression",
    }

    for name, kwargs in evaluation_kwargs.items():
        # allow kwargs to override base_kwargs values
        evaluation_kwargs[name] = base_kwargs.copy()
        evaluation_kwargs[name].update(kwargs)

    return evaluation_kwargs


def compute_evals(all_models, evaluation_kwargs, save_path=None, recompute=False):
    """
    The function compute the evaluations calling eval_model
    :param all_models:
    :param evaluation_kwargs:
    :param save_path:
    :param recompute:
    :return:
    """
    try:
        with open(save_path) as fp:
            all_metrics = json.load(fp)
    except Exception:
        all_metrics = {}
    # evaluate each model once at a time
    for eval_name, kwargs in tqdm(evaluation_kwargs.items()):
        metrics = {}
        if eval_name in all_metrics and not recompute:
            metrics = all_metrics[eval_name]
        for model in all_models:
            if model.name in metrics and not recompute:
                continue
            metrics[model.name] = eval_model(model, save_path, **kwargs)  # evaluate the model
        all_metrics[eval_name] = metrics
    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(all_metrics, fp, indent=2)
    return all_metrics


def get_run_metrics(
        run_path, step=-1, cache=True, skip_model_load=False, skip_baselines=False, **task_kwargs
):
    """
    The function returns the metrics for the task.
    :param run_path: The run path.
    :param step: The checkpoint that we want to evaluate.
    :param cache: If True try to cache previous metrics.
    :param skip_model_load: If True don't load the model.
    :param skip_baselines: If True skip the metrics for the baseline.
    :param task_kwargs: Kwargs for the task.
    :return: The metrics of the run.
    """

    # if we skip skip the load of the models, we only take conf
    if skip_model_load:
        _, conf = get_model_from_run(run_path, only_conf=True)
        all_models = []
    else:
        model, conf = get_model_from_run(run_path, step=-1)  # extract the model and the configurations
        try:
            model = model.cuda().eval()  # put the model in evaluation mode
        except:
            model = model.eval()
        all_models = [model]
        if not skip_baselines:
            # we add to all_models all the relevant for the comparison
            all_models += models.get_relevant_baselines(conf.training.task, **task_kwargs)
            # models that needs to be compared with it
    # builds and returns evaluation keywords, of type: strategy: {"prompting_strategy": type of strategy}
    evaluation_kwargs = build_evals(conf)

    if not cache:
        save_path = None
    elif step == -1:
        save_path = os.path.join(run_path, "metrics.json")
    else:
        save_path = os.path.join(run_path, f"metrics_{step}.json")

    # decide if the metrics have to be recomputed or not.
    recompute = False

    if save_path is not None and os.path.exists(save_path):
        checkpoint_created = os.path.getmtime(run_path)
        cache_created = os.path.getmtime(save_path)
        if checkpoint_created >= cache_created:
            recompute = True

    all_metrics = compute_evals(all_models, evaluation_kwargs, save_path, recompute)
    return all_metrics


def conf_to_model_name(conf):
    """
    From the configuration of the model returns the name of the model.
    :param conf: The configuration of the model.
    :return: the name of the model
    """
    if conf.model.family == "gpt2":
        return {
            (3, 2): "Transformer-xs",
            (6, 4): "Transformer-small",
            (12, 8): "Transformer",
        }[(conf.model.n_layer, conf.model.n_head)]
    if conf.model.family == "DNN":
        return "DNN"
    else:
        return conf.wandb.name


def baseline_names(name):
    """
    Transforms the baselines to their names.
    :param name: The name of the baseline.
    :return: The name for the plot.
    """
    if "OLS" in name:
        return "Least Squares"
    if name == "averaging":
        return "Averaging"
    if "NN" in name:
        k = name.split("_")[1].split("=")[1]
        return f"{k}-Nearest Neighbors"
    if "adaptive_lasso" in name:
        return f"Adaptive Lasso"
    if "lasso" in name:
        alpha = name.split("_")[1].split("=")[1]
        return f"Lasso (alpha={alpha})"
    if "gd" in name:
        return "2-layer NN, GD"
    if "decision_tree" in name:
        return "Greedy Tree Learning"
    if "xgboost" in name:
        return "XGBoost"
    return name


def read_run_dir(run_dir):
    """
    The function returns all the models in the run directory with some informations.
    :param run_dir: The run directory containing the folder of the different files.
    :return: The df containing a row for each run with additional information in its columns.
    """
    all_runs = {}
    for task in os.listdir(run_dir):
        if task == "mixtures":
            continue
        task_dir = os.path.join(run_dir, task)
        for run_id in os.listdir(task_dir):
            run_path = os.path.join(task_dir, run_id)
            _, conf = get_model_from_run(run_path, only_conf=True)
            params = {}
            params["run_id"] = run_id
            params["task"] = task
            params["model"] = conf_to_model_name(conf)
            params["kwargs"] = "_".join(
                f"{k}={v}" for k, v in conf.training.task_kwargs.items()
            )
            num_tasks = (
                conf.training.num_tasks if "num_tasks" in conf.training else None
            )
            params["num_tasks"] = num_tasks if num_tasks is not None else -1
            num_examples = (
                conf.training.num_training_examples
                if "num_training_examples" in conf.training
                else None
            )
            params["num_examples"] = num_examples if num_examples is not None else -1
            params["n_dims"] = conf.model.n_dims
            params["n_layer"] = conf.model.n_layer
            params["n_head"] = conf.model.n_head
            params["run_name"] = conf.wandb.name

            try:
                params["n_layers"] = conf.model.n_layers
            except:
                params["n_layers"] = -1
            try:
                params["hidden_size"] = conf.model.hidden_size
            except:
                params["hidden_size"] = -1

            for k, v in params.items():
                if k not in all_runs:
                    all_runs[k] = []
                all_runs[k].append(v)

    df = pd.DataFrame(all_runs).sort_values("run_name")
    # assert len(df) == len(df.run_name.unique())
    return df


if __name__ == "__main__":
    run_dir = sys.argv[1]
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        print(f"Evaluating task {task}")
        for run_id in tqdm(os.listdir(task_dir)):
            run_path = os.path.join(run_dir, task, run_id)
            metrics = get_run_metrics(run_path)
