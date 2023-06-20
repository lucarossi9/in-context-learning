import os

import matplotlib.pyplot as plt
import seaborn as sns
from eval import get_run_metrics, baseline_names, get_model_from_run
import fnmatch
import pandas as pd

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")

relevant_model_names = {
    "linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
    ],
    "sparse_linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
        "Lasso (alpha=0.01)",
    ],
    "decision_tree": [
        "Transformer",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
        "Greedy Tree Learning",
        "XGBoost",
    ],
    "relu_2nn_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
    ],
    "laplace_linear_regression": [
        "Transformer",
        "Adaptive Lasso",
        "posterior_mean_delta=1_sigma=0.1",
        "Lasso (alpha=0.01)",
    ]
}


def basic_plot(metrics, models=None, trivial=1.0):
    """
    Plot the error of the estimator as a function of the number of in-context examples.
    :param metrics: The metrics relative to the model.
    :param models: The models to plot-
    :param trivial: error of the trivial estimator
    :return:
    """
    fig, ax = plt.subplots(1, 1)
    if models is not None:
        metrics = {k: metrics[k] for k in models}

    color = 0
    ax.axhline(trivial, ls="--", color="gray")
    for name, vs in metrics.items():
        if name=="posterior_mean_delta=1_sigma=0.1":
          name = "posterior_mean"  # reset the name
        ax.plot(vs["mean"], "-", label=name, color=palette[color % 10], lw=2)
        if name!="posterior_mean":
            low = vs["bootstrap_low"]
            high = vs["bootstrap_high"]
            ax.fill_between(range(len(low)), low, high, alpha=0.3)  # fill with bootstrap CI
        color += 1
    ax.set_xlabel("in-context examples")
    ax.set_ylabel("squared error")
    ax.set_xlim(-1, len(low) + 0.1)
    ax.set_ylim(-0.1, 1.25)

    legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.set_size_inches(8, 5)
    for line in legend.get_lines():
        line.set_linewidth(3)

    return fig, ax


def collect_results(run_dir, df, valid_row=None, rename_eval=None, rename_model=None):
    """
    Collects all the results before plotting them
    :param run_dir: the directory where they are.
    :param df: the df with all models.
    :param valid_row: function checking if a row is valid.
    :param rename_eval: Flag if want to rename evals.
    :param rename_model: Flag if want to rename models.
    :return: all metrics to be plotted
    """
    all_metrics = {}
    for _, r in df.iterrows():  # iterate over the rows of the df
        if valid_row is not None and not valid_row(r):
            continue

        run_path = os.path.join(run_dir, r.task, r.run_id)
        _, conf = get_model_from_run(run_path, only_conf=True)

        metrics = get_run_metrics(run_path, skip_model_load=True)
        for eval_name, results in sorted(metrics.items()):
            processed_results = {}
            for model_name, m in results.items():
                if "gpt2" in model_name in model_name:
                    model_name = r.model
                    if rename_model is not None:
                        model_name = rename_model(model_name, r)
                if "DNN" in model_name:
                    model_name = r.model
                else:
                    model_name = baseline_names(model_name)
                m_processed = {}
                n_dims = conf.model.n_dims

                xlim = 2 * n_dims + 1
                if r.task in ["relu_2nn_regression", "decision_tree"]:
                    xlim = 200

                normalization = n_dims
                if r.task == "sparse_linear_regression":
                    normalization = int(r.kwargs.split("=")[-1])
                if r.task == "decision_tree":
                    normalization = 1

                for k, v in m.items():
                    v = v[:xlim]
                    v = [vv / normalization for vv in v]
                    m_processed[k] = v
                processed_results[model_name] = m_processed
            if rename_eval is not None:
                eval_name = rename_eval(eval_name, r)
            if eval_name not in all_metrics:
                all_metrics[eval_name] = {}
            all_metrics[eval_name].update(processed_results)
    return all_metrics


def plot_errors_layer_by_layer(results_path):
    """
    The function plots the errors of the probe models layer by layer
    :param results_path: path where the results are stored
    :return: None
    """
    num_files = len(fnmatch.filter(os.listdir(results_path), '*.*'))
    means = []
    percentiles_5 = []
    percentiles_95 = []
    for i in range(num_files):
        df = pd.read_csv(os.path.join(results_path, "run_" + str(i + 1) + ".csv"))
        # read results and compute statistics
        df = df.iloc[-50:, 1]
        means.append(df.mean())
        percentiles_5.append(df.quantile(0.05))
        percentiles_95.append(df.quantile(0.95))

    # Calculate the error bars
    lower_error = [means[i] - percentiles_5[i] for i in range(len(means))]
    upper_error = [percentiles_95[i] - means[i] for i in range(len(means))]

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(means)), means, yerr=[lower_error, upper_error], capsize=5, alpha=0.7)
    plt.xticks(range(len(means)), range(1, len(means) + 1))
    plt.xlabel('Layer')
    plt.ylabel('Mean squared error')
    plt.show()
