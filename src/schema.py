from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
)
from funcy import merge


model_schema = {
    "family": merge(tstring, allowed(["gpt2", "lstm", "DNN", "DNNSimplified", "transformer_mixtures",
                                      "transformer_tensor_pca", "transformer_linear_regression",
                                      "transformer_linear_regression_LSA"])),
    "n_positions": merge(tinteger, nullable, default(None)),  # maximum context length
    "n_dims": merge(tinteger, nullable, default(None)),  # latent dimension
    "n_embd": merge(tinteger, nullable, default(None)),  # embedding size
    "n_layer": merge(tinteger, nullable, default(None)),  # number of layers
    "n_head": merge(tinteger, nullable, default(None)),  # number of heads
    "n_layers": merge(tinteger, nullable, default(None)),  # number of heads
    "hidden_size": merge(tinteger, nullable, default(None)),  # number of heads
    "max_len_prompt": merge(tinteger, nullable, default(None)),  # number of heads
    "num_blocks": merge(tinteger, nullable, default(None)),  # number of blocks small transformer
    "initializer_range": merge(tfloat, nullable, default(None)),  # range of initialization weights
    "max_len": merge(tinteger, nullable, default(None))
}

curriculum_base_schema = {
    "start": merge(tfloat, nullable, default(None)),  # initial parameter
    "end": merge(tinteger, nullable, default(None)),  # limit of final value
    "inc": merge(tfloat, nullable, default(None)),  # how much to increment each time
    "interval": merge(tinteger, nullable, default(None)),  # increment every how many steps
}

curriculum_schema = {
    "dims": stdict(curriculum_base_schema),
    "points": stdict(curriculum_base_schema),
    "sigma": stdict(curriculum_base_schema),
}

TASK_LIST = [
    "linear_regression",
    "sparse_linear_regression",
    "linear_classification",
    "relu_2nn_regression",
    "decision_tree",
    "laplace_linear_regression",
    "noisy_laplace_linear_regression",
    "tensor_pca",
    "mixtures",
]

training_schema = {
    "task": merge(tstring, allowed(TASK_LIST)),
    "task_kwargs": merge(tdict, required),
    "num_tasks": merge(tinteger, nullable, default(None)),
    "num_training_examples": merge(tinteger, nullable, default(None)),
    "data": merge(tstring, allowed(["gaussian", "tensor", "mixtures"])),
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "train_steps": merge(tinteger, default(1000)),
    "save_every_steps": merge(tinteger, default(1000)),  # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),  # permanent checkpoints
    "resume_id": merge(tstring, nullable, default(None)),  # run uuid64
    "curriculum": stdict(curriculum_schema),
    "data_kwargs": merge(tdict, required),
    "check_full": merge(tboolean, nullable, default(False)),
    "simplified": merge(tboolean, default(False)),
    "lambda_": merge(tfloat, nullable, default(None)),
    "n_clusters": merge(tinteger, nullable, default(None)),
    "n_points": merge(tinteger, nullable, default(None)),
}

wandb_schema = {
    "project": merge(tstring, default("in-context-training")),
    "entity": merge(tstring, default("in-context")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)),
}
