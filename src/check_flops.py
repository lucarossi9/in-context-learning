from quinine import QuinineArgumentParser
from schema import schema
from models import build_model
from fvcore.nn import FlopCountAnalysis
import torch

# the script counts the number of flops of a model if we pass in command line the file conf/linear_regression.yaml
if __name__ == "__main__":

    # parse the arguments
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()

    # build the model
    model = build_model(args.model)

    # construct the input-outputs
    xs = torch.randn(args.training.batch_size, args.model.n_dims*2+1,
    args.model.n_dims)
    ys = torch.randn(args.training.batch_size, args.model.n_dims*2+1)

    # count the number of flops
    flops = FlopCountAnalysis(model, (xs, ys))

    # print the count together with the model
    print(f"number of flops {args.model.family}={flops.total()}")
