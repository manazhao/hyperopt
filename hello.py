from hyperopt import fmin, tpe, hp
import hyperopt.pyll.stochastic


def objective_function2(args):
  # args is a list of two elements which correspond to num_layers and num_nodes.
  num_layers, num_nodes = args
  # train classifier
  # get test error
  # return test error.
  return (num_layers - 5)**2 + num_nodes**2


# Space is defined by value combinations of num_layers and num_nodes. When a
# sample is generated from this space, it will be assigned to
# objective_function2.
# space2 = [hp.randint('num_layers', 20) - 10, hp.randint('num_nodes', 20) - 10]
#
# best = fmin(
#     fn=objective_function2, space=space2, algo=tpe.suggest, max_evals=1000)
#
# print(best)

# nested choice:
# https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce


def objective_function3(args):
  num_layers = args['num_layers']
  num_nodes = args['num_nodes']
  return (num_layers - 5)**2 + num_nodes**2


space3 = {
    'num_layers': hp.choice('num_layers', [3, 4, 5]),
    'num_nodes': hp.choice('num_nodes', [10, 50, 100])
}

best = fmin(
    fn=objective_function3, space=space3, algo=tpe.suggest, max_evals=1000)

print(best)
