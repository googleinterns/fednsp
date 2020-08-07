"""This file defines a simple framework for hyper-parameter tuning.

It takes the input as a json fine which has a dictionary of hyper-parameters
to be tuned along with the list of values for that hyper-parameter.
All combinations of the hyperparameter values are run and result and checkpoints
are stored in  a separate folder for each run. Additionally, a config.json
file will be created in each directory which stores the parameters with which
the training for  a given experiment was done.
"""

import json
import os
import itertools
import numpy as np

class HyperParamTuning():
  """Defines the hyper-parameter tuning class.

  Attributes:
      config_file: The name of the config file for the hyperparameters.
  """
  def __init__(self, config_file='hyperparam_tuning.json'):
    super().__init__()
    
    with open(config_file) as f:
      self.param_dict = json.load(f)

  def run(self):
    """Does the hyper-parameter tuning over all combinations
    of hyper-parameters in the config file.
    """
    hyperparameters, values = zip(*self.param_dict.items())
    configurations = [dict(zip(hyperparameters, v)) for v in itertools.product(*values)]

    num_runs = 1

    for config in configurations:

      print('{} run:'.format(num_runs))
      print('Training model with following configuration:')

      for hparam, value in config.items():
        print('Hyperparameter : {}, Value : {}'.format(hparam, value))

      run_log_directory = '/hyperparam_tuning/tuning_run_' + str(num_runs)
      print('Training logs saved in {}'.format(run_log_directory))

      execute_command = 'python train_model.py --chkdir=' + run_log_directory

      for hparam, value in config.items():
        execute_command += (" --" + hparam + " " + str(value))

      os.system(execute_command)

      num_runs += 1 

if __name__ == '__main__':

  hparams_class = HyperParamTuning()
  hparams_class.run()


