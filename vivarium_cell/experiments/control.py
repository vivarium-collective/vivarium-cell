"""
===================
Experiment Control
===================

Run experiments from the command line
"""

import os
import argparse

from vivarium.core.composition import EXPERIMENT_OUT_DIR


def make_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


def add_arguments(experiments_library):
    parser = argparse.ArgumentParser(description='trigger experiments')
    parser.add_argument(
        'experiment_id',
        type=str,
        choices=list(experiments_library.keys()),
        help='experiment name')
    return parser.parse_args()


def control(experiments_library, out_dir=None):
    """
    Execute experiments from the command line
    """

    args = add_arguments(experiments_library)

    if not out_dir:
        out_dir = EXPERIMENT_OUT_DIR
    make_dir(out_dir)

    if args.experiment_id:
        experiment_id = str(args.experiment_id)
        experiment = experiments_library[experiment_id]

        if callable(experiment):
            control_out_dir = os.path.join(out_dir, experiment_id)
            make_dir(control_out_dir)
            experiment(control_out_dir)

        elif isinstance(experiment, dict):
            name = experiment.get('name', experiment_id)
            exp_function = experiment.get('experiment', experiment_id)
            control_out_dir = os.path.join(out_dir, name)
            make_dir(control_out_dir)
            exp_function(control_out_dir)

    else:
        print('provide experiment number')