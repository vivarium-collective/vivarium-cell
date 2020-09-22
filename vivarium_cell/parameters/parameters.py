from __future__ import absolute_import, division, print_function

import os
import copy
import itertools
import math

import numpy as np
import matplotlib.pyplot as plt

from vivarium.library.units import units
from vivarium.core.composition import simulate_compartment_in_experiment
from vivarium.core.process import Generator

# processes for testing
from vivarium_cell.processes.convenience_kinetics import ConvenienceKinetics, get_glc_lct_config


def get_nested(dict, keys):
    d = dict
    for key in keys[:-1]:
        if key in d:
            d = d[key]
    try:
        value = d[keys[-1]]
    except:
        value = None
        print('value not found for: {}'.format(keys))
    return value

def set_nested(dict, keys, value, create_missing=True):
    d = dict
    for key in keys[:-1]:
        if key in d:
            d = d[key]
        elif create_missing:
            d = d.setdefault(key, {})
        else:
            return dict
    if keys[-1] in d or create_missing:
        d[keys[-1]] = value
    return dict

def get_parameters_logspace(min, max, number):
    ''' get list of n parameters logarithmically spaced between min and max '''
    range = np.logspace(np.log10(min), np.log10(max), number, endpoint=True)
    return list(range)

def run_compartment_get_output(compartment, parameters, condition, metrics, settings):
    settings['initial_state'] = condition
    settings['return_raw_data'] = True
    settings['compartment'] = parameters

    # run the simulation and get the last state
    sim_out = simulate_compartment_in_experiment(compartment, settings)
    time_vec = list(sim_out.keys())
    last_state = sim_out[time_vec[-1]]

    # pull out metric values from last_state
    output = []
    for output_value in metrics:
        output.append(get_nested(last_state, output_value))
    return output


def check_path_exists(path, dict):
    if len(path) > 0:
        key = path[0]
        if key in dict:
            sub_path = path[1:]
            sub_dict = dict[key]
            check_path_exists(sub_path, sub_dict)
        else:
            raise Exception('parameter path does not exist: {}'.format(path))


def parameter_scan(config):
    '''
    Pass in a config (dict) with:
        - compartment (object) -- a compartment class, for configuration by the parameters
        - scan_parameters (dict) -- each parameter location (tuple) mapped to a list of values
        - metrics (list) -- a list of output values (tuple) with the (port, key)
        - conditions (list) -- a list of state values (dict) with {port: {variable: value}}
            for the default state the condition is and empty dict, [{}]
        - settings (dict) -- simulation settings for the experiments

    Returns a list of all parameter combinations, and a dictionary with output values for those parameters
    '''

    compartment = config['compartment']
    scan_params = config['scan_parameters']
    metrics = config['metrics']
    settings = config.get('settings', {})
    conditions = config.get('conditions')
    if not conditions:
        conditions = [{}]
    n_conditions = len(conditions)

    ## Set up the parameters
    # get number of parameter sets for scan
    n_values = [len(v) for v in scan_params.values()]
    n_combinations = np.prod(np.array(n_values))
    print('parameter scan size: {}'.format(n_combinations))

    # get default parameters from baseline compartment object
    default_params = compartment.get_parameters()

    # check that scan parameters are in default parameters
    for path in scan_params.keys():
        check_path_exists(path, default_params)

    # get parameter sets for scan
    param_keys = list(scan_params.keys())
    param_values = list(scan_params.values())
    param_combinations = list(itertools.product(*param_values))  # a list of all parameter combinations
    param_sets = [dict(zip(param_keys, combo)) for combo in param_combinations]  # list of dicts with {param: value}

    # run all parameter sets and save results
    results = []
    for params_index, param_set in enumerate(param_sets):
        # set up the parameters
        # parameters = copy.deepcopy(default_params)
        parameters = {}
        for param_key, param_value in param_set.items():
            parameters = set_nested(parameters, param_key, param_value)

        ## run the parameter set for each condition's state
        for condition_index, condition_state in enumerate(conditions):
            print('running parameter set {}/{}, condition {}/{}'.format(
                params_index + 1,
                n_combinations,
                condition_index+1,
                n_conditions))

            output = run_compartment_get_output(
                compartment,
                parameters,
                condition_state,
                metrics,
                settings)

            result = {
                'parameter_index': params_index,
                'condition_index': condition_index,
                'output': output}
            results.append(result)

    # organize data by metric
    output_data = {
        'results': results,
        'metrics': metrics,
        'parameter_sets': {idx: param_set for idx, param_set in enumerate(param_sets)},
        'conditions': {idx: condition for idx, condition in enumerate(conditions)}}

    return organize_param_scan_results(output_data)


def organize_param_scan_results(data):
    results = data['results']
    metrics = data['metrics']
    parameter_sets = data['parameter_sets']
    conditions = data['conditions']

    param_indices = list(range(0, len(parameter_sets)))
    condition_indices = list(range(0, len(conditions)))

    # organize the results by metric
    metric_data = {
        metric: {
            condition_index: [None for param in param_indices]
            for condition_index in condition_indices}
        for metric in metrics}

    for result in results:
        param_index = result['parameter_index']
        condition_index = result['condition_index']
        output = result['output']
        for metric_index, datum in enumerate(output):
            metric = metrics[metric_index]
            metric_data[metric][condition_index][param_index] = datum

    return {
        'metric_data': metric_data,
        'parameter_sets': parameter_sets,
        'conditions': conditions}


def plot_scan_results(results, out_dir='out', filename='parameter_scan'):
    metric_data = results['metric_data']
    parameter_sets = results['parameter_sets']
    conditions = results['conditions']
    parameter_indices = [idx for idx, param in enumerate(parameter_sets)]

    # make the figure
    n_cols = 1
    lines_per_row = 8
    base_rows = len(metric_data)
    param_rows = math.ceil(len(parameter_indices)/lines_per_row)
    condition_rows = math.ceil(len(conditions)/lines_per_row)
    n_rows = base_rows + param_rows + condition_rows
    fig = plt.figure(figsize=(n_cols * 6, n_rows * 2))
    grid = plt.GridSpec(n_rows, n_cols)
    font = {'size': 6}
    plt.rc('font', **font)

    row_idx = 0
    col_idx = 0
    for metric, data, in metric_data.items():
        ax = fig.add_subplot(grid[row_idx, col_idx])
        for condition, param_data in data.items():
            ax.scatter(parameter_indices, param_data, label=condition)

        ax.legend(title='condition', bbox_to_anchor=(1.2, 1.0))
        ax.set_yscale('log')
        ax.set_ylabel(metric)
        ax.set_xticks(parameter_indices)
        ax.set_xlabel('parameter set #')

        row_idx += 1

    # prepare text
    param_text_row = 1 / lines_per_row / param_rows
    cond_text_row = 1 / lines_per_row / condition_rows
    parameter_text = [
        '{}: {}'.format(param_idx, param_set)
        for param_idx, param_set in parameter_sets.items()]
    condition_text = [
        '{}: {}'.format(condition_idx, condition)
        for condition_idx, condition in conditions.items()]

    # parameter text
    ax = fig.add_subplot(grid[base_rows:base_rows+param_rows, :])
    ax.text(0, 1.0, 'parameters')
    for text_idx, param in enumerate(parameter_text):
        ax.text(0, 0.9-text_idx*param_text_row, param)
    ax.axis('off')

    # condition text
    ax = fig.add_subplot(grid[base_rows+param_rows:, :])
    ax.text(0, 1.0, 'conditions')
    for text_idx, condition in enumerate(condition_text):
        ax.text(0, 0.9-text_idx*cond_text_row, condition)
    ax.axis('off')

    # save the figure
    fig_path = os.path.join(out_dir, filename)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.savefig(fig_path, bbox_inches='tight')


# testing
class TestConvienceKinetics(Generator):
    defaults = {
        'boundary_path': ('boundary',),
        'config':  {
            'process': {
                'reactions': {
                    # reaction1 is the reaction ID
                    'reaction1': {
                        'stoichiometry': {
                            # 1 mol A is consumed per mol reaction
                            ('internal', 'A'): -1,
                            ('internal', 'B'): -1,
                            # 2 mol C are produced per mol reaction
                            ('internal', 'C'): 2,
                        },
                        'is reversible': False,
                        'catalyzed by': [
                            ('internal', 'E'),
                        ],
                    }
                },
                'kinetic_parameters': {
                    'reaction1': {
                        ('internal', 'E'): {
                            'kcat_f': 1e1,  # kcat for forward reaction
                            ('internal', 'A'): 1e-1,  # km for A
                            ('internal', 'B'): 1e-1,  # km for B
                        },
                    },
                },
                'initial_state': {
                    'internal': {
                        'A': 1e2,
                        'B': 1e2,
                        'C': 1e2,
                        'E': 1e2,
                    },
                    'fluxes': {
                        'reaction1': 0.0,
                    }
                },
                'ports': {
                    'internal': ['A', 'B', 'C', 'E'],
                    'external': [],
                },
            }
        }
    }

    def __init__(self, config=None):
        self.config = self.defaults['config']
        self.boundary_path = self.or_default(config, 'boundary_path')

    def generate_processes(self, config):
        process = ConvenienceKinetics(config['process'])
        return {'process': process}

    def generate_topology(self, config):
        external_path = self.boundary_path + ('external',)
        return {
            'process': {
                'internal': ('internal',),
                'external': external_path,
                'fields': ('fields',),
                'fluxes': ('fluxes',),
                'global': self.boundary_path,
                'dimensions': ('dimensions',),
            }
        }


def scan_test():
    # initialize the compartment
    compartment = TestConvienceKinetics({})

    # parameters to be scanned, and their values
    scan_params = {
        ('process',
         'kinetic_parameters',
         'reaction1',
         ('internal', 'E'),
         'kcat_f'):
            get_parameters_logspace(1e0, 1e5, 6),
    }

    # metrics are the outputs of a scan
    metrics = [('fluxes', 'reaction1')]

    # define conditions
    conditions = [
        {'internal': {'E': 1e0}},
        {'internal': {'E': 1e1}},
        {'internal': {'E': 1e2}},
    ]

    # set up scan options
    timeline = [(5, {})]
    sim_settings = {
        'timeline': {
            'timeline': timeline,
            'ports': {
                'external': ('boundary', 'external')}}}

    # run scan
    scan_config = {
        'compartment': compartment,
        'scan_parameters': scan_params,
        'conditions': conditions,
        'metrics': metrics,
        'settings': sim_settings}
    results = parameter_scan(scan_config)
    return results


if __name__ == '__main__':
    out_dir = os.path.join('out', 'parameters', 'scan')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    results = scan_test()
    plot_scan_results(results, out_dir, 'test_scan')
