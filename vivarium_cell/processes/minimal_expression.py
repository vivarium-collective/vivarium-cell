from __future__ import absolute_import, division, print_function

import os
import random

from vivarium.core.process import Process
from vivarium.library.dict_utils import tuplify_port_dicts
from vivarium.core.composition import (
    simulate_process_in_experiment,
    PROCESS_OUT_DIR,
)
from vivarium.plots.simulation_output import plot_simulation_output
from vivarium_cell.library.regulation_logic import build_rule


NAME = 'minimal_expression'


class MinimalExpression(Process):
    '''
    a minimal protein expression process.
    TO BE USED ONLY AS TRAINING WHEELS

    parameters:
        expression_rates (dict) with {'mol_id': probability_of_expression (1/sec)}
    '''

    name = NAME
    defaults = {
        'step_size': 1,
        'regulation': {},
        'concentrations_deriver_key': 'concentrations_deriver',
    }

    def __init__(self, initial_parameters=None):
        if initial_parameters is None:
            initial_parameters = {}

        expression_rates = initial_parameters.get('expression_rates')
        self.internal_states = list(expression_rates.keys()) if expression_rates else []
        regulation_logic = initial_parameters.get('regulation', self.defaults['regulation'])
        self.regulation = {
            gene_id: build_rule(logic) for gene_id, logic in regulation_logic.items()}
        regulators = initial_parameters.get('regulators', [])
        self.internal_regulators = [state_id for port_id, state_id in regulators if port_id == 'internal']
        self.external_regulators = [state_id for port_id, state_id in regulators if port_id == 'external']

        parameters = {
            'expression_rates': expression_rates,
            'step_size': initial_parameters.get('step_size', self.defaults['step_size'])}
        parameters.update(initial_parameters)

        self.concentrations_deriver_key = self.or_default(initial_parameters, 'concentrations_deriver_key')

        super(MinimalExpression, self).__init__(parameters)

    def ports_schema(self):
        return {
            'global': {},
            'concentrations': {},
            'external': {
                state: {
                    '_default': 0.0}
                for state in self.external_regulators},
            'internal': {
                state: {
                    '_default': 0.0,
                    '_emit': True}
                for state in self.internal_states + self.internal_regulators}}

    def derivers(self):
        return {
            self.concentrations_deriver_key: {
                'deriver': 'concentrations_deriver',
                'port_mapping': {
                    'global': 'global',
                    'counts': 'internal',
                    'concentrations': 'concentrations'},
                'config': {
                    'concentration_keys': self.internal_states + self.internal_regulators}}}

    def next_update(self, timestep, states):
        internal = states['internal']
        step_size = self.parameters['step_size']
        n_steps = int(timestep / step_size)

        # get state of regulated reactions (True/False)
        flattened_states = tuplify_port_dicts(states)
        regulation_state = {}
        for gene_id, reg_logic in self.regulation.items():
            regulation_state[gene_id] = reg_logic(flattened_states)

        internal_update = {state_id: 0 for state_id in internal.keys()}
        for state_id in internal.keys():
            if state_id in regulation_state and not regulation_state[state_id]:
                break
            rate = self.parameters['expression_rates'][state_id]
            for step in range(n_steps):
                if random.random() < rate:
                    internal_update[state_id] += 1

        return {
            'internal': internal_update}



# test functions
def get_toy_expression_config():
    toy_expression_rates = {
        'protein1': 1e-2,
        'protein2': 1e-1,
        'protein3': 1e0}

    return {
        'expression_rates': toy_expression_rates}

def test_expression(end_time=10):
    expression_config = get_toy_expression_config()
    # load process
    expression = MinimalExpression(expression_config)
    settings = {'total_time': end_time}
    return simulate_process_in_experiment(expression, settings)



if __name__ == '__main__':
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    timeseries = test_expression(1000)
    plot_simulation_output(timeseries, {}, out_dir)

