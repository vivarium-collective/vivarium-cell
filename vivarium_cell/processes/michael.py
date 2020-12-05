'''
Execute by running: ``python vivarium_cell/processes/michael.py``

'''

import os

from vivarium.core.process import Process
from vivarium.core.composition import (
    simulate_process_in_experiment,
    PROCESS_OUT_DIR,
)
from vivarium.plots.simulation_output import plot_simulation_output
from enum import IntEnum

# a global NAME is used for the output directory and for the process name
NAME = 'HairGrowth'


class HairGrowth(Process):
    '''
    This process simulates the growth of one hair on a human head
    '''

    # give the process a name, so that it can register in the process_repository
    name = NAME

    # declare default parameters as class variables
    defaults = {
        'r': 1 / 30.437,  # anagen growth rate (cm/day)
        'r_c': (5 / 6) * .0416 / 14,  # catagen growth rate (cm/day)
        'q_A_C': 1 / (4 * 365),  # rate constant (A to C)
        'q_C_T1': 1 / (2 * 7),  # rate constant (C to T1)
        'q_T1_T2': 1 / (4 * 30.437),  # rate constant (T1 to T2)
        'q_T2_A': 1 / (2 * 7),  # rate constant (T2 to A)
    }

    def __init__(self, parameters=None):
        # parameters passed into the constructor merge with the defaults
        # and can be access through the self.parameters class variable
        super(HairGrowth, self).__init__(parameters)

    def ports_schema(self):
        '''
        ports_schema returns a dictionary that declares how each state will behave.
        Each key can be assigned settings for the schema_keys declared in Store:

        * `_default`
        * `_updater`
        * `_divider`
        * `_value`
        * `_properties`
        * `_emit`
        * `_serializer`
        '''

        return {
            'global': {
                'length': {
                    '_default': 0,
                    '_updater': 'accumulate',
                    '_emit': True,
                },
                'phase': {
                    '_default': 'anagen',
                    '_updater': 'set',
                    '_emit': True,
                },
            },
        }

    def next_update(self, timestep, states):
        # get the states
        length = states['global']['length']
        phase = states['global']['phase']

        # calculate timestep-dependent updates
        length_update = self.parameters["r"] * timestep
        phase_update = phase

        # return an update that mirrors the ports structure
        return {
            'global': {
                'length' : length_update,
                'phase' : phase_update
            }
        }


# functions to configure and run the process
def run_template_process():
    '''Run a simulation of the process.

    Returns:
        The simulation output.
    '''

    # initialize the process by passing in parameters
    parameters = {}
    template_process = HairGrowth(parameters)

    # declare the initial state, mirroring the ports structure
    initial_state = {
        'global': {
            'length': 0.0,
            'phase': 'anagen'
        },
     }

    # run the simulation
    sim_settings = {
        'total_time': 10,
        'initial_state': initial_state}
    output = simulate_process_in_experiment(template_process, sim_settings)

    return output


def test_template_process():
    '''Test that the process runs correctly.

    This will be executed by pytest.
    '''
    output = run_template_process()
    # TODO: Add assert statements to ensure correct performance.


def main():
    '''Simulate the process and plot results.'''
    # make an output directory to save plots
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    output = run_template_process()

    # plot the simulation output
    plot_settings = {}
    plot_simulation_output(output, plot_settings, out_dir)


# run module with python template/process/template_process.py
if __name__ == '__main__':
    main()
