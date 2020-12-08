'''
Execute by running: ``python vivarium/process/template_process.py``


'''

import os

from vivarium.core.process import Process
from vivarium.core.composition import (
    simulate_process_in_experiment,
    PROCESS_OUT_DIR,
)
from vivarium.plots.simulation_output import plot_simulation_output


# a global NAME is used for the output directory and for the process name
NAME = 'Kpump'


class Kpump(Process):


    # give the process a name, so that it can register in the process_repository
    name = 'Kpump'

    # declare default parameters as class variables
    defaults = {
        'Opening_rate': .50,
        'Closing_rate': .45,
        'total_gates': 10100
    }

    def __init__(self, parameters=None):
        # parameters passed into the constructor merge with the defaults
        # and can be access through the self.parameters class variable
        super(Kpump, self).__init__(parameters)

    def ports_schema(self):


        return {
            'global': {
                'open': {
                    '_default': 5100,
                    '_updater': 'accumulate',
                    '_emit': True,
                }
                ,
                'closed': {
                    '_default': 5000,
                    '_updater': 'accumulate',
                    '_emit': True,
                },
            }
        }


    def next_update(self, timestep, states):

        # get the states
        open = states['global']['open']
        closed = states['global']['closed']
        proportion_open = open/self.parameters['total_gates']

        # calculate timestep-dependent updates
        open_update = open - (open * (self.parameters['Closing_rate'] * proportion_open)) + (closed * (self.parameters['Opening_rate'] * (1 - proportion_open)))
        closed_update = self.parameters['total_gates'] - open_update

        # return an update that mirrors the ports structure
        return {
            'global': {
                'open': open_update,
                'closed': closed_update
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
    template_process = Kpump(parameters)

    # declare the initial state, mirroring the ports structure
    initial_state = {
        'global': {
            'open': 5100,
            'closed': 5000
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


# run module with python vivarium/process/template_process.py
if __name__ == '__main__':
    main()
