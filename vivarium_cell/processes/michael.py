'''
Simulation of growth for a single hair on a human head.

Hair has three growth phases. From wikipedia:

Anagen (growth) phase
    - grows ~ 1 cm/month on average, known range between 0.6 and 3.36cm / month.
    - lasts from three to five years
    - about 85%–90% of the hairs on one's head are in the anagen phase at any given time

Catagen (transitional) phase
    - lasts about two weeks
    - Signals only affecting 1 percent of all hair at any given time determine when anagen ends and the catagen begins
    - while hair is not growing during this phase, length of terminal fibers increases b/c the follicle pushes them upward

Telogen (resting/shedding) phase
    - follicle remains dormant for one to four months
    - 10%-15% in this phase at any given time
    - At some point, the hair base will break free from the root and the hair will be shed
    - Within two weeks, the new hair shaft will begin to emerge once the telogen phase is complete

We spend an exponentially distributed random amount of time in each phase,
with rate constants from the bullets above, keeping track of (apparent) hair
length over time.
Timesteps represent days, despite being presented as seconds in the output.


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
from numpy.random import exponential

# a global NAME is used for the output directory and for the process name
NAME = 'HairGrowth'


class Phase(IntEnum):
    ANAGEN   = 0
    CATAGEN  = 1
    TELOGEN1 = 2
    TELOGEN2 = 3

class HairGrowth(Process):
    '''
    Simulates the growth of one hair on a human head.
    '''

    # give the process a name, so that it can register in the process_repository
    name = NAME

    # declare default parameters as class variables
    defaults = {
        'r'      : 1 / 30.437,            # anagen growth rate (cm/day)
        'r_c'    : (5 / 6) * .0416 / 14,  # catagen growth rate (cm/day)
        'q_A_C'  : 1 / (4 * 365),         # rate constant (A to C)
        'q_C_T1' : 1 / (2 * 7),           # rate constant (C to T1)
        'q_T1_T2': 1 / (4 * 30.437),      # rate constant (T1 to T2)
        'q_T2_A' : 1 / (2 * 7),           # rate constant (T2 to A)
    }

    next_phase_change = None

    def __init__(self, parameters=None):
        # parameters passed into the constructor merge with the defaults
        # and can be access through the self.parameters class variable
        super(HairGrowth, self).__init__(parameters)

    def ports_schema(self):
        '''
        One port ``global``, with ``length`` and ``phase`` variables.
        '''

        return {
            'global': {
                'length': {
                    '_default': 0.0,
                    '_updater': 'accumulate',
                    '_emit': True,
                },
                'phase': {
                    '_default': Phase.ANAGEN,
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
        q = [self.parameters['q_A_C'],
             self.parameters['q_C_T1'],
             self.parameters['q_T1_T2'],
             self.parameters['q_T2_A']]

        # generate time of next phase change
        if self.next_phase_change is None:
            self.next_phase_change = exponential(1 / q[phase])

        self.next_phase_change -= timestep

        # length updates (get correct update depending on phase)
        length_update = [self.parameters["r"] * timestep,    # anagen
                         self.parameters["r_c"] * timestep,  # catagen
                         0,                                  # telogen1
                         -length                             # telogen2
                         ][phase]

        # phase updates
        phase_update = phase
        if self.next_phase_change <= 0:
            phase_update = (phase + 1) % 4
            self.next_phase_change = None

        # return an update that mirrors the ports structure
        return {
            'global': {
                'length': length_update,
                'phase': phase_update
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
            'phase': Phase.ANAGEN
        },
    }

    # run the simulation
    sim_settings = {
        'total_time': 10 * 365,
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
