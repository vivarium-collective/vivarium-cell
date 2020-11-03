'''
Execute by running: ``python vivarium/process/template_process.py``

TODO: Replace the template code to implement your own process.
'''

import os

from vivarium.core.process import Deriver
from vivarium.core.composition import (
    simulate_process_in_experiment,
    PROCESS_OUT_DIR,
)
from vivarium.plots.simulation_output import plot_simulation_output


NAME = 'spatial_geometry'


class SpatialGeometry(Deriver):
    """
    Assumes cylinder
    """

    name = NAME
    defaults = {
        'nodes': [],
        'edges': {},
        'density': 0.0,
    }

    def __init__(self, parameters=None):
        super(SpatialGeometry, self).__init__(parameters)

    def initial_state(self, config=None):
        pass

    def ports_schema(self):
        node_schema = {
            node_id: {
                'volume': {
                    '_default': 1.0,
                },
                'length': {
                    '_default': 1.0,
                },
                'radius': {
                    '_default': 1.0,
                },
                'molecules': {
                    '*': {
                        '_default': 0,
                    }
                }
            } for node_id in self.parameters['nodes'],
        }
        edge_schema = {
            edge_id: {
                'cross_sectional_area': 1.0,
            } for edge_id in self.parameters['edges'].keys()
        }

        return {**node_schema, **edge_schema}

    def next_update(self, timestep, states):

        # TODO -- get volume of each node from molecules and density

        #

        return {}





# functions to configure and run the process
def run_spatial_geometry_process():
    '''Run a simulation of the process.

    Returns:
        The simulation output.
    '''

    # initialize the process by passing in parameters
    parameters = {}
    spatial_geometry_process = SpatialGeometry(parameters)

    # declare the initial state, mirroring the ports structure
    initial_state = {}

    # run the simulation
    sim_settings = {
        'total_time': 10,
        'initial_state': initial_state}
    output = simulate_process_in_experiment(spatial_geometry_process, sim_settings)

    return output


def test_spatial_geometry_process():
    '''Test that the process runs correctly.

    This will be executed by pytest.
    '''
    output = run_spatial_geometry_process()
    # TODO: Add assert statements to ensure correct performance.

    return output


def main():
    '''Simulate the process and plot results.'''
    # make an output directory to save plots
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    output = test_spatial_geometry_process()

    # plot the simulation output
    plot_settings = {}
    plot_simulation_output(output, plot_settings, out_dir)


# run module with python vivarium/process/template_process.py
if __name__ == '__main__':
    main()
