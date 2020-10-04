
from vivarium_cell.experiments.control import control

from vivarium_cell.composites.lattice import (
    Lattice,
    make_lattice_config,
)
from vivarium_cell.composites.inclusion_body_growth import InclusionBodyGrowth

from vivarium.core.composition import (
    compartment_hierarchy_experiment,
    EXPERIMENT_OUT_DIR,
)




def run_experiment(out_dir):
    agent_id = '1'

    # initial state
    initial_state = {
        'agents': {
            agent_id: {
                'internal': {'A': 0},
                'external': {'A': 1},
                'global': {
                    'divide': False
                }
            },
        }
    }

    # declare the hierarchy
    hierarchy = {
        'generators': [
            {
                'type': Lattice,
                'config': make_lattice_config(),
                'topology': {
                    'global': ('global',),
                    'agents': ('agents',)
                }
            }
        ],
        'agents': {
            'generators': [
                {
                    'name': agent_id,
                    'type': InclusionBodyGrowth,
                    'config': {'agent_id': agent_id}
                },
            ]
        }
    }

    # configure experiment
    experiment = compartment_hierarchy_experiment(
        hierarchy=hierarchy,
        initial_state=initial_state)

    # run simulation
    experiment.update(time_total)
    output = experiment.emitter.get_data()
    experiment.end()




experiments_library = {
    '1': run_experiment,
}

if __name__ == '__main__':
    control(
        experiments_library=experiments_library,
        out_dir=EXPERIMENT_OUT_DIR)
