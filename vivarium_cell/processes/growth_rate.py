"""
===================
Growth rate process
===================
"""

import os

import numpy as np
from scipy import constants

from vivarium.library.units import units
from vivarium.core.process import Process
from vivarium.core.composition import (
    simulate_process_in_experiment,
    PROCESS_OUT_DIR,
)
from vivarium.plots.simulation_output import plot_simulation_output


NAME = 'growth_rate'
AVOGADRO = constants.N_A


class GrowthRate(Process):
    """The GrowthRate :term:`process class` models exponential cell growth.

    The cell's mass :math:`m_{t + h}` at time :math:`t + h` for
    :term:`timestep` :math:`h` and with growth rate :math:`r` is modeled
    as:

    .. math::

        m_{t + h} = m_t e^{rh}

    Configuration Options:

    * ``growth_rate``: The cell's growth rate :math:`r`. This rate is
      0.0005 by default to approximate an expected 22.5 minutes expected doubling
      time in LB medium: https://bionumbers.hms.harvard.edu/bionumber.aspx?s=n&v=0&id=110058
    """

    name = NAME
    defaults = {
        'growth_rate': 0.0005,
        'mw': {
            'biomass': AVOGADRO * units.fg / units.mol},
        'global_deriver_key': 'globals_deriver',
        'mass_deriver_key': 'mass_deriver',
    }

    def __init__(self, initial_parameters=None):
        super(GrowthRate, self).__init__(initial_parameters)

    def initial_state(self, config=None):
        if config is None:
            config = {}
        initial_mass = config.get('initial_mass', 1339)
        # TODO -- divide up molecules to get initial_mass
        return {
            'molecules': {
                molecule: initial_mass
            } for molecule in self.parameters['mw']
        }

    def ports_schema(self):
        return {
            'molecules': {
                molecule: {
                    '_emit': True,
                    '_default': 1.0,
                    '_divider': 'split',
                    '_properties': {
                        'mw': mw
                    }
                } for molecule, mw in self.parameters['mw'].items()
            },
            'global': {
                'growth_rate': {
                    '_default': self.parameters['growth_rate']
                }
            }
        }

    def derivers(self):
        return {
            self.parameters['mass_deriver_key']: {
                'deriver': 'mass_deriver',
                'port_mapping': {
                    'global': 'global'},
                'config': {}
            }
        }

    def next_update(self, timestep, states):
        molecules = states['molecules']
        growth_rate = states['global']['growth_rate']
        molecules_update = {}
        for molecule, value in molecules.items():
            molecules_update[molecule] = value * np.exp(growth_rate * timestep) - value

        return {
            'molecules': molecules_update
        }


def test_growth_rate():
    growth_rate = GrowthRate({})
    initial_state = growth_rate.initial_state({})
    settings = {
        'total_time': 1350,
        'initial_state': initial_state
    }
    output = simulate_process_in_experiment(growth_rate, settings)

    return output


def run_compartment(out_dir):
    data = test_growth_rate()
    mass = data['global']['mass']
    growth = mass[-1]/mass[0]
    print('growth: {}'.format(growth))

    plot_settings = {}
    plot_simulation_output(data, plot_settings, out_dir)



if __name__ == '__main__':
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    run_compartment(out_dir)
