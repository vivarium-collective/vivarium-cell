from __future__ import absolute_import, division, print_function

import copy

import numpy as np

from vivarium.library.units import units
from vivarium.library.dict_utils import deep_merge
from vivarium.core.process import Deriver

from vivarium_cell.processes.derive_globals import AVOGADRO


class NonSpatialEnvironment(Deriver):
    '''A non-spatial environment with volume'''

    name = 'nonspatial_environment'
    defaults = {
        'volume': 1e-12 * units.L,
    }

    def __init__(self, parameters=None):
        super(NonSpatialEnvironment, self).__init__(parameters)
        volume = parameters.get('volume', self.defaults['volume'])
        self.mmol_to_counts = (AVOGADRO.to('1/mmol') * volume).to('L/mmol')


    def ports_schema(self):
        bin_x = 1 * units.um
        bin_y = 1 * units.um
        depth = self.parameters['volume'] / bin_x / bin_y
        n_bin_x = 1
        n_bin_y = 1
        return {
            'external': {
                '*': {
                    '_value': 0,
                },
            },
            'fields': {
                '*': {
                    '_value': np.ones((1, 1)),
                },
            },
            'dimensions': {
                'depth': {
                    '_value': depth.to(units.um).magnitude,
                },
                'n_bins': {
                    '_value': [n_bin_x, n_bin_y],
                },
                'bounds': {
                    '_value': [
                        n_bin_x * bin_x.to(units.um).magnitude,
                        n_bin_y * bin_y.to(units.um).magnitude,
                    ],
                },
            },
            'global': {
                'location': {
                    '_value': [0.5, 0.5],
                },
                'volume': {
                    '_value': self.parameters['volume'],
                }
            },
        }

    def next_update(self, timestep, states):
        fields = states['fields']

        update = {
            'external': {
                mol_id: {
                    '_updater': 'set',
                    '_value': field[0][0],
                }
                for mol_id, field in fields.items()
            },
        }

        return update
