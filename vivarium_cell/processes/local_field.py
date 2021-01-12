import numpy as np

from vivarium.core.process import Deriver
from vivarium.library.units import units, remove_units

from vivarium_cell.library.lattice_utils import (
    get_bin_site,
    get_bin_volume,
    count_to_concentration,
)

CONCENTRATION_UNIT = 1  # TODO (ERAN) set value -- units.ng / units.mL
LENGTH_UNIT = units.um


class LocalField(Deriver):

    name = 'local_field'
    defaults = {}

    def __init__(self, parameters=None):
        super(LocalField, self).__init__(parameters)

    def ports_schema(self):
         return {
            'exchanges': {
                '*': {
                    '_default': 0,  # counts!
                }
            },
            'location': {
                '_default': [0.5 * LENGTH_UNIT, 0.5 * LENGTH_UNIT]
            },
            'fields': {
                '*': {
                    '_default': np.ones(1),
                    '_updater': 'accumulate',
                }
            },
            'dimensions': {
                'bounds': {
                    '_default': [1, 1],
                },
                'n_bins': {
                    '_default': [1, 1],
                },
                'depth': {
                    '_default': 1,
                },
            }
        }


    def next_update(self, timestep, states):
        location = remove_units(states['location'])
        n_bins = states['dimensions']['n_bins']
        bounds = states['dimensions']['bounds']
        depth = states['dimensions']['depth']
        exchanges = states['exchanges']

        # get bin
        bin_site = get_bin_site(location, n_bins, bounds)
        bin_volume = get_bin_volume(n_bins, bounds, depth) * units.L

        # apply exchanges
        delta_fields = {}
        reset_exchanges = {}
        for mol_id, value in exchanges.items():
            delta_fields[mol_id] = np.zeros(
                (n_bins[0], n_bins[1]), dtype=np.float64)
            exchange = value * units.count
            concentration = count_to_concentration(exchange, bin_volume)
            delta_fields[mol_id][bin_site[0], bin_site[1]] += concentration.to(
                units.mmol / units.L).magnitude
            reset_exchanges[mol_id] = {
                '_value': 0,
                '_updater': 'set'}

        return {
            'exchanges': reset_exchanges,
            'fields': delta_fields}


def test_local_fields():
    parameters = {}
    local_fields_process = LocalField(parameters)

    bounds = [5, 5]
    n_bins = [3, 3]
    initial_state = {
        'exchanges': {
            'A': 20
        },
        'location': [0.5, 0.5],
        'fields': {
            'A': np.ones((n_bins[0], n_bins[1]), dtype=np.float64)
        },
        'dimensions': {
            'bounds': bounds,
            'n_bins': n_bins,
            'depth': 1,
        }
    }

    output = local_fields_process.next_update(0, initial_state)


if __name__ == '__main__':
    test_local_fields()
