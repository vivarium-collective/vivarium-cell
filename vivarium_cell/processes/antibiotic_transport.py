'''
=====================================
Simulate Antibiotic Import and Export
=====================================
'''


from __future__ import absolute_import, division, print_function

import copy
import os

from vivarium.library.units import units
from vivarium.library.dict_utils import deep_merge, deep_merge_check
from vivarium.core.composition import (
    simulate_process_in_experiment,
    flatten_timeseries,
    save_timeseries,
    load_timeseries,
    PROCESS_OUT_DIR,
    assert_timeseries_close,
)
from vivarium.plots.simulation_output import plot_simulation_output


from vivarium_cell.processes.convenience_kinetics import ConvenienceKinetics
from vivarium_cell.data import REFERENCE_DATA_DIR


class AntibioticTransport(ConvenienceKinetics):

    name = 'antibiotic_transport'
    defaults = {
        'pump_kcat': 2e-4,
        'pump_km': 0.6,
        'pump_key': 'AcrAB-TolC',
        'antibiotic_key': 'antibiotic',
        'initial_internal_antibiotic': 0,
        'initial_external_antibiotic': 1,
        'initial_pump': 1,
        'global_deriver_key': 'global_deriver',
    }

    def __init__(self, initial_parameters=None):
        if initial_parameters is None:
            initial_parameters = {}
        super_defaults = super(AntibioticTransport, self).defaults
        deep_merge_check(self.defaults, super_defaults)
        self.defaults.update(super_defaults)
        parameters = copy.deepcopy(self.defaults)
        deep_merge(parameters, initial_parameters)

        kinetics_parameters = {
            'reactions': {
                'export': {
                    'stoichiometry': {
                        ('internal', parameters['antibiotic_key']): -1,
                        ('external', parameters['antibiotic_key']): 1,
                    },
                    'is_reversible': False,
                    'catalyzed by': [
                        ('pump_port', parameters['pump_key'])],
                },
            },
            'kinetic_parameters': {
                'export': {
                    ('pump_port', parameters['pump_key']): {
                        'kcat_f': parameters['pump_kcat'],
                        ('internal', parameters['antibiotic_key']):
                            parameters['pump_km'],
                    },
                },
            },
            'initial_state': {
                'fluxes': {
                    'import': 0.0,
                    'export': 0.0,
                },
                'internal': {
                    parameters['antibiotic_key']: parameters[
                        'initial_internal_antibiotic'],
                },
                'external': {
                    parameters['antibiotic_key']: parameters[
                        'initial_external_antibiotic'],
                },
                'pump_port': {
                    parameters['pump_key']: parameters['initial_pump'],
                },
            },
            'port_ids': ['internal', 'external', 'pump_port'],
            'global_deriver_key': parameters['global_deriver_key'],
        }

        super(AntibioticTransport, self).__init__(kinetics_parameters)


def run_antibiotic_transport():
    process = AntibioticTransport()
    settings = {
        'total_time': 4000,
        'environment': {
            'volume': 1e-15 * units.L,
        },
    }
    return simulate_process_in_experiment(process, settings)


def test_antibiotic_transport():
    timeseries = run_antibiotic_transport()
    flattened = flatten_timeseries(timeseries)
    reference = load_timeseries(os.path.join(
        REFERENCE_DATA_DIR, AntibioticTransport.name + '.csv'))
    assert_timeseries_close(flattened, reference)


def main():
    out_dir = os.path.join(PROCESS_OUT_DIR, AntibioticTransport.name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    timeseries = run_antibiotic_transport()
    plot_settings = {}
    plot_simulation_output(timeseries, plot_settings, out_dir)
    save_timeseries(timeseries, out_dir)


if __name__ == '__main__':
    main()
