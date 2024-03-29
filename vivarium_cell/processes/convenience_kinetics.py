'''
====================
Convenience Kinetics
====================

Convenience kinetics :cite:`liebermeister_bringing_2006` provides a
generic way to represent rate laws that follow Michaelis-Menten-style
enzyme kinetics. The generalized law can model arbitrary reaction
stoichiometries, catalytic enzymes, activators, and inhibitors.

If you are looking to model a catalyzed process, this may be the
:term:`process class` you need.

Executing this file directly simulates an instance of
:py:class:`ConvenienceKinetics` with the configuration from
:py:func:`get_glc_lct_config`.

------------
Bibliography
------------

.. bibliography:: /references.bib
    :style: plain

'''

import os

import numpy as np

from vivarium.core.process import Process
from vivarium.core.composition import (
    process_in_experiment,
    simulate_experiment,
    flatten_timeseries,
    save_timeseries,
    load_timeseries,
    PROCESS_OUT_DIR,
    assert_timeseries_close,
)
from vivarium.library.dict_utils import tuplify_port_dicts
from vivarium.library.units import units, remove_units
from vivarium.plots.simulation_output import plot_simulation_output

# vivarium-vivarium_cell imports
from vivarium_cell.library.kinetic_rate_laws import KineticFluxModel
from vivarium_cell.data import REFERENCE_DATA_DIR


#: The name of the process, which is used to name the subdirectory under
#: ``out/tests`` that stores the output data from executing this file.
NAME = 'convenience_kinetics'


class ConvenienceKinetics(Process):
    '''Michaelis-Menten-style enzyme kinetics model

     Arguments:
         initial_parameters: Configures the :term:`process` with the
             following configuration options:

             * **reactions** (:py:class:`dict`): Specifies the
               stoichiometry, reversibility, and catalysts of each
               reaction to model. For a non-reversible reaction
               :math:`A + B \\rightleftarrows 2C` catalized by an
               enzyme :math:`E`, we have the following reaction
               specification:

               .. code-block:: python

                 {
                     # reaction1 is a reaction ID
                     'reaction1': {
                         'stoichiometry': {
                             # 1 mol A is consumd per mol reaction
                             ('internal', 'A'): -1,
                             ('internal', 'B'): -1,
                             # 2 mol C are produced per mol reaction
                             ('internal', 'C'): 2,
                         },
                         'is reversible': False,
                         'catalyzed by': [
                             ('internal', 'E'),
                         ],
                     }
                 }

               Note that for simplicity, we assumed all the molecules
               and enzymes were in the ``internal`` port, but this is
               not necessary.
             * **kinetic_parameters** (:py:class:`dict`): Specifies
               the kinetics of the reaction by providing
               :math:`k_{cat}` and :math:`K_M` parameters for each
               enzyme. For example, let's say that for the reaction
               described above, :math:`k{cat} = 1`, :math:`K_A = 2`,
               and :math:`K_B = 3`. Then the reaction kinetics would
               be specified by:

               .. code-block:: python

                 {
                     'reaction1': {
                         ('internal', 'E'): {
                             'kcat_f': 1,  # kcat for forward reaction
                             ('internal', 'A'): 2,
                             ('internal', 'B'): 3,
                         },
                     },
                 }

               If the reaction were reversible, we could have
               specified ``kcat_r`` as the :math:`k_{cat}` of the
               reverse reaction.
             * **initial_state** (:py:class:`dict`): Provides the
               initial quantities of the molecules and enzymes. The
               initial reaction flux must also be specified. For
               example, to start with :math:`[E] = 1.2 mM` and
               :math:`[A] = [B] = [C] = 0 mM`, we would have:

               .. code-block:: python

                 {
                     'internal': {
                         'A': 0.0,
                         'B': 0.0,
                         'C': 0.0,
                         'E': 1.2,
                     },
                 }

               .. note:: Unlike the previous configuration options,
                   the initial state dictionary is not divided up by
                   reaction.

             * **ports** (:py:class:`dict`): Each item in the
               dictionary has a :term:`port` name as its key and a
               list of the :term:`variables` in that port as its
               value. Each port should be specified only once. For
               example, the reaction we have been using as an example
               would have:

               .. code-block:: python

                 {
                     'internal': ['A', 'B', 'C', 'E'],
                 }

     The ports of the process are the ports configured by the
     user, with the following modifications:

     * A ``fluxes`` port is added with variable names equal to
       the IDs of the configured reactions.
     * A ``exchanges`` port is added with the same variables as the
       ``external`` port.
     * A ``global`` port is added with a variable named
       ``mmol_to_counts``, which is set by a :term:`deriver`, and
       ``location``, which is set by the environment.
     * A ``dimensions`` port is added with variables from the
       environment that specify the environment length, width, depth,
       and number of bins.

     Example configuring a process to model the kinetics and reaction
     described above.

     >>> configuration = {
     ...     'reactions': {
     ...         # reaction1 is the reaction ID
     ...         'reaction1': {
     ...             'stoichiometry': {
     ...                 # 1 mol A is consumd per mol reaction
     ...                 ('internal', 'A'): -1,
     ...                 ('internal', 'B'): -1,
     ...                 # 2 mol C are produced per mol reaction
     ...                 ('internal', 'C'): 2,
     ...             },
     ...             'is reversible': False,
     ...             'catalyzed by': [
     ...                 ('internal', 'E'),
     ...             ],
     ...         }
     ...     },
     ...     'kinetic_parameters': {
     ...         'reaction1': {
     ...             ('internal', 'E'): {
     ...                 'kcat_f': 1,  # kcat for forward reaction
     ...                 ('internal', 'A'): 2,
     ...                 ('internal', 'B'): 3,
     ...             },
     ...         },
     ...     },
     ...     'initial_state': {
     ...         'internal': {
     ...             'A': 0.0,
     ...             'B': 0.0,
     ...             'C': 0.0,
     ...             'E': 1.2,
     ...         },
     ...         'fluxes': {
     ...             'reaction1': 0.0,
     ...         }
     ...     },
     ...     'ports': {
     ...         'internal': ['A', 'B', 'C', 'E'],
     ...         'external': [],
     ...     },
     ... }
     >>> kinetic_process = ConvenienceKinetics(configuration)
     '''

    name = NAME
    defaults = {
        'reactions': {},
        'initial_state': {
            'internal': {},
            'external': {}},
        'kinetic_parameters': {},
        'port_ids': [
            'internal',
            'external'
        ],
        'added_port_ids': [
            'fluxes',
            'exchanges',
            'global'
        ],
        'global_deriver_key': 'global_deriver',
        'local_field_key': 'local_field'}

    def __init__(self, parameters=None):
        super(ConvenienceKinetics, self).__init__(parameters)

        self.reactions = self.parameters['reactions']
        kinetic_parameters = self.parameters['kinetic_parameters']
        self.port_ids = self.parameters['port_ids'] + self.parameters['added_port_ids']

        # make the kinetic model
        self.kinetic_rate_laws = KineticFluxModel(self.reactions, kinetic_parameters)

    def initial_state(self, config=None):
        return self.parameters['initial_state']

    def ports_schema(self):

        schema = {port_id: {} for port_id in self.port_ids}
        initial_state = self.initial_state()
        for port, states in initial_state.items():
            for state_id in states:
                schema[port][state_id] = {
                    '_default': initial_state[port][state_id],
                    '_emit': True}

        # exchanges
        # Note: exchanges depends on a port called external
        if 'external' in schema:
            schema['exchanges'] = {
                state_id: {
                    '_default': 0.0,
                }
                for state_id in schema['external'].keys()
            }

        # fluxes
        for state in self.kinetic_rate_laws.reaction_ids:
            schema['fluxes'][state] = {
                '_default': 0.0,
                '_updater': 'set',
            }

        # global
        schema['global'] = {
            'mmol_to_counts': {
                '_default': 0.0 * units.L / units.mmol,
            },
        }

        return schema

    def derivers(self):
        return {
            self.parameters['local_field_key']: {
                'deriver': 'local_field',
                'port_mapping': {
                    'exchanges': 'exchanges',
                },
                'config': {}
            },
            self.parameters['global_deriver_key']: {
                'deriver': 'globals_deriver',
                'port_mapping': {
                    'global': 'global'},
                'config': {
                    'width': 1.0}
            }
        }

    def next_update(self, timestep, states):

        # get mmol_to_counts for converting flux to exchange counts
        mmol_to_counts = states['global']['mmol_to_counts']

        # kinetic rate law requires a flat dict with ('port', 'state') keys.
        flattened_states = remove_units(tuplify_port_dicts(states))

        # get flux, which is in units of mmol / L
        fluxes = self.kinetic_rate_laws.get_fluxes(flattened_states)

        # make the update
        # add fluxes to update
        update = {port: {} for port in self.port_ids}
        update.update({'fluxes': fluxes})

        # update exchanges
        for reaction_id, flux in fluxes.items():
            stoichiometry = self.reactions[reaction_id]['stoichiometry']
            for port_state_id, coeff in stoichiometry.items():
                for port_id in self.port_ids:
                    # separate the state_id and port_id
                    if port_id in port_state_id:
                        state_id = port_state_id[1]
                        state_flux = coeff * flux * timestep

                        if port_id == 'external':
                            # convert exchange fluxes to counts with mmol_to_counts
                            delta = int((state_flux * mmol_to_counts).magnitude)
                            existing_delta = update['exchanges'].get(
                                state_id, {}).get('_value', 0)
                            update['exchanges'][state_id] = existing_delta + delta
                        else:
                            update[port_id][state_id] = (
                                update[port_id].get(state_id, 0)
                                + state_flux
                            )

        # note: external and internal ports update change in mmol.
        return update



# functions
def get_glc_lct_transport():

    transport_reactions = {
        'LCTSt3ipp': {
            'stoichiometry': {
                ('internal', 'h_c'): 1.0,
                ('external', 'h_p'): -1.0,
                ('internal', 'lcts_c'): 1.0,
                ('external', 'lcts_p'): -1.0
            },
            'is reversible': False,
            'catalyzed by': [('internal', 'LacY')]
        },
        'GLCptspp': {
            'stoichiometry': {
                ('internal', 'g6p_c'): 1.0,
                ('external', 'glc__D_e'): -1.0,
                ('internal', 'pep_c'): -1.0,
                ('internal', 'pyr_c'): 1.0,
            },
            'is reversible': False,
            'catalyzed by': [('internal', 'EIIglc')]
        },
        'GLCt2pp': {
            'stoichiometry': {
                ('internal', 'glc__D_c'): 1.0,
                ('external', 'glc__D_p'): -1.0,
                ('internal', 'h_c'): 1.0,
                ('external', 'h_p'): -1.0,
            },
            'is reversible': False,
            'catalyzed by': [('internal', 'GalP')]
        },
    }

    transport_kinetics = {
        # lcts uptake by LacY
        'LCTSt3ipp': {
            ('internal', 'LacY'): {
                ('external', 'h_p'): None,
                ('external', 'lcts_p'): 1e0,
                'kcat_f': 7.8e2,  # 1/s
            }
        },
        # g6p PTS uptake by EIIglc
        'GLCptspp': {
            ('internal', 'EIIglc'): {
                ('external', 'glc__D_e'): 1e0,
                ('internal', 'pep_c'): 1e0,
                'kcat_f': 7.5e4,  # 1/s
            }
        },
        # glc uptake by GalP
        'GLCt2pp': {
            ('internal', 'GalP'): {
                ('external', 'glc__D_p'): 1e0,
                ('external', 'h_p'): None,
                'kcat_f': 1.5e2,  # 1/s
            }
        },
    }

    transport_initial_state = {
        'internal': {
            'EIIglc': 1.8e-3,  # (mmol/L)
            'g6p_c': 0.0,
            'pep_c': 1.8e-1,
            'pyr_c': 0.0,
            'LacY': 0,
            'lcts_p': 0.0,
        },
        'external': {
            'glc__D_e': 10.0,
            'lcts_e': 10.0,
        },
    }

    transport_ports = {
        'internal': [
            'g6p_c', 'pep_c', 'pyr_c', 'EIIglc', 'LacY', 'lcts_p'],
        'external': [
            'glc__D_e', 'lcts_e']
    }

    return {
        'reactions': transport_reactions,
        'kinetic_parameters': transport_kinetics,
        'initial_state': transport_initial_state,
        'ports': transport_ports}


def get_glc_lct_config():
    """
    :py:class:`ConvenienceKinetics` configuration for simplified glucose
    and lactose transport.Glucose uptake simplifies the PTS/GalP system
    to a single uptake kinetic with ``glc__D_e_external`` as the only
    cofactor.

    You can use this configuration with :py:class:`ConvenienceKinetics`
    like this:

    >>> configuration = get_glc_lct_config()
    >>> kinetic_process = ConvenienceKinetics(configuration)
    """
    transport_reactions = {
        'EX_glc__D_e': {
            'stoichiometry': {
                ('internal', 'g6p_c'): 1.0,
                ('external', 'glc__D_e'): -1.0,
                ('internal', 'pep_c'): -1.0,
                ('internal', 'pyr_c'): 1.0,
            },
            'is reversible': False,
            'catalyzed by': [('internal', 'EIIglc')]
        },
        'EX_lcts_e': {
            'stoichiometry': {
                ('external', 'lcts_e'): -1.0,
                ('internal', 'lcts_p'): 1.0,
            },
            'is reversible': False,
            'catalyzed by': [('internal', 'LacY')]
        }
    }
    transport_kinetics = {
        'EX_glc__D_e': {
            ('internal', 'EIIglc'): {
                ('external', 'glc__D_e'): 1e0,  # k_m for external [glc__D_e]
                ('internal', 'pep_c'): None,  # k_m = None makes a reactant non-limiting
                'kcat_f': 1e2,
            }
        },
        'EX_lcts_e': {
            ('internal', 'LacY'): {
                ('external', 'lcts_e'): 1e0,
                'kcat_f': 1e2,
            }
        }
    }

    transport_initial_state = {
        'internal': {
            'EIIglc': 1.0e-3,  # (mmol/L)
            'g6p_c': 0.0,
            'pep_c': 1.0e-1,
            'pyr_c': 0.0,
            'LacY': 0,
            'lcts_p': 0.0,
        },
        'external': {
            'glc__D_e': 10.0,
            'lcts_e': 10.0,
        },
    }

    transport_ports = {
        'internal': [
            'g6p_c', 'pep_c', 'pyr_c', 'EIIglc', 'LacY', 'lcts_p'],
        'external': [
            'glc__D_e', 'lcts_e']
    }

    return {
        'reactions': transport_reactions,
        'kinetic_parameters': transport_kinetics,
        'initial_state': transport_initial_state,
        'ports': transport_ports}


def get_toy_config():
    '''
    Returns
        A configuration for :py:class:`ConvenienceKinetics` that models
        a toy reaction for illustration purposes.
    '''
    toy_reactions = {
        'reaction1': {
            'stoichiometry': {
                ('internal', 'A'): 1,
                ('external', 'B'): -1},
            'is reversible': False,
            'catalyzed by': [('internal', 'enzyme1')]
        }
    }

    toy_kinetics = {
        'reaction1': {
            ('internal', 'enzyme1'): {
                ('external', 'B'): 0.2,
                'kcat_f': 5e1,
            }
        }
    }

    toy_ports = {
        'internal': ['A', 'enzyme1'],
        'external': ['B']
    }

    toy_initial_state = {
        'internal': {
            'A': 1.0,
            'enzyme1': 1e-1,
        },
        'external': {
            'B': 10.0,
        },
    }

    return {
        'reactions': toy_reactions,
        'kinetic_parameters': toy_kinetics,
        'initial_state': toy_initial_state,
        'ports': toy_ports}


def test_convenience_kinetics(end_time=2520):
    config = get_glc_lct_config()
    kinetic_process = ConvenienceKinetics(config)

    initial_state = kinetic_process.initial_state()
    initial_state['external'] = {
            'glc__D_e': 1.0,
            'lcts_e': 1.0}
    settings = {
        'environment': {
            'volume': 1e-14 * units.L,
            'concentrations': initial_state['external'],
        },
        'timestep': 1,
        'total_time': end_time}

    experiment = process_in_experiment(
        process=kinetic_process,
        settings=settings,
        initial_state=initial_state)

    return simulate_experiment(experiment, settings)


def test_convenience_kinetics_correlated_to_reference():
    timeseries = test_convenience_kinetics()
    flattened = flatten_timeseries(timeseries)
    reference_timeseries = load_timeseries(
        os.path.join(REFERENCE_DATA_DIR, NAME + '.csv'))
    assert_timeseries_close(
        flattened, reference_timeseries,
        tolerances={
            'internal_pyr_c': 9,
            'internal_pep_c': 9,
            'internal_g6p_c': 9,
        }
    )


if __name__ == '__main__':
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    timeseries = test_convenience_kinetics()

    plot_settings = {}
    plot_simulation_output(timeseries, plot_settings, out_dir)
    save_timeseries(timeseries, out_dir)
