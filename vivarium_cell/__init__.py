import numpy as np

from vivarium.library.units import Quantity, units
from vivarium.core.registry import (
    updater_registry,
    process_registry,
)
from vivarium_cell.library.lattice_utils import (
    get_bin_site,
    get_bin_volume,
    count_to_concentration,
)

# import processes
from vivarium_cell.processes.derive_globals import DeriveGlobals

# register processes
process_registry.register(DeriveGlobals.name, DeriveGlobals)

# register updaters
def update_field_with_exchange(current_value, new_value, states):
    '''Update environment with agent exchange

    Arguments:
        states: Dictionary with the following keys that specify the
            pre-update simulation state:

            * **global** (:py:class:`dict`): Contains the agent location
              under the ``location`` key.
            * **dimensions** (:py:class:`dict`): The contents of the
              environment's dimensions :term:`store` with the
              ``bounds``, ``n_bins``, and ``depth`` keys.

        new_value: Count of molecules to exchange with the environment
        current_value (numpy ndarray): The pre-update value of
            the field to update.

    Returns:
        The updated field.
    '''
    location = states['global']['location']
    n_bins = states['dimensions']['n_bins']
    bounds = states['dimensions']['bounds']
    depth = states['dimensions']['depth']
    exchange = new_value * units.count
    delta_field = np.zeros(
        (n_bins[0], n_bins[1]), dtype=np.float64)
    bin_site = get_bin_site(
        location, n_bins, bounds)
    bin_volume = get_bin_volume(
        n_bins, bounds, depth) * units.L
    concentration = count_to_concentration(exchange, bin_volume)
    delta_field[bin_site[0], bin_site[1]] += concentration.to(
        units.mmol / units.L).magnitude
    return current_value + delta_field

updater_registry.register(
    'update_field_with_exchange', update_field_with_exchange)
