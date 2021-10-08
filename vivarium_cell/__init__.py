import warnings

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
from vivarium_cell.processes.local_field import LocalField

# register processes
process_registry.register(DeriveGlobals.name, DeriveGlobals)
process_registry.register(LocalField.name, LocalField)

warnings.warn(
    'Vivarium Cell is deprecated. '
    'Please see https://vivarium-collective.github.io for alternatives.'
)
