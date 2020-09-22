from __future__ import absolute_import, division, print_function

from vivarium.library.units import units

from scipy import constants

nAvogadro = constants.N_A

COUNTS_UNITS = units.mmol
VOLUME_UNITS = units.L
MASS_UNITS = units.g
TIME_UNITS = units.s
CONC_UNITS = COUNTS_UNITS / VOLUME_UNITS


def molar_to_counts(fluxes, volume):
    '''
    input:
        fluxes -- list (molar)
        volume -- list (L)

    return:
        counts -- list
    '''
    # volume = cell_mass / density
    return (nAvogadro * volume * fluxes).astype(int)

def millimolar_to_counts(fluxes, volume):
    '''
    input:
        fluxes -- list (millimolar)
        volume -- list (L)

    return:
        counts -- list
    '''
    fluxes_mol = fluxes * 1e-3 # convert to molar
    return int(nAvogadro * volume * fluxes_mol)


def counts_to_molar(counts, volume):
    '''
    input:
        counts -- list
        volume -- list (L)

    return:
        fluxes -- list (molar)
    '''
    # volume = cell_mass / density
    return counts / (nAvogadro * volume)


def counts_to_millimolar(counts, volume):
    '''
    input:
        counts -- list
        volume -- list (L)

    return:
        fluxes -- list (molar)
    '''
    # volume = cell_mass / density
    return counts / (nAvogadro * volume * 1e-3)


def molar_to_molDCWhr(fluxes, dry_mass, cell_mass, density, timestep):
    '''
    input:
        flux: mmol/L
        dry_mass: fg
        cell_mass: fg
        density:
        timestep: seconds
    return:
        fluxes in mol/gDCW/hr
    '''
    # Coefficient to convert between flux (mol/g DCW/hr) basis and concentration (M) basis
    coefficient = dry_mass / cell_mass * density * timestep
    return (fluxes / coefficient).m_as(units.mmol / units.g / units.h)