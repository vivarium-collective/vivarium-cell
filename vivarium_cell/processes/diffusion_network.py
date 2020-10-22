import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants

from vivarium.core.process import Process
from vivarium.core.composition import (
    simulate_process_in_experiment,
    PROCESS_OUT_DIR,
)
from vivarium.plots.simulation_output import plot_simulation_output

NAME = 'diffusion_network'


class DiffusionNetwork(Process):

    name = NAME

    defaults = {
        'nodes': [],
        'edges': {},
        'mw': {},   # in fg
        'mesh_size': 50,    # in nm
        'time_step': 0.1,     # in s
    }

    def __init__(self, parameters=None):
        super(DiffusionNetwork, self).__init__(parameters)
        self.nodes = np.asarray(self.parameters['nodes'])
        self.edges = self.parameters['edges']
        self.mw = self.parameters['mw']
        self.molecule_ids = self.parameters['mw'].keys()
        self.mesh_size = self.parameters['mesh_size']
        self.time_step = self.parameters['time_step']

        # get molecule radii by molecular weights
        self.rp = calculate_rp_from_mw(self.molecule_ids, array_from(self.mw))

        # get diffusion constants per molecule
        self.diffusion_constants = compute_diffusion_constants_from_mw(
            self.molecule_ids, self.rp, self.mesh_size, self.edges)

        # construct A matrix based off of graph, all edges assumed bidirectional
        self.A = np.asarray([np.identity(len(self.nodes)) for mol in self.molecule_ids])
        for edge_id, edge in self.edges.items():
            node_index_1 = np.where(self.nodes == edge['nodes'][0])[0][0]
            node_index_2 = np.where(self.nodes == edge['nodes'][1])[0][0]
            dx = edge['dx']
            diffusion_constants = array_from(self.diffusion_constants[edge_id])
            if 'diffusion_scaling_constant' in edge:
                diffusion_constants *= edge['diffusion_scaling_constant']
            if 'diffusion_constants' in edge:
                diffusion_constants = edge['diffusion_constants']
            alpha = np.divide(np.multiply(diffusion_constants,
                                          self.time_step), dx ** 2)
            self.A[:, node_index_1, node_index_1] += alpha
            self.A[:, node_index_2, node_index_2] += alpha
            self.A[:, node_index_1, node_index_2] -= alpha
            self.A[:, node_index_2, node_index_1] -= alpha

    def ports_schema(self):
        '''
        ports_schema returns a dictionary that declares how each state will behave.
        Each key can be assigned settings for the schema_keys declared in Store:

        * `_default`
        * `_updater`
        * `_divider`
        * `_value`
        * `_properties`
        * `_emit`
        * `_serializer`
        '''
        schema = {
            node_id: {
                'volume': {
                    '_default': 1.0,
                    '_updater': 'set',
                },
                'length': {
                    '_default': 1.0,
                    '_updater': 'set',
                },
                'molecules': {
                    '*': {
                        '_default': 0,
                        '_updater': 'set',
                        '_emit': True,
                    }
                },
            } for node_id in self.parameters['nodes']
        }
        return schema

    def next_update(self, timestep, state):

        conc = np.asarray([np.multiply(array_from(state[node]['molecules']),
                                       array_from(self.mw)) / state[node]['volume']
                           for node in state])
        conc_final = np.asarray([np.matmul(np.linalg.inv(a), conc[:, i])
                                 for i, a in enumerate(self.A)]).T
        volumes = np.asarray(
            [state[node]['volume'] for node in state])
        count_final = np.rint(np.asarray([np.divide(node * volumes[i],
                                                    array_from(self.mw))
                                          for i, node in enumerate(conc_final)]))
        update = {
            node_id: {
                'molecules': array_to(self.molecule_ids,
                                      count_final[np.where(self.nodes == node_id)[0][0]].astype(int)),
            } for node_id in self.nodes
        }
        return update


# functions to configure and run the process
def run_diffusion_network_process(out_dir='out'):
    # initialize the process by passing initial_parameters
    n = int(1E6)
    molecule_ids = ['7', '6', '5', '4', '3', '2', '1', '0']
    initial_parameters = {
        'nodes': ['cytosol_front', 'nucleoid', 'cytosol_rear'],
        'edges': {
            '1': {
                'nodes': ['cytosol_front', 'nucleoid'],
                'dx': 0.75,
            },
            '2': {
                'nodes': ['nucleoid', 'cytosol_rear'],
                'dx': 0.75,
            },
            # '3': {
            #     'nodes': ['cytosol_front', 'cytosol_rear'],
            #     'dx': 0.75,
            # }
            },
        'mw': {
            '7': 20000E-5,
            '6': 9000E-5,
            '5': 5000E-5,
            '4': 2000E-5,
            '3': 1000E-5,
            '2': 100E-5,
            '1': 10E-5,
            '0': 4E-5,
        },
        'molecule_ids': molecule_ids,
        'mesh_size': 50,
    }

    diffusion_network_process = DiffusionNetwork(initial_parameters)

    # run the simulation
    sim_settings = {
        'total_time': 10,
        'initial_state': {
            'cytosol_front': {
                'volume': 0.3,
                'molecules': {
                    mol_id: n
                    for mol_id in molecule_ids}
            },
            'nucleoid': {
                'volume': 0.3,
                'molecules': {
                    mol_id: 0
                    for mol_id in molecule_ids}
            },
            'cytosol_rear': {
                'volume': 0.3,
                'molecules': {
                    mol_id: 0
                    for mol_id in molecule_ids}
            },
        }
    }
    output = simulate_process_in_experiment(diffusion_network_process, sim_settings)

    # helpful parameters to recalculate for plots
    edges = initial_parameters['edges']
    mw = initial_parameters['mw']
    molecule_ids = initial_parameters['mw'].keys()
    mesh_size = initial_parameters['mesh_size']

    # get molecule radii by molecular weights
    rp = calculate_rp_from_mw(molecule_ids, array_from(mw))

    # get diffusion constants per molecule
    diffusion_constants = compute_diffusion_constants_from_mw(
        molecule_ids, rp, mesh_size, edges)

    # plot the simulation output
    plot_output(output, out_dir)
    # plot_diff_range(diffusion_constants, rp, out_dir)
    # plot_nucleoid_diff(output, out_dir)
    # plot_radius_range(rp, output, out_dir)


# Plot functions
def plot_nucleoid_diff(output, out_dir):
    total_count = np.add(np.add(output['nucleoid']['molecules'][0],
                                output['cytosol_front']['molecules'][0]),
                         output['cytosol_rear']['molecules'][0])
    for i in range(len(output['nucleoid']['molecules'][0])):
        plt.plot(output['time'], [time[i] for time in output['nucleoid']['molecules']])
    size_str = np.rint(np.multiply(output['cytosol_front']['rp'][1], 2))
    plt.legend(['size = ' + str(size) + ' nm' for size in size_str], loc = 'lower right')
    plt.xlabel('time (s)')
    plt.ylabel('Number of molecules')
    plt.title('Percentage occupancy in nucleoid')
    out_file = out_dir + '/nucleoid_diff.png'
    plt.savefig(out_file)


def plot_diff_range(diffusion_constants, rp, out_dir):
    plt.plot(rp * 2, array_from(diffusion_constants['1']))
    plt.plot(rp * 2, array_from(diffusion_constants['3']))
    plt.yscale('log')
    plt.xlabel(r'Molecule size ($nm$)')
    plt.ylabel(r'Diffusion constant ($\mu m^2/s$)')
    plt.title('Diffusion constants into nucleoid')
    plt.legend(['With mesh', 'Without mesh'])
    out_file = out_dir + '/diff_range.png'
    plt.savefig(out_file)


def plot_radius_range(rp, output, out_dir):
    # import ipdb; ipdb.set_trace()
    for i in range(len(output['nucleoid']['molecules'])):
        plt.plot(output['time'], array_from(output['nucleoid']['molecules'])[i])
    size_str = np.rint(np.multiply(rp, 2))
    plt.legend(['size = ' + str(size) + ' nm' for size in size_str], loc = 'lower right')
    plt.xlabel('time (s)')
    plt.ylabel('Number of molecules')
    plt.title('Brownian diffusion into nucleoid with mesh')
    out_file = out_dir + '/radius_range.png'
    plt.savefig(out_file)


def plot_output(output, out_dir):
    plt.plot(output['time'], array_from(output['cytosol_front']['molecules'])[-5], 'b')
    plt.plot(output['time'], array_from(output['nucleoid']['molecules'])[-5], 'r')
    plt.plot(output['time'], array_from(output['cytosol_rear']['molecules'])[-5], 'g')
    plt.plot(output['time'], array_from(output['cytosol_front']['molecules'])[-1], 'b--')
    plt.plot(output['time'], array_from(output['nucleoid']['molecules'])[-1], 'r--')
    plt.plot(output['time'], array_from(output['cytosol_rear']['molecules'])[-1], 'g--')
    plt.xlabel('time (s)')
    plt.ylabel('Number of molecules')
    plt.title('Brownian diffusion over compartments')
    # plt.legend(['Cytosol front', 'Nucleoid',
    #             'Cytosol rear'])
    plt.legend(['Cytosol front: large molecule', 'Nucleoid: large molecule',
                'Cytosol rear: large molecule', 'Cytosol front: small molecule',
                'Nucleoid: small molecule', 'Cytosol rear: small molecule'])
    out_file = out_dir + '/simulation.png'
    plt.savefig(out_file)


# Diffusion constant functions
def compute_diffusion_constants_from_mw(molecule_ids, r_p, mesh_size, edges):
    '''
    Warnings: Temperature assumed to be 37°C.

    Sources:
    - Effective viscosities: Mullineaux et. al 2006, Microbial Cell Biology,
    doi: 10.1128/JB.188.10.3442-3448.2006
    - Mesh interference diffusion equation: Amsden, B 1999, Macromolecules,
    doi: 10.1021/ma980922a

    '''

    temp = 310.15 # in K
    K_B = scipy.constants.Boltzmann # in J/K
    n_eff_cytoplasm = 9.7   #cP
    n_eff_periplasm = 34    #cP
    cP_to_Pas = 1E-3
    m2_to_um2 = 1E12
    nm_to_m = 1E-9
    n = n_eff_cytoplasm * cP_to_Pas
    diffusion_constants = {}

    for edge_id, edge in edges.items():
        if 'periplasm' in edge['nodes']:
            n = n_eff_periplasm * cP_to_Pas

        # average radius of hole size
        r_o = mesh_size / 2

        # Einstein-Stokes equation for baseline diffusion constant
        dc = np.divide(K_B * temp * m2_to_um2, np.multiply(
            6 * np.pi * n * nm_to_m, r_p))

        # calculate impact on diffusion from mesh
        if 'nucleoid' in edge['nodes']:
            dc = np.multiply(dc, np.exp(np.multiply((-np.pi / 4),
                                        np.divide(r_p, r_o)**2)))
        diffusion_constants[edge_id] = array_to(molecule_ids, dc)

    return diffusion_constants


def calculate_rp_from_mw(molecule_ids, mw):
    # pulled from spatial_tool.py in WCM
    # TODO: find better way to do this for many molecule types
    dic_rp = {'protein': (0.0515, 0.392),
              'RNA': (0.0566, 0.38),
              'linear_DNA': (0.024, 0.57),
              'circular_DNA': (0.0125, 0.59),
              'supercoiled_DNA': (0.0145, 0.57),
              }

    r_p0, rp_power = dic_rp['protein']
    fg_to_kDa = 602217364.34
    r_p = np.multiply(r_p0, np.power(np.multiply(mw, fg_to_kDa), rp_power))
    return r_p

# Helper functions
def array_from(d):
    return np.array(list(d.values()))


def array_to(keys, array):
    return {
        key: array[index]
        for index, key in enumerate(keys)}


# The following functions are from spatial_tool.py in WCM for reference, written by Ray
def compute_diffusion_constant_from_rp(rp,
                                       loc = None,
                                       temp = None,
                                       parameters = None):
    '''
    Warning: The default values of the 'parameters' are E coli specific.

    This is the same function as 'compute_diffusion_constant_from_mw'
    except that it accepts the hydrodynamic radius of the macromolecules
    as the input. All hypothesis and statements in
    'compute_diffusion_constant_from_mw' apply here as well.

    Args:
        rp: the hydrodynamic radius of the macromolecule, units: nm.
        loc: The location of the molecule: 'nucleoid' or 'cytoplasm'.
        temp: The temperature of interest. unit: K.
        parameters: The 4 parameters required to compute the diffusion
            constant: xi, a, rh_nuc, rh_cyto.
            The default values are E coli specific.

    Returns:
        dc: the diffusion constant of the macromolecule, units: um**2/sec
    '''
    if loc is None:
        loc = 'cytosol'
    temp = 300 * units.K
    if parameters is None:
        parameters = (0.51, 0.53, 42, 10)

    # unpack constants required for the calculation
    xi, a, rh_nuc, rh_cyto = parameters  # unit: nm, 1, nm, nm

    # viscosity of water
    a_visc = 2.414*10**(-5)*units.Pa*units.s  # unit: Pa*sec
    b_visc = 247.8*units.K  # unit: K
    c_visc = 140*units.K  # unit: K
    eta_0 = a_visc*10**(b_visc/(temp - c_visc))  # unit: Pa*sec

    # determine Rh
    if loc == 'nucleoid':
        rh = rh_nuc  # unit: nm
    elif loc == 'cytoplasm':
        rh = rh_cyto  # unit: nm
    else:
        raise NameError(
            "The location can only be 'nucleoid' or 'cytoplasm'.")

    if not isinstance(rp, Unum):
        rp = units.nm * rp

    # compute DC(diffusion constant)
    dc_0 = K_B*temp/(6*np.pi*eta_0*rp)
    dc = dc_0*np.exp(
        -(xi**2/rh**2 + xi**2/rp.asNumber(units.nm)**2)**(-a/2))
    return dc


def compute_hydrodynamic_radius(mw, mtype = None):
    '''
    This function compute the hydrodynamic diameter of a macromolecules from
    its molecular weight. It is important to note that the hydrodynamic
    diameter is mainly used for computation of diffusion constant, and can
    be different from the observed diameter under microscopes or the radius
    of gyration, especially for loose polymers such as RNAs. This function
    is not E coli specific.

    References: Bioinformatics (2012). doi:10.1093/bioinformatics/bts537

    Args:
        mw: molecular weight of the macromolecules, units: Daltons.
        mtype: There are 5 possible mtype options: protein, RNA, linear_DNA,
            circular_DNA, and supercoiled_DNA.

    Returns: the hydrodynamic radius (in unit of nm) of the macromolecules
        using the following formula
        - rp = 0.0515*MW^(0.392) nm (Hong & Lei 2008) (protein)
        - rp = 0.0566*MW^(0.38) nm (Werner 2011) (RNA)
        - rp = 0.024*MW^(0.57) nm (Robertson et al 2006) (linear DNA)
        - rp = 0.0125*MW^(0.59) nm (Robertson et al 2006) (circular DNA)
        - rp = 0.0145*MW^(0.57) nm (Robertson et al 2006) (supercoiled DNA)
    '''
    if mtype is None:
        mtype = 'protein'

    dic_rp = {'protein': (0.0515, 0.392),
                'RNA': (0.0566, 0.38),
                'linear_DNA': (0.024, 0.57),
                'circular_DNA': (0.0125, 0.59),
                'supercoiled_DNA': (0.0145, 0.57),
                }

    if isinstance(mw, Unum):
        mw_unitless = mw.asNumber(units.g / units.mol)
    else:
        mw_unitless = mw

    if mtype in dic_rp:
        rp_0, rp_power = dic_rp[mtype]
    else:
        raise KeyError("The input 'mtype' should be one of the following 5 "
                        "options: 'protein', 'RNA', 'linear_DNA', "
                        "'circular_DNA', 'supercoiled_DNA'.")

    rp = rp_0*mw_unitless**rp_power
    rp = units.nm*rp
    return rp


def compute_diffusion_constant_from_mw(mw, mtype = None,
                                       loc = None,
                                       temp = None,
                                       parameters = None):
    '''
    Warning: The default values of the 'parameters' are E coli specific.

    This function computes the hypothesized diffusion constant of
    macromolecules within the nucleoid and the cytoplasm region.
    In literature, there is no known differentiation between the diffusion
    constant of a molecule in the nucleoid and in the cytoplasm up to the
    best of our knowledge in 2019. However, there is a good reason why we
    can assume that previously reported diffusion constant are in fact the
    diffusion constant of a protein in the nucleoid region:
    (1) The image traces of a protein within a bacteria usually cross the
        nucleoid regions.
    (2) The nucleoid region, compared to the cytoplasm, should be the main
    limiting factor restricting the magnitude of diffusion constant.
    (3) The same theory of diffusion constant has been implemented to
    mammalian cells, and the term 'rh', the average hydrodynamic radius of
    the biggest crowders, are different in mammalian cytoplasm, and it seems
    to reflect the hydrodynamic radius of the actin filament (note: the
    hydrodynamic radius of actin filament should be computed based on the
    average length of actin fiber, and is not equal to the radius of the
    actin filament itself.) (ref: Nano Lett. 2011, 11, 2157-2163).
    As for E coli, the 'rh' term = 40nm, which may correspond to the 80nm
    DNA fiber. On the other hand, for the diffusion constant of E coli in
    the true cytoplasm, we will expect the value of 'rh' term to be
    approximately 10 nm, which correspond to the radius of active ribosomes.

    However, all the above statements are just hypothesis. If you want to
    compute the diffusion constant of a macromolecule in the whole E coli
    cell, you should set loc = 'nucleoid': the formula for this is obtained
    from actual experimental data set. When you set loc = 'cytoplasm', the
    entire results are merely hypothesis.

    Ref: Kalwarczyk, T., Tabaka, M. & Holyst, R.
    Bioinformatics (2012). doi:10.1093/bioinformatics/bts537

    D_0 = K_B*T/(6*pi*eta_0*rp)
    ln(D_0/D_cyto) = ln(eta/eta_0) = (xi^2/Rh^2 + xi^2/rp^2)^(-a/2)
    D_0 = the diffusion constant of a macromolecule in pure solvent
    eta_0 = the viscosity of pure solvent, in this case, water
    eta = the size-dependent viscosity experienced by the macromolecule.
    xi = average distance between the surface of proteins
    rh = average hydrodynamic radius of the biggest crowders
    a = some constant of the order of 1
    rp = hydrodynamic radius of probed molecule

    In this formula, since we allow the changes in temperature, we also
    consider the viscosity changes of water under different temperature:
    Ref: Dortmund Data Bank
    eta_0 = A*10^(B/(T-C))
    A = 2.414*10^(-5) Pa*sec
    B = 247.8 K
    C = 140 K

    Args:
        mw: molecular weight(unit: Da) of the macromolecule
        mtype: There are 5 possible mtype options: protein, RNA, linear_DNA,
        circular_DNA, and supercoiled_DNA.
        loc: The location of the molecule: 'nucleoid' or 'cytoplasm'.
        temp: The temperature of interest. unit: K.
        parameters: The 4 parameters required to compute the diffusion
            constant: xi, a, rh_nuc, rh_cyto.
            The default values are E coli specific.

    Returns:
        dc: the diffusion constant of the macromolecule, units: um**2/sec
    '''
    if mtype is None:
        mtype = 'protein'
    if loc is None:
        loc = 'nucleoid'
    if temp is None:
        temp = 300 * units.K
    if parameters is None:
        parameters = (0.51, 0.53, 42, 10)

    rp = compute_hydrodynamic_radius(mw, mtype = mtype)
    dc = compute_diffusion_constant_from_rp(rp,
                                       loc = loc,
                                       temp = temp,
                                       parameters = parameters)
    return dc


def compute_nucleoid_size(l_cell, d_cell,
                          length_scaling_parameters = None,
                          nucleoid_area_ratio = None):
    '''
    Warning 1: This function contains default values that are specific to
    E coli grew on M9 media under aerobic condition only. The shape and size
    of nucleoid of a bacteria can be very different across species.
    Warning 2: This function is not suitable to compute the nucleoid size of
    E coli when its shape turn filamentaous. This is because the scaling
    formula of the length of the nucleoid is obtained from a dnaC mutant
    E coli strain. According to our reference, when the cells are allowed
    to replicate their DNA normally, a constant N/C area ratio is maintained
    even for filamentous variants (treated with cephalexin). However, if the
    DNA is not allowed to replicate, the N/C area ratio will decrease as the
    cell elongate. It is therefore recommended to carefully examine the
    condition when the length of the cells grow beyond 3 um.
    Warning 3: the default values of length_scaling_parameters and
    nucleoid_area_ratio are set to be E coli specific, grew on M9 media
    under aerobic condition. The N/C area ratio for E coli grew on LB under
    aerobic condition is close to 0.4. For E coli grew under anaerobic
    condition it is close to 0.5.
    Warning 4: It is also important to note that a single cell can contain
    more than 1 nucleoid, and this formula may not be suitable for these
    cases.

    Reference on nucleoid length:
    Wu, F. et al. Curr. Biol. (2019). doi:10.1016/j.cub.2019.05.015
    Reference on nucleoid/cytoplasm area ratio:
    Gray, W. T. et al. Cell (2019). doi:10.1016/j.cell.2019.05.017

    Args:
        l_cell: the length of the cell, in units of um
        d_cell: the width of the cell, in units of um
        length_scaling_parameters: the parameters used for the scaling
            formula of the length of the nucleoid with respect to the length
            of the whole cell.
        nucleoid_area_ratio: the nucleoid/cytoplasm area ratio measured
            under microscope.
    Returns:
        l_nuc: the length of the nucleoid
        d_nuc: the diameter of the nucleoid
    '''
    if length_scaling_parameters is None:
        length_scaling_parameters = (6.6, 8.3)
    if nucleoid_area_ratio is None:
        nucleoid_area_ratio = 0.60

    l_sat, l_c = length_scaling_parameters # unit: um
    if isinstance(l_cell, Unum):
        l_cell = l_cell.asNumber(units.um)
    if isinstance(d_cell, Unum):
        d_cell = d_cell.asNumber(units.um)
    if isinstance(l_sat, Unum):
        l_sat = l_sat.asNumber(units.um)
    if isinstance(l_c, Unum):
        l_c = l_c.asNumber(units.um)

    l_nuc = l_sat * (1 - np.exp(-l_cell / l_c))
    d_nuc = nucleoid_area_ratio * l_cell * d_cell / l_nuc
    return units.um*l_nuc, units.um*d_nuc


# run module is run as the main program with python vivarium/process/template_process.py
if __name__ == '__main__':
    # make an output directory to save plots
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    run_diffusion_network_process(out_dir)



