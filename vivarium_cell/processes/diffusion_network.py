import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants

from iteround import saferound

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

        # get molecule radii by molecular weights
        self.rp = calculate_rp_from_mw(self.molecule_ids, array_from(self.mw))

        # get diffusion constants per molecule
        self.diffusion_constants = compute_diffusion_constants_from_mw(
            self.molecule_ids, self.rp, self.mesh_size, self.edges)



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
                },
                'length': {
                    '_default': 1.0,
                },
                'molecules': {
                    '*': {
                        '_default': 0,
                        '_emit': True,
                        '_updater': 'accumulate',
                    }
                },
            } for node_id in self.parameters['nodes']
        }
        return schema

    def next_update(self, timestep, state):
        A = np.asarray([np.identity(len(self.nodes)) for mol in self.molecule_ids])

        # construct A matrix based off of graph, all edges assumed bidirectional
        for edge_id, edge in self.edges.items():
            node_index_1 = np.where(self.nodes == edge['nodes'][0])[0][0]
            node_index_2 = np.where(self.nodes == edge['nodes'][1])[0][0]
            cross_sectional_area = edge['cross_sectional_area']
            vol_1 = state[edge['nodes'][0]]['volume']
            vol_2 = state[edge['nodes'][1]]['volume']
            dx = state[edge['nodes'][0]]['length'] / 2 + state[edge['nodes'][1]]['length'] / 2
            diffusion_constants = array_from(self.diffusion_constants[edge_id])
            if 'diffusion_scaling_constant' in edge:
                diffusion_constants *= edge['diffusion_scaling_constant']
            if 'diffusion_constants' in edge:
                diffusion_constants = edge['diffusion_constants']
            alpha = diffusion_constants * (cross_sectional_area / dx) * timestep
            A[:, node_index_1, node_index_1] += alpha / vol_1
            A[:, node_index_2, node_index_2] += alpha / vol_2
            A[:, node_index_1, node_index_2] -= alpha / vol_1
            A[:, node_index_2, node_index_1] -= alpha / vol_2

        volumes = np.asarray(
            [state[node]['volume'] for node in state])
        conc_initial = np.asarray([np.multiply(array_from(state[node]['molecules']),
                                               array_from(self.mw)) / state[node]['volume']
                                   for node in state])
        conc_final = np.asarray([np.matmul(np.linalg.inv(a), conc_initial[:, i])
                                 for i, a in enumerate(A)]).T
        count_initial = np.asarray([array_from(state[node]['molecules'])
                                    for node in state])
        count_final = np.asarray([saferound(col, 0) for col in np.asarray(
            [np.divide(node * volumes[i],
                       array_from(self.mw))
             for i, node in enumerate(conc_final)]).T]).T
        delta = np.subtract(count_final, count_initial)

        assert (np.array_equal(np.ndarray.sum(count_initial, axis=0),
                np.ndarray.sum(count_final, axis=0))), 'Molecule count is not conserved'

        update = {
            node_id: {
                'molecules': array_to(self.molecule_ids,
                                      delta[np.where(self.nodes ==
                                                     node_id)[0][0]].astype(int)),
            } for node_id in self.nodes
        }
        return update


# TODO: change this to multiple tests and add asserts
def test_diffusion_network_process(out_dir='out'):
    # initialize the process by passing initial_parameters
    n = int(1E6)
    molecule_ids = ['7', '6', '5', '4', '3', '2', '1', '0']
    # molecule_ids = ['0']
    initial_parameters = {
        'nodes': ['cytosol_front', 'nucleoid', 'cytosol_rear'],
        'edges': {
            '1': {
                'nodes': ['cytosol_front', 'nucleoid'],
                'cross_sectional_area': np.pi * 0.3 ** 2,
            },
            '2': {
                'nodes': ['nucleoid', 'cytosol_rear'],
                'cross_sectional_area': np.pi * 0.3 ** 2,
            },
            # '3': {
            #     'nodes': ['cytosol_front', 'cytosol_rear'],
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
        'mesh_size': 50,
    }

    diffusion_network_process = DiffusionNetwork(initial_parameters)

    # run the simulation
    sim_settings = {
        'total_time': 10,
        'initial_state': {
            'cytosol_front': {
                'length': 0.75,
                'volume': 0.25,
                'molecules': {
                    mol_id: n
                    for mol_id in molecule_ids}
            },
            'nucleoid': {
                'length': 0.75,
                'volume': 0.5,
                'molecules': {
                    mol_id: 0
                    for mol_id in molecule_ids}
            },
            'cytosol_rear': {
                'length': 0.75,
                'volume': 0.25,
                'molecules': {
                    mol_id: 0
                    for mol_id in molecule_ids}
            },
        }
    }

    output = simulate_process_in_experiment(diffusion_network_process, sim_settings)
    rp = diffusion_network_process.rp
    diffusion_constants = diffusion_network_process.diffusion_constants

    # plot the simulation output
    plot_output(output, out_dir)
    # plot_diff_range(diffusion_constants, rp, out_dir)
    plot_nucleoid_diff(rp, output, out_dir)
    plot_radius_range(rp, output, out_dir)


# Plot functions
def plot_nucleoid_diff(rp, output, out_dir):
    plt.figure()
    plt.plot(rp*2, np.average(array_from(output['nucleoid']['molecules']), axis=1)/1E6)
    plt.xlabel('Molecule size (nm)')
    plt.ylabel('Percentage of time in nucleoid (%)')
    # plt.ylim(0, 0.35)
    plt.title('Percentage occupancy in nucleoid over 30 min with mesh')
    out_file = out_dir + '/nucleoid_diff.png'
    plt.savefig(out_file)


def plot_diff_range(diffusion_constants, rp, out_dir):
    plt.figure()
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
    plt.figure()
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
    plt.figure()
    plt.plot(output['time'], array_from(output['cytosol_front']['molecules'])[0], 'b')
    plt.plot(output['time'], array_from(output['nucleoid']['molecules'])[0], 'r')
    plt.plot(output['time'], array_from(output['cytosol_rear']['molecules'])[0], 'g')
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
    Warnings: Temperature assumed to be 37°C. These values are all E. coli specific.

    This function uses Einstein-Stokes to calculate a baseline diffusion constant
    based on viscosities found in E. coli from tracking GFP diffusion. GFP is
    significantly smaller than the range of mesh sizes, and is assumed to not
    be impacted by the presence of a DNA polymer mesh. The diffusion constants
    are then scaled using a mesh interference diffusion equation if the nucleoid
    is one of the nodes in an edge. Molecule radii and mesh radii are in nm.
    Viscosities are in cP. Temperature is in K.

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
    # TODO: find better way to do this for many molecule types
    dic_rp = {'protein': (0.0515, 0.392),
              'RNA': (0.0566, 0.38),
              'linear_DNA': (0.024, 0.57),
              'circular_DNA': (0.0125, 0.59),
              'supercoiled_DNA': (0.0145, 0.57),
              }

    # TODO: use different molecule types, currently uses protein assumption for all molecules
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


# run module is run as the main program with python vivarium/process/template_process.py
if __name__ == '__main__':
    # make an output directory to save plots
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    test_diffusion_network_process(out_dir)



