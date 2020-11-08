from __future__ import absolute_import, division, print_function

import os
import math
import random
import itertools

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
import numpy as np

from vivarium.library.dict_utils import get_value_from_path


DEFAULT_BOUNDS = [10, 10]

# constants
PI = math.pi

# colors for phylogeny initial agents
HUES_RANGE = [[0, 210], [270, 330]]  # skip blues (210-270), they do not show up well
HUE_INCREMENT = 30
HUES = [
    hue/360
    for hue in np.concatenate([
        np.linspace(hr[0], hr[1], int((hr[1]-hr[0])/HUE_INCREMENT)+1)
        for hr in HUES_RANGE])
    ]
DEFAULT_HUE = HUES[0]
DEFAULT_SV = [100.0/100.0, 70.0/100.0]
FLOURESCENT_SV = [0.75, 1.0]  # SV for fluorescent colors

def check_plt_backend():
    # reset matplotlib backend for non-interactive plotting
    plt.close('all')
    if plt.get_backend() == 'TkAgg':
        matplotlib.use('Agg')

class LineWidthData(Line2D):
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop('linewidth', 1)
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72./self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data))-trans((0, 0)))*ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)

def plot_agent(
        ax, data, color, agent_shape, membrane_width=0.1,
        membrane_color=[1, 1, 1],
    ):
    '''Plot an agent

    Args:
        ax: The axes to draw on.
        data (dict): The agent data dictionary.
        color (list): HSV color of agent body.
        agent_shape (str): One of ``rectangle``, ``segment``, and ``circle``.
        membrane_width (float): Width of drawn agent boundary.
        membrane_color (list): RGB color of drawn agent boundary.
    '''
    x_center = data['boundary']['location'][0]
    y_center = data['boundary']['location'][1]

    # get color, convert to rgb
    rgb = hsv_to_rgb(color)

    if agent_shape == 'rectangle':
        theta = data['boundary']['angle'] / PI * 180 + 90  # rotate 90 degrees to match field
        length = data['boundary']['length']
        width = data['boundary']['width']

        # get bottom left position
        x_offset = (width / 2)
        y_offset = (length / 2)
        theta_rad = math.radians(theta)
        dx = x_offset * math.cos(theta_rad) - y_offset * math.sin(theta_rad)
        dy = x_offset * math.sin(theta_rad) + y_offset * math.cos(theta_rad)

        x = x_center - dx
        y = y_center - dy

        # Create a rectangle
        shape = patches.Rectangle(
            (x, y), width, length,
            angle=theta,
            linewidth=membrane_width,
            edgecolor=membrane_color,
            facecolor=rgb
        )
        ax.add_patch(shape)

    elif agent_shape == 'segment':
        theta = data['boundary']['angle'] / PI * 180 + 90  # rotate 90 degrees to match field
        length = data['boundary']['length']
        width = data['boundary']['width']

        radius = width / 2

        # get the two ends
        length_offset = (length / 2) - radius
        theta_rad = math.radians(theta)
        dx = - length_offset * math.sin(theta_rad)
        dy = length_offset * math.cos(theta_rad)

        x1 = x_center - dx
        y1 = y_center - dy
        x2 = x_center + dx
        y2 = y_center + dy

        # segment plot
        membrane = LineWidthData(
            [x1, x2], [y1, y2],
            color=membrane_color,
            linewidth=width,
            solid_capstyle='round')
        line = LineWidthData(
            [x1, x2], [y1, y2],
            color=rgb,
            linewidth=width-membrane_width,
            solid_capstyle='round')
        ax.add_line(membrane)
        ax.add_line(line)

    elif agent_shape == 'circle':
        diameter = data['boundary']['diameter']

        # get bottom left position
        radius = (diameter / 2)
        x = x_center - radius
        y = y_center - radius

        # Create a circle
        circle = patches.Circle(
            (x, y), radius, linewidth=membrane_width,
            edgecolor=membrane_color,
        )
        ax.add_patch(circle)

def plot_agents(
    ax, agents, agent_colors=None, agent_shape='segment', dead_color=None,
    membrane_width=0.1, membrane_color=[1, 1, 1],
):
    '''Plot agents.

    Args:
        ax: the axis for plot
        agents (dict): a mapping from agent ID to that agent's data,
            which should have keys ``location``, ``angle``, ``length``,
            and ``width``.
        agent_colors (dict): Mapping from agent ID to HSV color.
        dead_color (list): List of 3 floats that define HSV color to use
            for dead cells. Dead cells only get treated differently if
            this is set.
        membrane_width (float): Width of agent outline to draw.
        membrane_color (list): List of 3 floats that define the RGB
            color to use for agent outlines.
    '''
    if not agent_colors:
        agent_colors = dict()
    for agent_id, agent_data in agents.items():
        color = agent_colors.get(agent_id, [DEFAULT_HUE]+DEFAULT_SV)
        if dead_color and 'boundary' in agent_data and 'dead' in agent_data['boundary']:
            if agent_data['boundary']['dead']:
                color = dead_color
        plot_agent(ax, agent_data, color, agent_shape, membrane_width,
                membrane_color)

def mutate_color(baseline_hsv):
    mutation = 0.1
    new_hsv = [
        (n + np.random.uniform(-mutation, mutation))
        for n in baseline_hsv]
    # wrap hue around
    new_hsv[0] = new_hsv[0] % 1
    # reflect saturation and value
    if new_hsv[1] > 1:
        new_hsv[1] = 2 - new_hsv[1]
    if new_hsv[2] > 1:
        new_hsv[2] = 2 - new_hsv[2]
    return new_hsv

def color_phylogeny(ancestor_id, phylogeny, baseline_hsv, phylogeny_colors={}):
    """
    get colors for all descendants of the ancestor
    through recursive calls to each generation
    """
    phylogeny_colors.update({ancestor_id: baseline_hsv})
    daughter_ids = phylogeny.get(ancestor_id)
    if daughter_ids:
        for daughter_id in daughter_ids:
            daughter_color = mutate_color(baseline_hsv)
            color_phylogeny(daughter_id, phylogeny, daughter_color)
    return phylogeny_colors

def get_phylogeny_colors_from_names(agent_ids):
    '''Get agent colors using phlogeny saved in agent_ids
    This assumes the names use daughter_phylogeny_id() from meta_division
    '''

    # make phylogeny with {mother_id: [daughter_1_id, daughter_2_id]}
    phylogeny = {agent_id: [] for agent_id in agent_ids}
    for agent1, agent2 in itertools.combinations(agent_ids, 2):
        if agent1 == agent2[0:-1]:
            phylogeny[agent1].append(agent2)
        elif agent2 == agent1[0:-1]:
            phylogeny[agent2].append(agent1)

    # get initial ancestors
    daughters = list(phylogeny.values())
    daughters = set([item for sublist in daughters for item in sublist])
    mothers = set(list(phylogeny.keys()))
    ancestors = list(mothers - daughters)

    # agent colors based on phylogeny
    agent_colors = {agent_id: [] for agent_id in agent_ids}
    for agent_id in ancestors:
        hue = random.choice(HUES)  # select random initial hue
        initial_color = [hue] + DEFAULT_SV
        agent_colors.update(color_phylogeny(agent_id, phylogeny, initial_color))

    return agent_colors


def plot_snapshots(data, plot_config):
    '''Plot snapshots of the simulation over time

    The snapshots depict the agents and environmental molecule
    concentrations.

    Arguments:
        data (dict): A dictionary with the following keys:

            * **agents** (:py:class:`dict`): A mapping from times to
              dictionaries of agent data at that timepoint. Agent data
              dictionaries should have the same form as the hierarchy
              tree rooted at ``agents``.
            * **fields** (:py:class:`dict`): A mapping from times to
              dictionaries of environmental field data at that
              timepoint.  Field data dictionaries should have the same
              form as the hierarchy tree rooted at ``fields``.
            * **config** (:py:class:`dict`): The environmental
              configuration dictionary  with the following keys:

                * **bounds** (:py:class:`tuple`): The dimensions of the
                  environment.

        plot_config (dict): Accepts the following configuration options.
            Any options with a default is optional.

            * **n_snapshots** (:py:class:`int`): Number of snapshots to
              show per row (i.e. for each molecule). Defaults to 6.
            * **agent_shape** (:py:class:`str`): the shape of the agents.
              select from **rectangle**, **segment**
            * **phylogeny_names** (:py:class:`bool`): This selects agent
              colors based on phylogenies seved in their names using
              meta_division.py daughter_phylogeny_id()
            * **out_dir** (:py:class:`str`): Output directory, which is
              ``out`` by default.
            * **filename** (:py:class:`str`): Base name of output file.
              ``snapshots`` by default.
            * **skip_fields** (:py:class:`Iterable`): Keys of fields to
              exclude from the plot. This takes priority over
              ``include_fields``.
            * **include_fields** (:py:class:`Iterable`): Keys of fields
              to plot.
            * **field_label_size** (:py:class:`float`): Font size of the
              field label.
            * **dead_color** (:py:class:`list` of 3 :py:class:`float`s):
              Color for dead cells in HSV. Defaults to [0, 0, 0], which
              is black.
            * **default_font_size** (:py:class:`float`): Font size for
              titles and axis labels.
    '''
    check_plt_backend()

    n_snapshots = plot_config.get('n_snapshots', 6)
    out_dir = plot_config.get('out_dir', False)
    filename = plot_config.get('filename', 'snapshots')
    agent_shape = plot_config.get('agent_shape', 'segment')
    phylogeny_names = plot_config.get('phylogeny_names', True)
    skip_fields = plot_config.get('skip_fields', [])
    include_fields = plot_config.get('include_fields', None)
    field_label_size = plot_config.get('field_label_size', 20)
    default_font_size = plot_config.get('default_font_size', 36)
    dead_color = plot_config.get('dead_color', [0, 0, 0])

    # get data
    agents = data.get('agents', {})
    fields = data.get('fields', {})
    config = data.get('config', {})
    bounds = config.get('bounds', DEFAULT_BOUNDS)
    edge_length_x = bounds[0]
    edge_length_y = bounds[1]

    # time steps that will be used
    if agents and fields:
        assert set(list(agents.keys())) == set(list(fields.keys())), 'agent and field times are different'
        time_vec = list(agents.keys())
    elif agents:
        time_vec = list(agents.keys())
    elif fields:
        time_vec = list(fields.keys())
    else:
        raise Exception('No agents or field data')

    time_indices = np.round(np.linspace(0, len(time_vec) - 1, n_snapshots)).astype(int)
    snapshot_times = [time_vec[i] for i in time_indices]

    # get fields id and range
    field_ids = []
    if fields:
        if include_fields is None:
            field_ids = set(fields[time_vec[0]].keys())
        else:
            field_ids = set(include_fields)
        field_ids -= set(skip_fields)
        field_range = {}
        for field_id in field_ids:
            field_min = min([min(min(field_data[field_id])) for t, field_data in fields.items()])
            field_max = max([max(max(field_data[field_id])) for t, field_data in fields.items()])
            field_range[field_id] = [field_min, field_max]

    # get agent ids
    agent_ids = set()
    if agents:
        for time, time_data in agents.items():
            current_agents = list(time_data.keys())
            agent_ids.update(current_agents)
        agent_ids = list(agent_ids)

        # set agent colors
        if phylogeny_names:
            agent_colors = get_phylogeny_colors_from_names(agent_ids)
        else:
            agent_colors = {}
            for agent_id in agent_ids:
                hue = random.choice(HUES)  # select random initial hue
                color = [hue] + DEFAULT_SV
                agent_colors[agent_id] = color

    # make the figure
    n_rows = max(len(field_ids), 1)
    n_cols = n_snapshots + 1  # one column for the colorbar
    figsize = (12 * n_cols, 12 * n_rows)
    max_dpi = min([2**16 // dim for dim in figsize]) - 1
    fig = plt.figure(figsize=figsize, dpi=min(max_dpi, 100))
    grid = plt.GridSpec(n_rows, n_cols, wspace=0.2, hspace=0.2)
    original_fontsize = plt.rcParams['font.size']
    plt.rcParams.update({'font.size': default_font_size})

    # plot snapshot data in each subsequent column
    for col_idx, (time_idx, time) in enumerate(zip(time_indices, snapshot_times)):
        if field_ids:
            for row_idx, field_id in enumerate(field_ids):

                ax = init_axes(
                    fig, edge_length_x, edge_length_y, grid, row_idx,
                    col_idx, time, field_id, field_label_size,
                )
                ax.tick_params(
                    axis='both', which='both', bottom=False, top=False,
                    left=False, right=False,
                )

                # transpose field to align with agents
                field = np.transpose(np.array(fields[time][field_id])).tolist()
                vmin, vmax = field_range[field_id]
                im = plt.imshow(field,
                                origin='lower',
                                extent=[0, edge_length_x, 0, edge_length_y],
                                vmin=vmin,
                                vmax=vmax,
                                cmap='BuPu')
                if agents:
                    agents_now = agents[time]
                    plot_agents(
                        ax, agents_now, agent_colors, agent_shape, dead_color)

                # colorbar in new column after final snapshot
                if col_idx == n_snapshots - 1:
                    cbar_col = col_idx + 1
                    ax = fig.add_subplot(grid[row_idx, cbar_col])
                    if row_idx == 0:
                        ax.set_title('Concentration (mmol/L)', y=1.08)
                    ax.axis('off')
                    if vmin == vmax:
                        continue
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("left", size="5%", pad=0.0)
                    fig.colorbar(im, cax=cax, format='%.6f')
                    ax.axis('off')
        else:
            row_idx = 0
            ax = init_axes(
                fig, bounds[0], bounds[1], grid, row_idx, col_idx,
                time, ""
            )

            if agents:
                agents_now = agents[time]
                plot_agents(ax, agents_now, agent_colors, agent_shape, dead_color)

    plt.rcParams.update({'font.size': original_fontsize})
    if out_dir:
        fig_path = os.path.join(out_dir, filename)
        fig.subplots_adjust(wspace=0.7, hspace=0.1)
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)


def plot_tags(data, plot_config):
    '''Plot snapshots of the simulation over time

    The snapshots depict the agents and the levels of tagged molecules
    in each agent by agent color intensity.

    Arguments:
        data (dict): A dictionary with the following keys:

            * **agents** (:py:class:`dict`): A mapping from times to
              dictionaries of agent data at that timepoint. Agent data
              dictionaries should have the same form as the hierarchy
              tree rooted at ``agents``.
            * **config** (:py:class:`dict`): The environmental
              configuration dictionary  with the following keys:

                * **bounds** (:py:class:`tuple`): The dimensions of the
                  environment.

        plot_config (dict): Accepts the following configuration options.
            Any options with a default is optional.

            * **n_snapshots** (:py:class:`int`): Number of snapshots to
              show per row (i.e. for each molecule). Defaults to 6.
            * **out_dir** (:py:class:`str`): Output directory, which is
              ``out`` by default.
            * **filename** (:py:class:`str`): Base name of output file.
              ``tags`` by default.
            * **tagged_molecules** (:py:class:`typing.Iterable`): The
              tagged molecules whose concentrations will be indicated by
              agent color. Each molecule should be specified as a
              :py:class:`tuple` of the path in the agent compartment
              to where the molecule's count can be found, with the last
              value being the molecule's count variable.
            * **convert_to_concs** (:py:class:`bool`): if True, convert counts
              to concentrations.
            * **background_color** (:py:class:`str`): use matplotlib colors,
              ``black`` by default
            * **tag_label_size** (:py:class:`float`): The font size for
              the tag name label
            * **default_font_size** (:py:class:`float`): Font size for
              titles and axis labels.
            * **membrane_width** (:py:class:`float`): Width to use for
                drawing agent edges.
            * **membrane_color** (:py:class:`list`): RGB color to use
                for drawing agent edges.
            * **tag_colors** (:py:class:`dict`): Mapping from tag ID to
                the HSV color to use for that tag as a list.
    '''
    check_plt_backend()

    n_snapshots = plot_config.get('n_snapshots', 6)
    out_dir = plot_config.get('out_dir', False)
    filename = plot_config.get('filename', 'tags')
    agent_shape = plot_config.get('agent_shape', 'segment')
    background_color = plot_config.get('background_color', 'black')
    tagged_molecules = plot_config['tagged_molecules']
    tag_path_name_map = plot_config.get('tag_path_name_map', {})
    tag_label_size = plot_config.get('tag_label_size', 20)
    default_font_size = plot_config.get('default_font_size', 36)
    convert_to_concs = plot_config.get('convert_to_concs', True)
    membrane_width = plot_config.get('membrane_width', 0.1)
    membrane_color = plot_config.get('membrane_color', [1, 1, 1])
    tag_colors = plot_config.get('tag_colors', dict())

    if tagged_molecules == []:
        raise ValueError('At least one molecule must be tagged.')

    # get data
    agents = data['agents']
    config = data.get('config', {})
    bounds = config['bounds']
    edge_length_x, edge_length_y = bounds

    # time steps that will be used
    time_vec = list(agents.keys())
    time_indices = np.round(
        np.linspace(0, len(time_vec) - 1, n_snapshots)
    ).astype(int)
    snapshot_times = [time_vec[i] for i in time_indices]

    # get tag ids and range
    tag_ranges = {}

    for time, time_data in agents.items():
        for agent_id, agent_data in time_data.items():
            volume = agent_data.get('boundary', {}).get('volume', 0)
            for tag_id in tagged_molecules:
                level = get_value_from_path(agent_data, tag_id)
                if convert_to_concs:
                    level = level / volume if volume else 0
                if tag_id in tag_ranges:
                    tag_ranges[tag_id] = [
                        min(tag_ranges[tag_id][0], level),
                        max(tag_ranges[tag_id][1], level)]
                else:
                    # add new tag
                    tag_ranges[tag_id] = [level, level]

                    # select random initial hue
                    if tag_id not in tag_colors:
                        hue = random.choice(HUES)
                        tag_color = [hue] + FLOURESCENT_SV
                        tag_colors[tag_id] = tag_color

    # make the figure
    n_rows = len(tagged_molecules)
    n_cols = n_snapshots + 1  # one column for the colorbar
    figsize = (12 * n_cols, 12 * n_rows)
    max_dpi = min([2**16 // dim for dim in figsize]) - 1
    fig = plt.figure(figsize=figsize, dpi=min(max_dpi, 100))
    grid = plt.GridSpec(n_rows, n_cols, wspace=0.2, hspace=0.2)
    original_fontsize = plt.rcParams['font.size']
    plt.rcParams.update({'font.size': default_font_size})

    # plot tags
    for row_idx, tag_id in enumerate(tag_ranges.keys()):
        used_agent_colors = []
        concentrations = []
        for col_idx, (time_idx, time) in enumerate(
            zip(time_indices, snapshot_times)
        ):
            tag_name = tag_path_name_map.get(tag_id, tag_id)
            ax = init_axes(
                fig, edge_length_x, edge_length_y, grid,
                row_idx, col_idx, time, tag_name, tag_label_size,
            )
            ax.tick_params(
                axis='both', which='both', bottom=False, top=False,
                left=False, right=False,
            )
            ax.set_facecolor(background_color)

            # update agent colors based on tag_level
            min_tag, max_tag = tag_ranges[tag_id]
            agent_tag_colors = {}
            tag_h, tag_s, _ = tag_colors[tag_id]
            for agent_id, agent_data in agents[time].items():
                # get current tag concentration, and determine color
                level = get_value_from_path(agent_data, tag_id)
                if convert_to_concs:
                    volume = agent_data.get('boundary', {}).get('volume', 0)
                    level = level / volume if volume else 0
                if min_tag != max_tag:
                    concentrations.append(level)
                    intensity = (level - min_tag)/ (max_tag - min_tag)
                    agent_color = tag_h, tag_s, intensity
                    agent_rgb = matplotlib.colors.hsv_to_rgb(agent_color)
                    used_agent_colors.append(agent_rgb)
                else:
                    agent_color = tag_h, tag_s, 0

                agent_tag_colors[agent_id] = agent_color

            plot_agents(ax, agents[time], agent_tag_colors, agent_shape,
                    None, membrane_width, membrane_color)

            # colorbar in new column after final snapshot
            if col_idx == n_snapshots - 1:
                cbar_col = col_idx + 1
                ax = fig.add_subplot(grid[row_idx, cbar_col])
                if row_idx == 0:
                    if convert_to_concs:
                        ax.set_title('Concentration (counts/fL)', y=1.08)
                ax.axis('off')
                if min_tag == max_tag:
                    continue
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("left", size="5%", pad=0.0)
                norm = matplotlib.colors.Normalize(
                    vmin=min_tag, vmax=max_tag)
                # Sort colors and concentrations by concentration
                sorted_idx = np.argsort(concentrations)
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'row_{}'.format(row_idx),
                    [
                        np.array(used_agent_colors)[sorted_idx][0],
                        np.array(used_agent_colors)[sorted_idx][-1],
                    ],
                )
                mappable = matplotlib.cm.ScalarMappable(norm, cmap)
                fig.colorbar(mappable, cax=cax, format='%.6f')

    plt.rcParams.update({'font.size': original_fontsize})
    if out_dir:
        fig_path = os.path.join(out_dir, filename)
        fig.subplots_adjust(wspace=0.7, hspace=0.1)
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)

def initialize_spatial_figure(bounds, fontsize=18):

    x_length = bounds[0]
    y_length = bounds[1]

    # set up figure
    n_ticks = 4
    plot_buffer = 0.02
    buffer = plot_buffer * min(bounds)
    min_edge = min(x_length, y_length)
    x_scale = x_length/min_edge
    y_scale = y_length/min_edge

    # make the figure
    fig = plt.figure(figsize=(8*x_scale, 8*y_scale))
    plt.rcParams.update({'font.size': fontsize, "font.family": "Times New Roman"})

    plt.xlim((0-buffer, x_length+buffer))
    plt.ylim((0-buffer, y_length+buffer))
    plt.xlabel(u'\u03bcm')
    plt.ylabel(u'\u03bcm')

    # specify the number of ticks for each edge
    [x_bins, y_bins] = [int(n_ticks * edge / min_edge) for edge in [x_length, y_length]]
    plt.locator_params(axis='y', nbins=y_bins)
    plt.locator_params(axis='x', nbins=x_bins)
    ax = plt.gca()

    return fig, ax

def get_agent_trajectories(agents, times):
    trajectories = {}
    for agent_id, series in agents.items():
        time_indices = series['boundary']['location']['time_index']
        series_times = [times[time_index] for time_index in time_indices]

        positions = series['boundary']['location']['value']
        angles = series['boundary']['angle']['value']
        series_values = [[x, y, theta] for ((x, y), theta) in zip(positions, angles)]

        trajectories[agent_id] = {
            'time': series_times,
            'value': series_values,
        }
    return trajectories

def get_agent_type_colors(agent_ids):
    """ get colors for each agent id by agent type
    Assumes that agents of the same type share the beginning
    of their name, followed by '_x' with x as a single number
    TODO -- make this more general for more digits and other comparisons"""
    agent_type_colors = {}
    agent_types = {}
    for agent1, agent2 in itertools.combinations(agent_ids, 2):
        if agent1[0:-2] == agent2[0:-2]:
            agent_type = agent1[0:-2]
            if agent_type not in agent_type_colors:
                color = plt.rcParams['axes.prop_cycle'].by_key()['color'][len(agent_type_colors)]
                agent_type_colors[agent_type] = color
            else:
                color = agent_type_colors[agent_type]
            agent_types[agent1] = agent_type
            agent_types[agent2] = agent_type
    for agent in agent_ids:
        if agent not in agent_types:
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][len(agent_type_colors)]
            agent_type_colors[agent] = color
            agent_types[agent] = agent

    return agent_types, agent_type_colors

def plot_agent_trajectory(agent_timeseries, config, out_dir='out', filename='trajectory'):
    check_plt_backend()

    # trajectory plot settings
    legend_fontsize = 18
    markersize = 25

    bounds = config.get('bounds', DEFAULT_BOUNDS)
    field = config.get('field')
    rotate_90 = config.get('rotate_90', False)

    # get agents
    times = np.array(agent_timeseries['time'])
    agents = agent_timeseries['agents']
    agent_types, agent_type_colors = get_agent_type_colors(list(agents.keys()))

    if rotate_90:
        field = rotate_field_90(field)
        for agent_id, series in agents.items():
            agents[agent_id] = rotate_agent_series_90(series, bounds)
        bounds = rotate_bounds_90(bounds)

    # get each agent's trajectory
    trajectories = get_agent_trajectories(agents, times)

    # initialize a spatial figure
    fig, ax = initialize_spatial_figure(bounds, legend_fontsize)

    # move x axis to top
    ax.tick_params(labelbottom=False,labeltop=True,bottom=False,top=True)
    ax.xaxis.set_label_coords(0.5, 1.12)

    if field is not None:
        field = np.transpose(field)
        shape = field.shape
        im = plt.imshow(field,
                        origin='lower',
                        extent=[0, shape[1], 0, shape[0]],
                        cmap='Greys')
        # colorbar for field concentrations
        cbar = plt.colorbar(im, pad=0.02, aspect=50, shrink=0.7)
        cbar.set_label('concentration', rotation=270, labelpad=20)

    for agent_id, trajectory_data in trajectories.items():
        agent_trajectory = trajectory_data['value']

        # convert trajectory to 2D array
        locations_array = np.array(agent_trajectory)
        x_coord = locations_array[:, 0]
        y_coord = locations_array[:, 1]

        # get agent type and color
        agent_type = agent_types[agent_id]
        agent_color = agent_type_colors[agent_type]

        # plot line
        ax.plot(x_coord, y_coord, linewidth=2, color=agent_color, label=agent_type)
        ax.plot(x_coord[0], y_coord[0],
                 color=(0.0, 0.8, 0.0), marker='.', markersize=markersize)  # starting point
        ax.plot(x_coord[-1], y_coord[-1],
                 color='r', marker='.', markersize=markersize)  # ending point

    # create legend for agent types
    agent_labels = [
        mlines.Line2D([], [], color=agent_color, linewidth=2, label=agent_type)
        for agent_type, agent_color in agent_type_colors.items()]
    agent_legend = plt.legend(
        title='agent type', handles=agent_labels, loc='upper center',
        bbox_to_anchor=(0.3, 0.0), ncol=2, prop={'size': legend_fontsize})
    ax.add_artist(agent_legend)

    # create a legend for start/end markers
    start = mlines.Line2D([], [],
            color=(0.0, 0.8, 0.0), marker='.', markersize=markersize, linestyle='None', label='start')
    end = mlines.Line2D([], [],
            color='r', marker='.', markersize=markersize, linestyle='None', label='end')
    marker_legend = plt.legend(
        title='trajectory', handles=[start, end], loc='upper center',
        bbox_to_anchor=(0.7, 0.0), ncol=2, prop={'size': legend_fontsize})
    ax.add_artist(marker_legend)

    fig_path = os.path.join(out_dir, filename)
    plt.subplots_adjust(wspace=0.7, hspace=0.1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)


def rotate_bounds_90(bounds):
    return [bounds[1], bounds[0]]

def rotate_field_90(field):
    return np.rot90(field, 3)  # rotate 3 times for 270

def rotate_agent_series_90(series, bounds):
    location_series = series['boundary']['location']
    angle_series = series['boundary']['angle']

    if isinstance(location_series, dict):
        # this ran with time_indexed_timeseries_from_data
        series['boundary']['location']['value'] = [[y, bounds[0] - x] for [x, y] in location_series['value']]
        series['boundary']['angle']['value'] = [theta + PI / 2 for theta in angle_series['value']]
    else:
        series['boundary']['location'] = [[y, bounds[0] - x] for [x, y] in location_series]
        series['boundary']['angle'] = [theta + PI / 2 for theta in angle_series]
    return series

def plot_temporal_trajectory(agent_timeseries, config, out_dir='out', filename='temporal'):
    check_plt_backend()

    bounds = config.get('bounds', DEFAULT_BOUNDS)
    field = config.get('field')
    rotate_90 = config.get('rotate_90', False)

    # get agents
    times = np.array(agent_timeseries['time'])
    agents = agent_timeseries['agents']

    if rotate_90:
        field = rotate_field_90(field)
        for agent_id, series in agents.items():
            agents[agent_id] = rotate_agent_series_90(series, bounds)
        bounds = rotate_bounds_90(bounds)

    # get each agent's trajectory
    trajectories = get_agent_trajectories(agents, times)

    # initialize a spatial figure
    fig, ax = initialize_spatial_figure(bounds)

    if field is not None:
        field = np.transpose(field)
        shape = field.shape
        im = plt.imshow(field,
                        origin='lower',
                        extent=[0, shape[1], 0, shape[0]],
                        cmap='Greys'
                        )

    for agent_id, trajectory_data in trajectories.items():
        agent_trajectory = trajectory_data['value']

        # convert trajectory to 2D array
        locations_array = np.array(agent_trajectory)
        x_coord = locations_array[:, 0]
        y_coord = locations_array[:, 1]

        # make multi-colored trajectory
        points = np.array([x_coord, y_coord]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=plt.get_cmap('cool'))
        lc.set_array(times)
        lc.set_linewidth(6)

        # plot line
        line = plt.gca().add_collection(lc)

    # color bar
    cbar = plt.colorbar(line, ticks=[times[0], times[-1]], aspect=90, shrink=0.4)
    cbar.set_label('time (s)', rotation=270)

    fig_path = os.path.join(out_dir, filename)
    plt.subplots_adjust(wspace=0.7, hspace=0.1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)

def init_axes(
    fig, edge_length_x, edge_length_y, grid, row_idx, col_idx, time,
    molecule, ylabel_size=20
):
    ax = fig.add_subplot(grid[row_idx, col_idx])
    if row_idx == 0:
        plot_title = 'time: {:.4f} s'.format(float(time))
        plt.title(plot_title, y=1.08)
    if col_idx == 0:
        ax.set_ylabel(
            molecule, fontsize=ylabel_size, rotation='horizontal',
            horizontalalignment='right',
        )
    ax.set(xlim=[0, edge_length_x], ylim=[0, edge_length_y], aspect=1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    return ax
