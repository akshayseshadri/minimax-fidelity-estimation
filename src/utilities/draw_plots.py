"""
    Provides convenient functions to handle data and plotting using matplotlib

    Author: Akshay Seshadri
"""

from cycler import cycler
from pathlib import PurePath
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def handle_keypress(event):
    """
        Handle key events sent to the figure window(s).
    """
    if event.key == 'w':
        plt.close()
    elif event.key=='q':
        plt.close("all")

class Plot_Manager():
    """
        Facilitates drawing multiple, different types of plots using matplotlib. This is done by providing just handful of functions that help in making different types of plots (without having to remember
        the syntax and various options) offered by matplotlib.

        Intended for use in other code while plotting, but can also be useful when doing quick plotting using the Python/IPython interpreter.
        Pyplot has a MATLAB-like interface, but this class uses the object-oriented interface provided by matplotlib as it offers more customizability.
        However, the object-oriented interface used by this class is, for most part, not exposed "outside" (but it is still accessible through the class attributes).
        In that sense, it emulates the behaviour of pyplot to a certain extent. Consider using pyplot directly if that is easier or better suits the intended application.

        TODO: Add matplotlibrc file support to set default values.
    """
    def __init__(self, color_cycle = 'default', use_tex_fonts = True):
        """
            Initializes various attributes to be used later.
        """
        self.data_key_list    = list()            # unique identifier for data
        self.data_desc_dict   = dict()            # optional description of the data, indexed by data id
        self.xdata_dict       = dict()            # a dictionary containing x-data, indexed by data key
        self.ydata_dict       = dict()            # a dictionary containing y-data, indexed by data key
        self.data_params_dict = dict()            # a dictionary of parameters, indexed by data key, associated with the data to be used in plotting

        self.num_data = 0

        # in case figure needs to be "held", save the figure and axis object
        self.fig = None
        self.ax  = None
        # in case twin-axis needs to be used
        self.axt = None
        # in case inset needs to be made
        self.axin = None

        # list storing the figure and axes objects for all the plots
        self.fig_dict = dict()
        self.ax_dict = dict()

        # controls when to plot (versus parsing arguments)
        self.ready_to_plot = False

        # dictionary holding a few commonly used color cycles
        # default color cycle (to be used if no color cycle is specified) is based on seaborn theme, but starts with black
        # most styles used here are available by default in matplotlib: https://github.com/matplotlib/matplotlib/tree/master/lib/matplotlib/mpl-data/stylelib
        self.color_cycle_dict = {'default': ['#000000', '#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD'],\
                                 'mpl_default': ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF'],\
                                 'seaborn': ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD'],\
                                 'seaborn-dark-palette': ['#001C7F', '#017517', '#8C0900', '#7600A1', '#B8860B', '#006374'],\
                                 'ggplot': ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']}
                                 
        # use the color cycle provided
        if type(color_cycle) in [list, tuple]:
            self.color_cycle = color_cycle
        elif str(color_cycle).lower() in self.color_cycle_dict.keys():
            self.color_cycle = self.color_cycle_dict[str(color_cycle).lower()]
        else:
            raise ValueError("Please provide a list of hexadecimal color strings for specifying the color cycle " +\
                             "or choose one of 'default', 'mpl_default', 'seaborn', 'seaborn-dark-palette', 'ggplot'")

        # dictionary containing valid types of plot
        # format: 'key used for performing actual plotting' : [list of allowed strings that can be used to specify this type of plot]
        self.valid_plot_types = {'lineplot': ['plot', 'line', 'lineplot', 'line plot', '2dplot', '2d plot'],\
                                 'discrete': ['discrete', 'discrete_data', 'discrete plot', 'discreteplot'],\
                                 'hist': ['hist', 'histogram'],\
                                 'errorbar': ['errbar', 'errorbar', 'error bar']}

        # default values of parameters that can be used to modify the appearance of the plots
        self.default_params = {'figsize': (18, 6), 'plot_type': 'lineplot', 'subplots': (1, 1, 1), 'fmt': '', 'linestyle': '-', 'marker': None, 'title': '',\
                               'xlabel': '', 'ylabel': '', 'xlim': None, 'ylim': None, 'hold': True, 'grid': False, 'plot_params': {}, 'label': '', 'legend': False,\
                               'color_cycle': self.color_cycle, 'color': None, 'xticks': None, 'yticks': None, 'xticklabels': None, 'yticklabels': None,\
                               'xtick_params': {'labelsize': 38, 'pad': 18, 'direction': 'out', 'length': 6, 'width': 2},\
                               'ytick_params': {'labelsize': 38, 'pad': 10, 'direction': 'out', 'length': 6, 'width': 2},\
                               'xlabel_size': 30, 'ylabel_size': 30, 'title_size': 30, 'linewidth': 3, 'markersize': 15, 'yscale': None, 'twinx': False, 'inset_axes': None,\
                               'style': 'seaborn-white', 'scientific_notn': {'scilimits': (-2, 3), 'offsetsize': 24}, 'show_data_key': False, 'save_fig': False,\
                               'dirname': './', 'filename': None, 'fig_format': 'pdf', 'bbox_inches': 'tight', 'show_fig': True}

        # list of keyword arguments accepted by a Line2D object
        Line2D_kwargs = ['agg_filter', 'alpha', 'animated', 'antialiased', 'clip_box', 'clip_on', 'clip_path', 'color', 'contains', 'dash_capstyle', 'dash_joinstyle', 'dashes', 'data',\
                         'drawstyle', 'figure', 'fillstyle', 'gid', 'in_layout', 'label', 'linestyle', 'linewidth', 'marker', 'markeredgecolor', 'markeredgewidth', 'markerfacecolor',\
                         'markerfacecoloralt', 'markersize', 'markevery', 'path_effects', 'picker', 'pickradius', 'rasterized', 'sketch_params', 'snap', 'solid_capstyle', 'solid_joinstyle',\
                         'transform', 'url', 'visible', 'xdata', 'ydata', 'zorder']

        # list of keyword arguments accepted by a Patch object
        Patch_kwargs = ['agg_filter', 'alpha', 'animated', 'antialiased', 'capstyle', 'clip_box', 'clip_on', 'clip_path', 'color', 'contains', 'edgecolor', 'facecolor', 'figure', 'fill', 'gid',\
                        'hatch', 'in_layout', 'joinstyle', 'label', 'linestyle', 'linewidth', 'path_effects', 'picker', 'rasterized', 'sketch_params', 'snap', 'transform', 'url', 'visible', 'zorder']

        # a dictionary (indexed by plot_type) of keyword arguments that axes.(plot_type) accepts, which are passed on to this plotting function through plot_params dictionary
        self.plot_type_kwargs = {'lineplot': ['scalex', 'scaley'] + Line2D_kwargs,\
                                 'discrete': ['scalex', 'scaley'] + Line2D_kwargs,\
                                 'hist': ['bins', 'range', 'density', 'weights', 'cumulative', 'bottom', 'histtype', 'align', 'orientation', 'rwidth', 'log', 'color', 'label', 'stacked'] + Patch_kwargs,\
                                 'errorbar': ['yerr', 'xerr', 'fmt', 'ecolor', 'elinewidth', 'capsize', 'barsabove', 'lolims', 'uplims', 'xlolims', 'xuplims', 'errorevery', 'capthick'] + Line2D_kwargs}

        # use tex fonts if required
        if use_tex_fonts:
            mpl.rcParams['font.family'] = 'sans-serif'
            mpl.rcParams['font.sans-serif'] = 'Helvetica'
            mpl.rcParams['mathtext.fontset'] = 'cm'
            # ensure that font type is 42 (TrueType) for enabling vector graphics when pdf is saved
            mpl.rcParams['pdf.fonttype'] = 42

    def _parse_kwargs(self, kwargs_param, base_values = 'pm_default'):
        """
            Parses the dictionary kwargs_param supplied to add_data or draw_plots that contains keyword arguments to make the plot.
            Returns a dictionary of data parameters with appropriately assigned default or supplied values for each keyword argument.

            'base_values' refers to the values that need to be used for arguments (that are a part of 'default_params') that have not been supplied.
            If it is set to 'pm_default', the base values used are those in default_params attribute.
            Else, if a data key is given, the values in the parameter dictionary corresponding to that data_key is used as the base.
            Only the arguments that have been supplied through kwargs_param are modified.

            Some keyword arguments are specific to a given plot type, so they is not a part of the 'default_params' attribute.
            These are parsed and stored in 'plot_params' dictionary that is directly supplied to the respective plotting functions.
        """
        # start with the either the default values of parameters or the values for the given data_key, and modify only the parameters that have been supplied as arguments
        # note that deep copy is not required below because we will be completely replacing the value for any given key
        base_values = str(base_values)
        if base_values.lower() == 'pm_default':
            data_params = self.default_params.copy()
        elif base_values in self.data_key_list:
            data_params = self.data_params_dict[base_values].copy()
        else:
            raise ValueError("Incorrect value supplied to 'base_values'. It accepts either a data key or 'pm_default' as input.")

        # infer the plot type from available plot types
        try:
            # plot type can be specified in a number of ways listed in valid_plot_types attribute
            # so we infer the corresponding key in valid_plot_types for keeping a uniform and consistent reference to the plot type
            plot_type = str(kwargs_param['plot_type']).lower()
            plot_type_match_found = False
            for (plot_type_key, plot_type_list) in self.valid_plot_types.items():
                if plot_type in plot_type_list:
                    plot_type = plot_type_key
                    plot_type_match_found = True
                    break
            if not plot_type_match_found:
                raise ValueError("Please specify a valid plot type. Currently supported options are %s" %(' '.join(self.valid_plot_types.keys()),))

            # check if plot_type has been changed
            if plot_type != data_params['plot_type']:
                plot_type_changed = True
        except KeyError:
            # plot_type has not been supplied as a keyword argument, so use the default value for plot_type
            plot_type = data_params['plot_type']

            # plot_type has not been changed
            plot_type_changed = False

        # make sure that plot_params has not directly been supplied; every parameter is parsed and plot_params is populated automatically
        if 'plot_params' in kwargs_param.keys():
            raise ValueError("Please specify all the parameters directly as arguments. 'plot_params' dictionary will be created automatically for the specified plot type.")

        # a separate dictionary for plot_params so as to not mutate it's values in data_params, to be added back after necessary modifications; deep copy could have been
        # used while copying data_params, but we don't do so because deep copy can be costly and only plot_params dictionary needs to be handled separately
        # we delete the plot_params key from data_params so that it does not interfere with parsing data_params later
        plot_params = data_params.pop('plot_params').copy()

        # if plot_type has been changed, we need to parse the parameters in plot_params and keep only the parameters that are valid for the new plot_type
        # the remaining parameters are added to data_params if they are valid parameters for data_params, else the parameter is discarded and a warning is issued
        if plot_type_changed:
            # list of parameters that need to be removed
            # param cannot be deleted from the plot_params dictionary during the iteration, so we collect it in a list and delete them after the iteration is complete
            remove_params_list = list()
            # if data_params is default_params, then plot_params is an empty dictionary and the following iteration will not occur
            for (param, param_val) in plot_params.items():
                # if param is a valid argument for the new plot_type, keep it in plot_params dictionary (which is passed to the plotting function specific to the plot type)
                if param in self.plot_type_kwargs[plot_type]:
                    pass
                # if param is not a valid parameter for the new plot_type, add it to data_params if it is a valid data_params parameter
                # note that there is no chance of param being a part of both plot_params and data_params at this point because data_params
                # is a dictionary corresponding to a data key (without any modification), which is already parsed
                # so we simply check if it is a default_params parameter, and add it to data_params if it is
                elif param in self.default_params.keys():
                    data_params[param] = param_val
                    # param has been added to data_params, so remove it from plot_params
                    remove_params_list.append(param)
                else:
                    remove_params_list.append(param)
                    warnings.warn("'%s' is no longer a valid parameter after changing the plot type to '%s', so it has been discarded." %(param, plot_type))

            # remove the specified parameters from plot_params
            for param in remove_params_list:
                del plot_params[param]

        # get all the parameters used for customizing the plot
        # precedence is given to parameters that are a part of plot_params, i.e., if param is a valid plot_param key, it is assigned there, and if not, it is assigned to data_params
        for (param, param_val) in kwargs_param.items():
            # add param to plot_params dictionary which is passed to the plotting function specific to the plot type, if it is a valid argument
            if param in self.plot_type_kwargs[plot_type]:
                plot_params[param] = param_val
            # add param to data_params dictionary directly
            elif param in data_params.keys():
                data_params[param] = param_val
            else:
                raise ValueError("'%s' is not a valid parameter. Refer to the docstring for a list of accepted parameters for general plotting as well as for '%s' plot type." %(param, plot_type))

        # move relevant parameters from data_params to plot_params (which may not have been supplied through kwargs_param), as plot_params is what is passed to the plotting function
        # note that the above parsing of plot_type_kwargs is also necessary because we need to identify which parameters supplied through kwargs_param are invalid
        for (param, param_val) in list(data_params.items()):
            if param in self.plot_type_kwargs[plot_type]:
                # add the relevant parameter to plot_params and remove it as from data_params (it will later be added to data_params as a part of plot_params dictionary)
                # only modify the parameters have not already been added to plot_params (to prevent overwriting values supplied through kwargs_param)
                if param not in plot_params.keys():
                    plot_params[param] = param_val
                del data_params[param]

        # make sure that the plot_type inferred above is set in data_params dictionary
        data_params['plot_type'] = plot_type
        # save the inferred plot_params to data_params
        data_params['plot_params'] = plot_params

        return data_params

    def add_data(self, ydata, xdata = [], key = '', desc = '', **kwargs_param):
        """
            Caches the data which is then used for plotting when required

            'ydata' is the data that is used to make different types of plots, and is a necessary argument.
            'xdata' is optional data that is used to plot on the x-axis for some commonly used types of plots.
            'key' specifies an identifier that can be used to refer to this data later. If nothing is specified, a unique number is assigned.
            'desc' is a string containing a description for the data. Optional, but recommended if using the method in an interpreter to identify the data.

            Only the above options are specified explicitly as function arguments. The parameters to custommize the plot are not mentioned explicitly, but are captured through
            kwargs_param. These parameters can be supplied to the add_data method as usual function arguments. The allowed parameters are:
                - figsize (specifies the size of the figure)
                - 'plot_type' is used to specify what type of plot is intended. Currently supported options are:
                    -- 'plot' or 'line' or 'line plot'   : "usual" line plot
                    -- 'discrete'                        : line plot, but with discrete data points (equivalent to plot_type = 'line plot' and 'marker' = 'o')
                    -- 'hist' or 'histogram'             : histogram
                    -- 'errbar' or 'error bar'           : error bar
                - subplots (either a tuple of arguments or a dictionary of keyword arguments that specifies how to create a subplot)
                - fmt (format string for quick plotting of the format '[marker][line][color]'; for example, 'o-r' is equivalent to marker = 'o', linestyle = '-', and color = 'r'; recommended only when using the interpreter)
                - linestyle (specifies the style used for the line, where commonly used options are '-', '--', '-.', ':', '')
                - marker (specifies the style used for marker, where commonly used options are 'o', '.', '^', 's', 'X', 'D', '*')
                - title (title for the plot)
                - xlabel, ylabel (two separate parameters, used to specify the labels on the x and y axis respectively)
                - xlim, ylim (two separate parameters, used to specify x and y axis limits respectively)
                - hold (specifies whether to hold the plot for additional plotting that is to follow)
                - plot_params (a dictionary that is supplied directly to the plotting function of the axes object for fine-grained control; do not modify this dictionary directly;
                               the supplied arguments are parsed and automatically assigned to plot_params dictionary; see 'plot_type_kwargs' attribute for a list of accepeted parameters for specified plot type)
                - label (specifies a label for the plot to be used in a legend)
                - legend (specifies whether to plot a legend, and if yes, what formatting to use)
                    -- special options: 'pretty' (sets the legend at centre-top, and makes it a fancy box; recommened to not use with a title)
                - color (specifies the color of the lines for that particular plot)
                - xticks (a list of tick locations to be drawn on the x-axis; ticks are removed if tick locations is an empty list; if None, default matplotlib values used)
                - yticks (a list of tick locations to be drawn on the y-axis; ticks are removed if tick locations is an empty list; if None, default matplotlib values used)
                - xticklabels (a list of tick labels or a dictionary of 'labels' and font properties to be drawn on the x-axis; if None, default matplotlib values used;
                               Warning: if supplied without xticks, unexpected behaviour may be seen)
                - yticklabels (a list of tick labels or a dictionary of 'labels' and font properties to be drawn on the y-axis; if None, default matplotlib values used;
                               Warning: if supplied without yticks, unexpected behaviour may be seen)
                - xtick_params (a dictionary that controls the behaviour of the tick labels for x-axis; size can be specified through labelsize, and color through labelcolor)
                - ytick_params (a dictionary that controls the behaviour of the tick labels for y-axis; size can be specified through labelsize, and color through labelcolor)
                - xlabel_size (controls the size of the label for x-axis)
                - ylabel_size (controls the size of the label for y-axis)
                - title_size (controls the size of the title)
                - linewidth (controls the width of the line)
                - markersize (controls the size of the markers when discrete ticks ('o', '-o', etc.) are included in linestyle)
                - grid (specifies whether to include a grid in the plot)
                - yscale (specifies which y-axis scale to apply ("linear", "log", "symlog", etc.))
                - twinx (specifies if the plot should have two y-axes sharing the x-axis)
                - inset_axes is a dictionary used to create an inset plot. The dictionary accepts the keys:
                    -- any parameter supplied to matplotlib.axes.Axes.inset_axes
                    -- 'new' (optional parameter; if True, a new inset_axes is created instead of using one that may have been previously created; default is False)
                    -- 'indicate_inset_zoom' (optional parameter; if True, lines connecting the inset to the location in the parent axes are drawn)
                - style (specifies what rc style or list of styles to use for the plots (can choose from many defaults or provide path to custom style file;
                         list of mpl provided style files can be obtained from plt.style.available; styles can be combined, with rightmost taking precedence))
                - scientific_notn (a dictionary of options or False to turn off; the key 'scilimits' with value (m, n) specifies the range (10^m, 10^n) outside which 10^x notation is used; set to (0, 0) to use it always;
                                   the key 'offsetsize' specifies the size of font for 10^x)
                - show_data_key (specifies whether to show the key for the data in the title of the plot)

                The plotted figure can be saved using the following options.
                - save_fig (specifies whether to save the figure; if false, the plot is displayed on GUI)
                - dirpath (path to the directory where the figure is saved)
                - filename (name of the figure to be used while saving)
                - fig_format (format of the figure used while saving)
                - bbox_inches (specifies the region of figure that needs to be saved)
            Default values of these parameters can be found in the attribute default_params.
            Note that linestyle, marker take precedence over fmt, while fmt takes precedence over color.

            Note: Many keyword arguments are specific to a given plot type. These keyword arguments can be supplied as normal arguments to the function, and these are parsed and
                  added to plot_params dictionary, which is directly supplied to the plotting function appropriate for the given plot type.
                  For a list of arguments accepted by a specific plot type, check the attribute plot_type_kwargs, which is a dictionary indexed by plot_type.

            As noted above, an option is provided to specify the parameters for the intended plot_type using the 'plot_params' dictionary directly.
            This enables fine-grained control of the plot to the extent allowed by matplotlib.

            The set of options provided above is expected to be enough for common plotting tasks.
            However, if even more customization is desired, the figure and axes objects are available through 'fig_dict' and 'ax_dict' attributes of the class,
            which are indexed by the data key. It is possible, though, that these objects are modified as more data is plotted.
        """
        if not len(xdata):                        # if xdata is not given, then just plot the indices on the x-axis
            xdata = range(len(ydata))

        if not len(ydata):
            raise ValueError("ydata must be supplied.")

        # the data identifying key and description
        data_key  = str(key)
        data_desc = str(desc)

        # ensure that the data_key provided is unique
        if data_key in self.data_key_list and len(data_key):
            raise ValueError("Each key must be unique. %s has already been used." %data_key)

        # if data_key is provided, include it in the list; if not, assign the index of the data as the data_key
        if len(data_key):
            self.data_key_list.append(data_key)
        else:
            data_key = str(len(self.data_key_list))       # point to the length of the data, if it is unique
            if data_key in self.data_key_list:            # if not, assign the largest number that is not a data id
                for data_key_elt in self.data_key_list:
                    # if data_key_elt is an integer, make sure data_key is one more than that
                    try:
                        if int(data_key) <= int(data_key_elt):
                            data_key = str(int(data_key_elt) + 1)
                    except ValueError:
                        pass
            self.data_key_list.append(data_key)

        # store the description of the data
        self.data_desc_dict[data_key] = data_desc

        # add the data to the data list
        self.xdata_dict[data_key] = xdata
        self.ydata_dict[data_key] = ydata

        # parse the supplied keyword arguments through kwargs_param, and obtain the dictionary of parameters for the given data
        data_params = self._parse_kwargs(kwargs_param)

        # add the parameters to the data_params_dict corresponding to the given data
        self.data_params_dict[data_key] = data_params

        self.num_data += 1

    def modify_data(self, key_list = "all", xdata = None, ydata = None, **kwargs_param):
        """
            Modifies the data corresponding to the specified list of keys with the given parameters.

            The x & y data can also be updated if necessary.
            The parameters to be modified can be supplied as usual function arguments with the new value that needs to be assigned.
        """
        if key_list == "all":
            data_key_list = self.data_key_list
        elif type(key_list) in [str, int]:
            data_key_list = [key_list]
        else:
            data_key_list = key_list

        for data_key in data_key_list:
            data_key = str(data_key)

            # ensure data_key is valid
            if not data_key in self.data_params_dict:
                raise ValueError("Invalid data identifier")

            # modify the data
            if not xdata in [None, False]:
                self.xdata_dict[data_key] = xdata
            if not ydata in [None, False]:
                self.ydata_dict[data_key] = ydata

            # obtain an updated dictionary of data parameters from the given keyword arguments
            data_params = self._parse_kwargs(kwargs_param, base_values = data_key)
            
            # modify the data parameters
            self.data_params_dict[data_key] = data_params

    def remove_data(self, key_list = "all"):
        """
            Empty the data cache for the specified data keys, and delete the key itself.
        """
        if key_list == "all":
            # clear everything
            self.xdata_dict   = dict()
            self.ydata_dict   = dict()
            self.data_key_list = list()

            self.num_data = 0
        else:
            if type(key_list) in [str, int]:
                data_key_list = [key_list]
            else:
                data_key_list = key_list

            # clear the specified data indices
            for data_key in data_key_list:
                data_key = str(data_key)
                # ensure that given data_key is valid
                if not data_key in self.data_key_list:
                    raise ValueError("%s is not a valid key" %data_key)

                del self.xdata_dict[data_key]
                del self.ydata_dict[data_key]
                del self.data_params_dict[data_key]
                del self.data_desc_dict[data_key]
                self.data_key_list.remove(data_key)

                self.num_data -= 1

    def release_hold(self):
        """
            Removes the 'fig' so that a new figure object is created.
        """
        self.fig = None

    def draw_plots(self, key_list = "all", override_data_params = False, **kwargs_param):
        """
            Plots the data corresponding to the specified list of keys, formatted with the associated parameters.
            These plots use the parameters specified in add_data method as noted above.
            Any additional parameter supplied to draw plots is applied across all the data that is being plotted.
            These parameters must be compataible with each plot type being plotted.

            However, it is possible to force draw_plots function to use parameters supplied to it (instead of those with the data),
            by setting override_data_params to True.
            The intended use is to enforce a uniform plotting style across all the plots when deemed necessary.

            The list of arguments that will be used by this function can be found in the docstring of add_data method.

            Evaluation logic:
            Parse each data key supplied. If use_global_params is True, then plot with the parameters supplied with draw_plots.
            Else, use the parameters provided with the data for plotting.

            Design choice:
                - Using **kwargs_param in the function definition makes it less clear what the function does, and is not ideal in that sense. One would have to refer to the docstring of 'add_data'
                  to see what variables this function expects.
                - We go with such an approach to make the class scalable, the reason for which is described below. We briefly describe the implementation to motivate the reason.
                - First, the variables passed through **kwargs_param are made function attributes through __dict__.update(**kwargs_param), and we abbreviate self.draw_plots as _dp for ease of use.
                  Then, if we want to access a variable 'foo' passed through **kwargs_param, we simply use _dp.foo inside the function.
                  [We follow Python convention that variables starting with '_' are intended for internal use. We add '_' in the reference to the method itself, instead of including it for each attribute.]
                - Now suppose we want to add a new feature that relies on a variable 'bar'. We can add 'bar' (with its default value) to self.default_params dictionary, and then update the
                  docstring of add_data. Then, we can use that variable in draw_plots function by simply using '_dp.bar'. This avoids the hassle of adding the variable in multiple function definitions
                  (and possibly, some places within the function), and thus make scaling up less error prone.
        """
        # plot only when all data keys have been "parsed"
        # when draw_plots is called for the first time, key_list needs to be parsed and given appropriate parameters
        # in that case, execute the following code (a recursion)
        if not self.ready_to_plot:
            if key_list == "all":
                data_key_list = self.data_key_list
            elif type(key_list) in [str, int]:
                data_key_list = [key_list]
            else:
                data_key_list = key_list

            for data_key in data_key_list:
                data_key = str(data_key)
                # ensure data_key is valid
                if not data_key in self.data_key_list:
                    raise ValueError("%s is not a valid data key" %(data_key,))

                if override_data_params:
                    # use the arguments supplied by draw_plots function
                    # gives uniformity to all plots, without having to individually specify these arguments for each plot
                    data_params = self._parse_kwargs(kwargs_param)
                else:
                    # use the arguments supplied by each data_key
                    # in case of any overlap in data_params and kwargs_param, the value supplied in kwargs_param takes precedence
                    data_params = self._parse_kwargs(kwargs_param, base_values = data_key)

                # parse the arguments and plot the data
                self.ready_to_plot = True

                self.draw_plots(key_list = data_key, override_data_params = override_data_params, **data_params)

            # clear the canvas after all plotting is done, preparing for any new data that may be added and plotted
            self.fig = None
            return

        # a reference to the draw_plots function
        _dp = self.draw_plots

        # populate the attributes of draw_plots function (which is a method) with variables supplied through **kwargs_param
        # note that method attributes are immutable (and they are probably destroyed after the function executes)
        # so the variables _dp.foo are read-only
        _dp.__dict__.update(**kwargs_param)

        # get ready for next parsing
        # we include this in the beginning, for if the plotting throws an error, there will be no chance to reset this attribute
        self.ready_to_plot = False

        # key_list is not a list anymore at this point
        data_key = str(key_list)

        # get the x and y data that need to be plotted
        xdata = self.xdata_dict[data_key]
        ydata = self.ydata_dict[data_key]

        # use a style template for the plot
        if _dp.style:
            plt.style.use(_dp.style)

        # specify which colors to cycle through when plotting many lines in the same figure
        if _dp.color_cycle:
            # use the color cycle provided
            if type(_dp.color_cycle) in [list, tuple]:
                color_cycle = _dp.color_cycle
            elif str(_dp.color_cycle).lower() in self.color_cycle_dict.keys():
                color_cycle = self.color_cycle_dict[str(_dp.color_cycle).lower()]
            else:
                raise ValueError("Please provide a list of hexadecimal color strings for specifying the color cycle " +\
                                 "or choose one of 'default', 'mpl_default', 'seaborn', 'seaborn-dark-palette', 'ggplot'")
            mpl.rcParams['axes.prop_cycle'] = cycler('color', color_cycle)

        # if create a new figure if none exists, or use an existing figure to plot on top of it
        if not self.fig:
            # create a figure and add keyboard shortcuts to handle the figure
            fig = plt.figure(figsize = _dp.figsize)
            self.fig = fig
            fig.canvas.mpl_connect('key_press_event', handle_keypress)
            # get some spacing between the subplots and the edges
            fig.subplots_adjust(left = 0.15, bottom = 0.25)

            # create subplots
            if type(_dp.subplots) == int:
                ax = fig.add_subplot(_dp.subplots)
            elif type(_dp.subplots) in [list, tuple]:
                ax = fig.add_subplot(*_dp.subplots)
            elif type(_dp.subplots) == dict:
                ax = fig.add_subplot(**_dp.subplots)
        else:
            fig = self.fig
            ax = self.ax

        # "duplicate" the axes object if two y-axes required in the same plot
        if _dp.twinx:
            if self.axt:
                ax = self.axt
            else:
                ax = ax.twinx()

        # create an inset axes if required
        # the inset axes is created from the "parent" axes object, and plots are drawn on the inset axes
        if _dp.inset_axes is not None:
            # determines if a new inset should be created or if the previous inset (if it exists) should be used for plotting
            try:
                new_inset = bool(_dp.inset_axes['new'])
                del _dp.inset_axes['new']
            except KeyError:
                new_inset = False

            if (self.axin is not None) and (not new_inset):
                ax = self.axin
            else:
                # input a dictionary with all keys in _dp.inset_axes except indicate_inset_zoom
                # the returned inset axes is named ax, overriding the parent axes object
                # the parent axes object can be referred to later using self.ax
                ax = ax.inset_axes(**{key: value for (key, value) in _dp.inset_axes.items() if not key == 'indicate_inset_zoom'})

        # save the figure and axes objects
        self.fig_dict[data_key] = fig
        self.ax_dict[data_key] = ax

        # store the figure and axes object for reuse if hold is True
        if _dp.hold:
            self.fig = fig
            if _dp.twinx or _dp.inset_axes is not None:
                # inset_axes is given precedence over twinx because the twinx axes is "converted to" inset_axes if latter is not None
                # this is to account for the fact that we will have to draw an inset for the twinx axes if twinx is True and inset_axes is not None
                if _dp.inset_axes is not None:
                    self.axin = ax
                else:
                    self.axt = ax
            else:
                self.ax  = ax
        else:
            self.fig = None
            self.ax = None
            self.axt = None
            self.axin = None

        # add the key to the title if required
        if _dp.show_data_key:
            title_key = r' $\mathtt{[key: %s]}$' %(str(data_key),)
        else:
            title_key = ''

        # set the title, xlabel and ylabel to the plot, with specified font sizes
        ax.set_title(str(_dp.title) + title_key, fontsize = _dp.title_size)
        ax.set_xlabel(str(_dp.xlabel), fontsize = _dp.xlabel_size)
        ax.set_ylabel(str(_dp.ylabel), fontsize = _dp.ylabel_size)

        # set limit for x and y axes
        if _dp.xlim is not None:
            ax.set_xlim(_dp.xlim[0], _dp.xlim[1])
        if _dp.ylim is not None:
            ax.set_ylim(_dp.ylim[0], _dp.ylim[1])

        # set x and y ticks
        if _dp.xticks is not None:
            ax.set_xticks(_dp.xticks)
        if _dp.yticks is not None:
            ax.set_yticks(_dp.yticks)

        # set x and y tick labels (and font properties)
        if _dp.xticklabels is not None:
            if type(_dp.xticklabels) in [list, tuple]:
                ax.set_xticklabels(labels = _dp.xticklabels)
            elif type(_dp.xticklabels) == dict:
                ax.set_xticklabels(**_dp.xticklabels)
        if _dp.yticklabels is not None:
            if type(_dp.yticklabels) in [list, tuple]:
                ax.set_yticklabels(labels = _dp.yticklabels)
            elif type(_dp.yticklabels) == dict:
                ax.set_yticklabels(**_dp.yticklabels)

        # format x and y ticks
        if _dp.xtick_params:
            ax.tick_params(axis = 'x', **_dp.xtick_params)
        if _dp.ytick_params:
            ax.tick_params(axis = 'y', **_dp.ytick_params)

        # add a grid if required
        if _dp.grid:
            ax.grid(True)

        # convert the tick labels on x and y axes to use 10^x notation for numbers
        if _dp.scientific_notn:
            try:
                # use scientific notation; MathText enforces a x 10^b instead of ae^b
                ax.ticklabel_format(style = 'sci', scilimits = _dp.scientific_notn['scilimits'], axis = 'both', useMathText = True)
            except KeyError:
                raise ValueError("scientific_notn must be a dictionary with scilimits as a key")

            try:
                # set the font size for 10^x, if required
                ax.xaxis.get_offset_text().set_fontsize(_dp.scientific_notn['offsetsize'])
                ax.yaxis.get_offset_text().set_fontsize(_dp.scientific_notn['offsetsize'])
            except KeyError:
                pass

        # set the desired yscale ('linear', 'log', and so forth)
        if _dp.yscale:
            ax.set_yscale(str(_dp.yscale))

        # color may have moved from data_params to plot_params dict within data_params depending on plot type, so handle appropriately
        try:
            color = _dp.color
        except AttributeError:
            color = _dp.plot_params['color']
        if color:
            # in case of a second y-axis, color the ticks and the label as well
            if _dp.twinx:
                ax.tick_params(axis = 'y', labelcolor = color)
                ax.yaxis.label.set_color(color)

        # make different types of plots, as specified
        if _dp.plot_type == 'lineplot':
            # make a usual line plot; additional parameters can be speficied through plot_params dictionary
            ax.plot(xdata, ydata, _dp.fmt, **_dp.plot_params)
        elif _dp.plot_type == 'discrete':
            # remove a line between different data points, and include a marker instead
            # if a marker is supplied, use that; else, use 'o'
            plot_params = _dp.plot_params.copy()
            plot_params['linestyle'] = ''
            # marker may be part of _dp attribute or plot_params depending on plot_type
            try:
                plot_params['marker'] = _dp.marker or 'o'
            except AttributeError:
                plot_params['marker'] = plot_params['marker'] or 'o'
            ax.plot(xdata, ydata, _dp.fmt, **plot_params)
        elif _dp.plot_type == 'hist':
            # ydata is made into a histogram; options for the histogram can be specified through plot_params dictionary
            ax.hist(ydata, **_dp.plot_params)
        elif _dp.plot_type == 'errorbar':
            # xerr, yerr (arrays containing the errors), along with other options can be specified through plot_params dictionary
            ax.errorbar(xdata, ydata, **_dp.plot_params)
        else:
            raise ValueError("Please specify a valid plot type. Currently supported options are %s" %(' '.join(self.valid_plot_types.keys()),))

        # create lines connecting the inset with the original location of the image
        if _dp.inset_axes is not None:
            try:
                indicate_inset_zoom = bool(_dp.inset_axes['indicate_inset_zoom'])
            except KeyError:
                indicate_inset_zoom = False

            if indicate_inset_zoom:
                if _dp.twinx:
                    (self.axt).indicate_inset_zoom(ax)
                else:
                    (self.ax).indicate_inset_zoom(ax)

        # set the legend for the plot, and set zorder for legend if provided (largest zorder is on top); zorder works only within this axis
        if _dp.legend:
            if type(_dp.legend) == dict:
                if 'zorder' in _dp.legend.keys():
                    zorder = _dp.legend['zorder']
                    del _dp.legend['zorder']
                    axlegend = ax.legend(**_dp.legend)
                    axlegend.set_zorder(zorder)
                else:
                    ax.legend(**_dp.legend)
            elif str(_dp.legend).lower() == 'pretty':
                if len(_dp.title) == 0:
                    # the following places the lower center point of the legend at the location (x = 0.5, y = 0.95); used when bbox_to_anchor is a 2-tuple
                    # here, (x, y) is defined with respect the the axis dimensions (axis width, axis length)
                    ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, 0.95), ncol = 4, fontsize = 30, fancybox = True)
                else:
                    # the following chooses the best location for the legend in the upper half region of the figure
                    # when bbox_to_anchor is a 4-tuple (x, y, width, height) that specifies a region (x, y is lower left corner of the region) and 'loc' acts within that region (see matplotlib legend documentation)
                    ax.legend(loc = 'best', bbox_to_anchor = (0, 0.5, 1, 0.5), fontsize = 30, ncol = 2, fancybox = True)
            else:
                ax.legend()

        # save the figure if required, and if not, show the plot on a GUI
        if _dp.save_fig:
            if not _dp.filename:
                raise ValueError("Provide a valid filename")

            # get a proper path for the user's operating system
            # pathlib accepts forward slashes and internally changes it to whatever is appropriate for the OS
            image_filename = PurePath('%s/%s' %(_dp.dirname, _dp.filename))
            # add the specified extension
            try:
                image_filename = image_filename.with_suffix(_dp.fig_format)
            except ValueError:
                # if '.' not supplied with fig_format, it is not a valid suffix
                image_filename = image_filename.with_suffix('.' + _dp.fig_format)

            # matplotlib's savefig accepts PurePath (or PathLike) objects for fname ("filename") argument
            fig.savefig(image_filename, format = _dp.fig_format, bbox_inches = _dp.bbox_inches)

        if _dp.show_fig and not _dp.save_fig:
            # if the figure is not being saved, we show the figure
            # we provide the option to toggle showing the figure because Jupyter notebooks
            # throw a warning when fig.show is used
            fig.show()
