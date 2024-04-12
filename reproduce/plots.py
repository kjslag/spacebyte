import collections
import numbers
import numpy as np
import torch
import matplotlib.pyplot as plt

import mpl_toolkits.axes_grid1

def tensor_items(xs, dtype=None):
    if isinstance(xs, list):
        return [tensor_items(x) for x in xs]
    elif isinstance(xs, tuple):
        return tuple(tensor_items(x) for x in xs)
    elif isinstance(xs, dict):
        return {k: tensor_items(v) for k,v in xs.items()}
    elif isinstance(xs, torch.Tensor):
        if dtype is None:
            dtype = xs.dtype
            if dtype == torch.bfloat16:
                dtype = torch.float32
        return xs.item() if xs.dim()==0 else xs.detach().cpu().to(dtype=dtype).numpy()
    elif isinstance(xs, np.ndarray):
        return xs if dtype is None else xs.astype(dtype, copy=True)
    elif hasattr(xs, '__next__') and not hasattr(xs, '__getitem__'):
        return (tensor_items(x) for x in xs)
    else:
        return xs

def gridplot_data(dataset, x_func, y_func, label_funcs, *args, grid_kwargs={}, **kwargs):
    if not isinstance(label_funcs[0], list):
        label_funcs = [label_funcs]
    
    subplots = [ [subplot_data(dataset, x_func, y_func, f, *args, **kwargs) for f in fs] for fs in label_funcs ]
    grid(*subplots, **grid_kwargs)

# xys = [(optional xs, ys, optional 'legend label', options), ...]
# options is a dictionary with the following options:
# 'color' = matlibplot color
# others are passed to matlibplot
def plot(xys, scale='linear', *,
         labels=(None,None,None), dataRange=None,
         subplot=plt, figsize=(9,5),
         plotrange=None, # ([x_min, x_max], [y_min, y_max])
         new_figure=True,
         legend=True, # or a dict of legend options
         args=['o'],
         **kwargs):

    if len(xys) == 0:
        print(f'plots.plot: nothing to plot: {xys}')
        return
    
    if not isinstance(xys, list) or isinstance(xys[0], numbers.Number):
        xys = [xys]
    
    xys = [ (dataRange,  xy) if not isinstance(xy, tuple) else \
            (dataRange, *xy) if not (hasattr(xy[1], '__getitem__') and
                                     isinstance(xy[1][0], (int, float, torch.Tensor))) else \
            xy for xy in xys]

    use_legend = False
    if scale == 'log':
        scale = ('linear', 'log')
    if isinstance(scale, str):
        scale = (scale, scale)
    
    is_plt = subplot is plt
    if new_figure and is_plt:
        plt.figure(figsize=figsize)
    for x, y, *other in xys:
        label = None
        if len(other) > 0 and not isinstance(other[0], dict):
            label = other[0]
            del other[0]
            use_legend = legend is not False

        options = {}
        if len(other) > 0:
            options = other[0]
            del other[0]

        assert len(other) == 0, other

        if x is None:
        #     assert not callable(y)
            x = np.arange(1, len(y)+1)
        # elif isinstance(x, tuple): # x = (min, max, optional: # points)
        #     if len(x) == 2:
        #         x = (*x, 1000 if callable(y) else len(y))
        #     x = np.linspace(*x) if not (callable(y) and scale[0] == 'log') else \
        #         np.exp(np.linspace(np.log(x[0]), np.log(x[1]), x[2]))
         
        # if callable(y):
        #     try:
        #         y = y(x)
        #     except:
        #         y = [y(xi) for xi in x]
        #     joined0 = True
        
        if isinstance(x[0], str):
            plotter = subplot.bar
            subplot.xticks(rotation=90)
        else:
            plotter = subplot.plot

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu()

        if 'args' in options:
            args0 = options['args']
            del options['args']
        else:
            args0 = args

        assert len(x) == len(y)
        try:
            plotter(x, y, *args0, label=label, **(kwargs|options))
        except:
            print(f'{x=}')
            print(f'{y=}')
            print(f'{label=}')
            raise

    if scale[0] != 'linear':
        (plt.xscale if is_plt else subplot.set_xscale)(scale[0])
    (plt.yscale if is_plt else subplot.set_yscale)(scale[1])
    
    if use_legend:
        subplot.legend(**legend) if isinstance(legend, dict) else subplot.legend()

    assert isinstance(labels, (tuple, list))
    for f,l in zip((plt.xlabel if is_plt else subplot.set_xlabel,
                    plt.ylabel if is_plt else subplot.set_ylabel,
                    plt.title  if is_plt else subplot.set_title), labels):
        if l is not None:
            f(l)

    if plotrange is not None:
        if plotrange[0] is not None:
            plt.xlim(*plotrange[0]) if is_plt else subplot.set_xlim(*plotrange[0])
        if plotrange[1] is not None:
            plt.ylim(*plotrange[1]) if is_plt else subplot.set_ylim(*plotrange[1])

    if new_figure and is_plt:
        plt.show()

    return subplot

def plot_data(dataset, x_func, y_func, label_func, *args,
              legend_labels=[], # in order to specify ordering
              post_process=lambda data: data,
              options_func=lambda label: {}, subplot=plot, **kwargs):
    xyls_dict = collections.defaultdict(list)
    def sortable_label(label):
        try:
            return (legend_labels.index(label), str(l))
        except ValueError:
            (len(legend_labels), str(l))
    for d in dataset:
        l = label_func(d)
        if l is not None:
            xyls_dict[(sortable_label(l), l)].append((x_func(d), y_func(d)))
    for k in xyls_dict:
        xyls_dict[k].sort()
    plot_data = [([x for x,_ in xys], [y for _,y in xys], str(l), options_func(l))
                 for (_,l), xys in sorted(xyls_dict.items())]
    return subplot(post_process(plot_data), *args, **kwargs)

def subplot_data(*args, **kwargs):
    return plot_data(*args, subplot=SubPlot, **kwargs)

# example:
# plots.grid([plots.SubPlot(torch.arange(5)), plots.SubPlot(torch.arange(4))], figsize=6)
def grid(*subplots, figsize=16):
    subplots = np.matrix(subplots) # NOTE: use grid([...], [...]) instead of grid([[...], [...]])
    n_rows, n_cols = subplots.shape
    if isinstance(figsize, int):
        figsize = ( figsize, figsize * n_rows/n_cols * 0.8 * n_cols/(1+n_cols) )
    fig, axs = plt.subplots(*subplots.shape, figsize=figsize)
#     fig, axs = plt.subplots(*subplots.shape, figsize=(figsize, figsize * (1+n_rows)/(1+n_cols)))
    axs = np.matrix(axs).reshape(*subplots.shape)

    for i in range(n_rows):
        for j in range(n_cols):
            sub = subplots[i,j]
            sub.plot(*sub.args, **sub.kwargs, subplot=axs[i,j])
    
    plt.tight_layout(pad=1*n_rows/n_cols)
#     plt.tight_layout(pad=10*n_rows/n_cols)
    plt.show()

class Sub:
    def __init__(self, plot, *args, **kwargs):
        self.plot   = plot
        self.args   = args
        self.kwargs = kwargs

class SubPlot(Sub):
    def __init__(self, *args, **kwargs):
        super().__init__(plot, *args, **kwargs)

class SubImage(Sub):
    def __init__(self, *args, **kwargs):
        super().__init__(image, *args, **kwargs)

def images(images, grid_figsize=16, **kwargs):
    grid(*[SubImage(image, **kwargs) for image in images], figsize=grid_figsize)

def image(image, title=None, subplot=plt, figsize=(9,5), legend=None, **kwargs):
    C, H, W = image.shape

    is_plt = subplot is plt
    if is_plt:
        plt.figure(figsize=figsize)

    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()

    if C == 1:
        image = image[0]
        if 'cmap' not in kwargs:
            kwargs = dict(cmap='gray', vmin=0, vmax=1) | kwargs
        elif legend is None:
            legend = True
    elif C == 3:
        image = image.transpose(1, 2, 0)
    else:
        assert False

    im = subplot.imshow(image, **kwargs)
    if legend:
        if is_plt:
            plt.colorbar()
        else:
            cax = mpl_toolkits.axes_grid1.make_axes_locatable(subplot).append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
    if title is not None:
        subplot.set_title(title)
    subplot.axis('off')
    if is_plt:
        plt.show()

def hist(x, *args, show=True, **kwargs):
    kwargs = kwargs | dict(bins='auto', density=True, histtype='step')
    plt.hist(tensor_items(x), *args, **kwargs)
    if show:
        plt.show()
