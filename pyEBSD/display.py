import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyEBSD.misc import range_map

_displays = []

# TODO numpy-style docstring
# `nancolor` is a triple (R, G, B), with each element on the interval [0, 1].
# Expects images to have 2 spatial dimensions.
# TODO input validation?
def _display(im: np.ndarray, ax=None, newfigtitle: str="", newaxtitle: str="",
             nancolor: tuple=(0, 0, 0)):
    """Display an image on PyPlot axes `ax`.
    Creates a new figure if `ax` is None.
    Only displays the first 3 channels.
    Image must have exactly 3 channels.
    To show, call ``pyplot.show()``.
    """
    im = im.copy()
    im[np.isnan(im).any(axis=2)] = nancolor
    if ax is None:
        fig = plt.figure()
        fig.suptitle(newfigtitle)
        ax = fig.add_subplot()
        ax.set_title(newaxtitle)
    ax.imshow(im)

# Expects all quaternion elements to be on the interval [-1, 1].
# Displays first 3 channels.
# TODO input validation?
def display_quats(im: np.ndarray, ax=None, newfigtitle: str="",
                  newaxtitle: str="", nancolor: tuple=(0, 0, 0)):
    # TODO docstring
    _display(
        range_map(im[..., :3], (-1, 1), (0, 1)),
        ax=ax, newfigtitle=newfigtitle, newaxtitle=newaxtitle, nancolor=nancolor
    )

# TODO proper docstrings (incl. for public methods)
# intended to (usually) act as a callable
class _volumetric_display:
    """``disp = volumetric_display(...)``
    Create an interactive slice-based display of a multichannel volumetric
    image.
    To show, call ``pyplot.show()``.
    Only displays the first 3 channels.
    Image must have at least 3 channels.
    """
    def __init__(self, im: np.ndarray, axis: int, title: str="",
                 nancolor: tuple=(0, 0, 0)):
        # create a persistent reference to self so that a Python reference
        # counter doesn't delete the object (it needs to continue existing in
        # case the user runs ``pyplot.plot()`` later)
        # TODO this might be kind of dangerous. Make sure it can't lead to what
        # is effectively a memory leak.
        _displays.append(self)
        # TODO this is very sloppy and can have unintended side-effects on code
        # that is not ours; ideally, we should only remove these keybinds for
        # this figure, not the entire matplotlib session
        # remove default "n" and "p" keymappings
        for key, vals in mpl.rcParams.items():
            if key.startswith("keymap."):
                if "n" in vals: vals.remove("n")
                if "p" in vals: vals.remove("p")        
        self.im = im[..., :3].copy()
        self.im[np.isnan(self.im).any(axis=3)] = nancolor
        self.axis = axis
        
        self.slices = [slice(None)] * self.im.ndim
        self.slices[axis] = 0
        
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect("key_press_event", self._process_key)
        self.fig.canvas.mpl_connect("close_event", self._handle_close)
        self.title(title)
        
        self.ax = self.fig.add_subplot()
        self.ax.imshow(self.im[tuple(self.slices)])
        self._update_slice_text()
    def __del__(self):
        if self in _displays: _displays.remove(self)
    def _handle_close(self, _: mpl.backend_bases.CloseEvent):
        _displays.remove(self)
    def _update_slice_text(self):
        self.ax.set_title(
            f"{self.slices[self.axis] + 1} / {self.im.shape[self.axis]}"
        )

    def _next_slice(self):
        if self.slices[self.axis] + 1 < self.im.shape[self.axis]:
            self.slices[self.axis] += 1
    def _prev_slice(self):
        if self.slices[self.axis] > 0:
            self.slices[self.axis] -= 1
    def _process_key(self, event: mpl.backend_bases.KeyEvent):
        if   event.key == "n": self._next_slice()
        elif event.key == "p": self._prev_slice()
        self.ax.images[0].set_data(self.im[tuple(self.slices)])
        self._update_slice_text()
        self.fig.canvas.draw()
    # TODO implement. But to get this to block we need to implement our
    # own event loop or something (see ``help(pyplot.Figure.show)``)
    # def show(self):
    #     """This is a blocking call.""" # TODO proper docstring
    #     self.fig.show()
    def title(self, title: str):
        # TODO docstring?
        self.fig.suptitle(title)

# TODO maybe this should create one figure with 3 subplots? (But we must still
# be able to control each subplot independently.)
def _volumetric_displays(im: np.ndarray, titles: tuple=3*("",),
                        nancolor: tuple=(0, 0, 0)):
    # TODO docstring
    for axis in range(3):
        _volumetric_display(
            im, axis=axis, title=titles[axis], nancolor=nancolor
        )

def volumetric_displays_quats(im: np.ndarray, titles: tuple=3*("",),
                              nancolor: tuple=(0, 0, 0)):
    # TODO docstring
    # TODO input validation
    _volumetric_displays(
        range_map(im, (-1, 1), (0, 1)), titles=titles, nancolor=nancolor
    )