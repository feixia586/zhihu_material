import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects
import numpy as np

def center_spines(ax=None, centerx=0, centery=0):
  """Centers the axis spines at <centerx, centery> on the axis "ax", and
  places arrows at the end of the axis spines."""
  if ax is None:
    ax = plt.gca()
  
  # Set the axis's spines to be centered at the given point
  # (Setting all 4 spines so that the tick marks go in both directions)
  ax.spines['left'].set_position(('data', centerx))
  ax.spines['bottom'].set_position(('data', centery))
  ax.spines['right'].set_position(('data', centerx - 1))
  ax.spines['top'].set_position(('data', centery - 1))

  # Hide the line (but not ticks) for "extra" spines
  for side in ['right', 'top']:
      ax.spines[side].set_color('none')

  # On both the x and y axes...
  for axis, center in zip([ax.xaxis, ax.yaxis], [centerx, centery]):
    # Hide the ticklabels at <centerx, centery>
    formatter = CenteredFormatter()
    formatter.center = center
    axis.set_major_formatter(formatter)

  # Add offset ticklabels at <centerx, centery> using annotation
  # (Should probably make these update when the plot is redrawn...)
  xlabel, ylabel = map(formatter.format_data, [centerx, centery])
  if xlabel == ylabel:
    ax.annotate('%s' % xlabel, (centerx, centery),
            xytext=(-4, -4), textcoords='offset points',
            ha='right', va='top')
  else:
    ax.annotate('(%s, %s)' % (xlabel, ylabel), (centerx, centery),
            xytext=(-4, -4), textcoords='offset points',
            ha='right', va='top')


class CenteredFormatter(mpl.ticker.ScalarFormatter):
  """Acts exactly like the default Scalar Formatter, but yields an empty
  label for ticks at "center"."""
  center = 0
  def __call__(self, value, pos=None):
    if value == self.center:
      return ''
    else:
      return mpl.ticker.ScalarFormatter.__call__(self, value, pos)

###
# some basic configuration
###
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3.0

###
# sigmoid function and its gradient
###
fig = plt.figure('Sigmoid')
x = np.arange(-10, 10, 0.01)
y = 1.0 / (1.0 + np.exp(-x))
ax = fig.add_subplot(1, 2, 1)
ax.set_title(r'$\sigma(x)$', position=[0.25, 1])
plt.plot(x, y)
center_spines()

x = np.arange(-10, 10, 0.01)
y = np.exp(x) / (np.exp(x) + 1.0) ** 2
ax = fig.add_subplot(1, 2, 2)
ax.set_title(r'$\mathrm{d}\sigma(x) / \mathrm{d}x$', position=[0.25, 1])
plt.plot(x, y)
center_spines()

###
# tanh (hyperbolic tangent) function and its gradient
###
fig = plt.figure('tanh')
x = np.arange(-10, 10, 0.01)
y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
ax = fig.add_subplot(1, 2, 1)
ax.set_title(r'$\tanh(x)$', position=[0.25, 1])
plt.plot(x, y)
center_spines()

x = np.arange(-10, 10, 0.01)
y = (2 / (np.exp(x) + np.exp(-x))) ** 2
ax = fig.add_subplot(1, 2, 2)
ax.set_title(r'$\mathrm{d}\tanh(x) / \mathrm{d}x$', position=[0.25, 1])
plt.plot(x, y)
center_spines()

###
# ReLU function and its gradient
###
fig = plt.figure('ReLU')
x = np.arange(-6, 6, 0.01)
y = [0 if a < 0 else a for a in x]
ax = fig.add_subplot(1, 2, 1)
ax.set_title(r'$\mathrm{ReLU}(x)$', position=[0.25, 1])
plt.plot(x, y)
center_spines()

x = np.arange(-6, 6, 0.01)
y = [0 if a < 0 else 1 for a in x]
ax = fig.add_subplot(1, 2, 2)
ax.set_title(r'$\mathrm{dReLU}(x) / \mathrm{d}x$', position=[0.25, 1])
plt.plot(x, y)
center_spines()

###
# Leaky ReLU function and its gradient
###
fig = plt.figure('Leaky ReLU')
x = np.arange(-10, 10, 0.01)
y = [0.01 * a if a < 0 else a for a in x]
ax = fig.add_subplot(1, 2, 1)
ax.set_title(r'$f(x)$', position=[0.25, 1])
plt.plot(x, y)
center_spines()

x = np.arange(-10, 10, 0.01)
y = [0.01 if a < 0 else 1 for a in x]
ax = fig.add_subplot(1, 2, 2)
ax.set_title(r'$\mathrm{d}f(x) / \mathrm{d}x$', position=[0.25, 1])
plt.plot(x, y)
center_spines()

###
# ELU function and its gradient
###
fig = plt.figure('ELU')
x = np.arange(-10, 10, 0.01)
y = [(1.0 * (np.exp(a) - 1)) if a < 0 else a for a in x]
ax = fig.add_subplot(1, 2, 1)
ax.set_title(r'$f(x)$', position=[0.25, 1])
plt.plot(x, y)
center_spines()

x = np.arange(-10, 10, 0.01)
y = [(1.0 * np.exp(a)) if a < 0 else 1 for a in x]
ax = fig.add_subplot(1, 2, 2)
ax.set_title(r'$\mathrm{d}f(x) / \mathrm{d}x$', position=[0.25, 1])
plt.plot(x, y)
center_spines()
plt.show()
