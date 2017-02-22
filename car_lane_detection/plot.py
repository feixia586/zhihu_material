from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pylab

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



#####
# Gaussian Filter
#####
fig = plt.figure('Gaussian Filter')

ax = fig.add_subplot(1, 2, 1)
data = np.array([[1, 4, 7, 4, 1],
                 [4, 20, 33, 30, 4],
                 [7, 33, 55, 33, 7],
                 [4, 20, 33, 20, 4],
                 [1, 4, 7, 4, 1]])
ax.text(-0.1, 0.5, r'$\frac{1}{341}$', fontsize=30) # np.sum(data) == 341
ny, nx = data.shape
tbl = plt.table(cellText=data, loc=(0, 0), cellLoc='center')
tbl.set_fontsize(20)
tc = tbl.properties()['child_artists']
for cell in tc:
  cell.set_height(0.8/ny)
  cell.set_width(0.8/nx)

# hide spines and ticks
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])


ax = fig.add_subplot(1, 2, 2, projection='3d')
X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X, Y)
Z = pylab.bivariate_normal(X, Y)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0)
ax.set_xticklabels([])  # hide axis labels
ax.set_yticklabels([])
ax.set_zticklabels([])
# fig.colorbar(surf, shrink=0.5, aspect=5)

#####
# Parameter space and Hough space
#####
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3.0

fig = plt.figure('Hough Space')

x = np.arange(-0.5, 2.5, 0.01)
y = -x + 2
ax = fig.add_subplot(1, 3, 1)
plt.plot(x, y)
ax.text(0.5, 2, r'y=mx+b')
x = np.arange(0, 1, 0.01)
y = x
plt.plot(x, y, '--')
ax.text(0.3, 0.05, r'$\theta$', fontsize=25)
ax.text(0.4, 0.6, r'$\rho$', fontsize=25)
plt.xlabel('x')
plt.ylabel('y')
center_spines()
ax.set_xticklabels([])
ax.set_yticklabels([])

ax = fig.add_subplot(1, 3, 2)
ax.set_xlim([-2, 1])
ax.set_ylim([-0.25, 1])
plt.plot([-1], [0.5], 'o')
x = np.arange(-1, 0, 0.01)
y = [0.5] * len(x)
plt.plot(x, y, 'k--')
y = np.arange(0, 0.5, 0.01)
x = [-1] * len(y)
plt.plot(x, y, 'k--')
plt.xlabel('m')
plt.ylabel('b')
center_spines()
ax.set_xticklabels([])
ax.set_yticklabels([])

ax = fig.add_subplot(1, 3, 3)
ax.set_xlim([-1, 2])
ax.set_ylim([-0.25, 1])
plt.plot([1.2], [0.3], 'o')
x = np.arange(0, 1.2, 0.01)
y = [0.3] * len(x)
plt.plot(x, y, 'k--')
y = np.arange(0, 0.3, 0.01)
x = [1.2] * len(y)
plt.plot(x, y, 'k--')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\rho$')
center_spines()
ax.set_xticklabels([])
ax.set_yticklabels([])



plt.show()

