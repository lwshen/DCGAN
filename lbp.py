from skimage.feature import local_binary_pattern
from skimage import data
import matplotlib.pyplot as plt

radius = 2
n_points = 8 * radius
method = 'uniform'

gen = data.load('./output/gen/0.jpg')
test = data.load('./output/test/0.jpg')

refs = {
    'gen': local_binary_pattern(gen, n_points, radius, method),
    'test': local_binary_pattern(test, n_points, radius, method)
}


# plot histograms of LBP of textures
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(nrows=2, ncols=4,
                                                       figsize=(12, 6))
plt.gray()

ax1.imshow(gen)
ax1.axis('off')
plt.hist(ax5, gen)
ax5.set_ylabel('Percentage')

ax2.imshow(test)
ax2.axis('off')
plt.hist(ax5, test)
ax6.set_xlabel('Uniform LBP values')

ax3.imshow(refs['gen'])
ax3.axis('off')
plt.hist(ax7, refs['gen'])

ax4.imshow(refs['test'])
ax4.axis('off')
plt.hist(ax8, refs['test'])

plt.show()