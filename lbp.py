from skimage.feature import local_binary_pattern
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
import os

radius = 2
n_points = 8 * radius
method = 'uniform'

dir = os.getcwd()
gen = data.load(dir + '/output/gen/00000.jpg', as_gray=True)
test = data.load(dir + '/output/test/00000.jpg', as_gray=True)
print(type(gen))
print(gen.shape)

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
ax5.hist(gen, bins='auto')
ax5.set_ylabel('Percentage')

ax2.imshow(test)
ax2.axis('off')
ax6.hist(test, bins='auto')
ax6.set_xlabel('Uniform LBP values')

ax3.imshow(refs['gen'])
ax3.axis('off')
ax7.hist(refs['gen'], bins='auto')

ax4.imshow(refs['test'])
ax4.axis('off')
ax8.hist(refs['test'], bins='auto')

plt.show()

T = 30

fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(nrows=2, ncols=3,
                                                       figsize=(18, 12))
ax1.imshow(gen)
ax1.axis('off')

arr1 = abs(gen - test) * 255
ax2.imshow(arr1)
ax2.axis('off')

shape = arr1.shape
result1 = np.zeros(shape)
for x in range(0, shape[0]):
    for y in range(0, shape[1]):
        if arr1[x, y] >= T:
            result1[x, y] = 255
ax5.imshow(result1)
ax5.axis('off')

ax3.imshow(test)
ax3.axis('off')

arr2 = abs(refs['gen'] - refs['test']) / 31.0 * 255
ax4.imshow(arr2)
ax4.axis('off')

shape = arr2.shape
result2 = np.zeros(shape)
for x in range(0, shape[0]):
    for y in range(0, shape[1]):
        if arr2[x, y] >= T:
            result2[x, y] = 255
ax6.imshow(result2)
ax6.axis('off')

plt.show()