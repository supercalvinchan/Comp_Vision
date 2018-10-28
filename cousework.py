
# Import libaries
import imageio
import numpy as np
import matplotlib.pyplot as plt
import noisy
import scipy
import scipy.signal
import math
import time
from math import pi

1.1
# Design the filter h
h = [[1/3,1/3,1/3], [1/3,1/3,1/3], [1/3,1/3,1/3]]

# Print the filter
print(h)

# Convolve the corrupted image with h using scipy.signal.convolve2d function
image_filtered = scipy.signal.convolve2d(image_noisy, h, mode='same')
plt.imshow(image_filtered, cmap='gray')

1.2
# Design the filter h
h = [[1/25,1/25,1/25,1/25,1/25],[1/25,1/25,1/25,1/25,1/25],[1/25,1/25,1/25,1/25,1/25],[1/25,1/25,1/25,1/25,1/25],[1/25,1/25,1/25,1/25,1/25]]

# Print the filter
print(h)

# Convolve the corrupted image with h using scipy.signal.convolve2d function
image_filtered = scipy.signal.convolve2d(image_noisy, h, mode='same')
plt.imshow(image_filtered, cmap='gray')

1.3
# Design the filter h
h = [[1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],[1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],[1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],[1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],[1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],[1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],[1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],[1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],[1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81]]

# Print the filter
print(h)

# Convolve the corrupted image with h using scipy.signal.convolve2d function
image_filtered = scipy.signal.convolve2d(image_noisy, h, mode='same')
plt.imshow(image_filtered, cmap='gray')

1.4
When the window size increases, the picture become more blurred and smooth. The moving average filter has removed high frequency component.

2.1
# Design the Sobel filters
h_sobel_x = [[1,0,-1],[2,0,-2],[1,0,-1]]
h_sobel_y = [[1,2,1],[0,0,0],[-1,-2,-1]]

# Print the filters
print(h_sobel_x)
print(h_sobel_y)

# Sobel filtering
sobel_x = scipy.signal.convolve2d(image_noisy, h_sobel_x, mode='same')
sobel_y = scipy.signal.convolve2d(image_noisy, h_sobel_y, mode='same')

# Calculate the gradient magnitude
sobel_mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)

# Display the magnitude
plt.imshow(sobel_mag, cmap='gray')


2.2
# Design the Gaussian filter
def gaussian_filter_2d(sigma):
    # sigma: the parameter sigma in the Gaussian kernel (unit: pixel)
    #
    # return: a 2D array for the Gaussian kernel

    # The filter radius is 3.5 times sigma
    rad = int(math.ceil(3.5 * sigma))
    sz = 2 * rad + 1

    #initializes the matrix
    h = np.zeros(((sz-1),(sz-1))).astype(float)


    #calculating standard difference
    #in python matrix, vertical comes first in index
    x = (sz-1)/2
    y = (sz-1)/2

    #looping through each coordinate to change values
    for j in range(0,sz-1):#vertical axis
        for i in range(0,sz-1): #horizontal axis
            h[j][i] = (1/(2*pi*sigma*sigma))*math.exp(-((j-y)*(j-y)+(i-x)*(i-x))/(2*sigma*sigma))

      #divide the sum of the all kernels
    return h/h.sum()


# Display the Gaussian filter when sigma = 3 pixel
sigma = 3
h = gaussian_filter_2d(sigma)
plt.imshow(h)


2.3
# Perform Gaussian smoothing before Sobel filtering
sigma = 3
h = gaussian_filter_2d(sigma)
image_smoothed = scipy.signal.convolve2d(image_noisy, h, mode='same')

# Sobel filtering
sobel_x = scipy.signal.convolve2d(image_smoothed, h_sobel_x, mode='same')
sobel_y = scipy.signal.convolve2d(image_smoothed, h_sobel_y, mode='same')

# Calculate the gradient magnitude
sobel_mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)

# Display the magnitude
plt.imshow(sobel_mag, cmap='gray')

2.4
# Create the Gaussian filter
sigma = 7
h = gaussian_filter_2d(sigma)

# Perform Gaussian smoothing
start = time.time()
image_smoothed = scipy.signal.convolve2d(image_noisy, h, mode='same')
duration = time.time() - start
print('It takes {0:.6f} seconds for performing Gaussian smoothing.'.format(duration))

# Sobel filtering
sobel_x = scipy.signal.convolve2d(image_smoothed, h_sobel_x, mode='same')
sobel_y = scipy.signal.convolve2d(image_smoothed, h_sobel_y, mode='same')

# Calculate the gradient magnitude
sobel_mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)

# Display the magnitude
plt.imshow(sobel_mag, cmap='gray')

2.5
# Design the Gaussian filter
def gaussian_filter_1d(sigma):
    # sigma: the parameter sigma in the Gaussian kernel (unit: pixel)
    #
    # return: a 1D array for the Gaussian kernel

    # The filter radius is 3.5 times sigma
    rad = int(math.ceil(3.5 * sigma))
    sz = 2 * rad + 1

    #size (sz-1)
    x = (sz-1)/2
    h = np.zeros((sz-1))

    for i in range(0,sz-1):
        h[i] = (1/(math.sqrt(2*pi)*sigma))*math.exp((-(i-x)*(i-x))/(2*sigma*sigma))
    return h

# Display the Gaussian filters when sigma = 7 pixel
sigma = 7

# The Gaussian filter along x-axis. Its shape is (1, sz).
h_x = gaussian_filter_1d(sigma)
h_x = np.expand_dims(h_x, axis=0)

# The Gaussian filter along y-axis. Its shape is (sz, 1).
h_y = gaussian_filter_1d(sigma)
h_y = np.expand_dims(h_y, axis=-1)

# Display the filters
plt.subplot(1, 2, 1)
plt.imshow(h_x)
plt.subplot(1, 2, 2)
plt.imshow(h_y)

2.6
# Perform separable Gaussian smoothing before Sobel filtering
start = time.time()
image_smoothed = scipy.signal.convolve2d(image_noisy, h_x, mode='same')
image_smoothed = scipy.signal.convolve2d(image_smoothed, h_y, mode='same')
duration = time.time() - start
print('It takes {0:.6f} seconds for performing Gaussian smoothing.'.format(duration))

# Sobel filtering
sobel_x = scipy.signal.convolve2d(image_smoothed, h_sobel_x, mode='same')
sobel_y = scipy.signal.convolve2d(image_smoothed, h_sobel_y, mode='same')

# Calculate the gradient magnitude
sobel_mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)

# Display the magnitude
plt.imshow(sobel_mag, cmap='gray')



2.7
The first picture compares to the second one(sigma of the gaussian increases) is more blurred in edge but remove more noise.By applying separatable filters, the running time reduces, and it is obvious when the kernel size increases.

3.1
# Design the filter
h = [[0,1,0],[1,-4,1],[0,1,0]]

# Laplacian filtering
lap = scipy.signal.convolve2d(image_noisy, h, mode='same')

# Display the results
plt.imshow(lap, cmap='gray')

3.2
# Design the Gaussian filter
sigma = 3

# The Gaussian filter along x-axis
h_x = gaussian_filter_1d(sigma)
h_x = np.expand_dims(h_x, axis=0)

# The Gaussian filter along y-axis
h_y = gaussian_filter_1d(sigma)
h_y = np.expand_dims(h_y, axis=-1)

# Gaussian smoothing
image_smoothed = scipy.signal.convolve2d(image_noisy, h_x, mode='same')
image_smoothed = scipy.signal.convolve2d(image_smoothed, h_y, mode='same')

# Design the Laplacian filter
h = [[0,1,0],[1,-4,1],[0,1,0]]

# Laplacian filtering
lap = scipy.signal.convolve2d(image_smoothed, h, mode='same')
plt.imshow(lap, cmap='gray')

3.3
The first one is very sensitive to noise, so we can spot that the picture is more messy. The second one is more clear, because it reduces high frequency noise.

4.
4 hours.
