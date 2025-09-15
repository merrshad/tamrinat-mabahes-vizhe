import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


img = mpimg.imread('./images/before_gray(Origin).jpg')


gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])


plt.imsave('./images/after_gray.jpg', gray, cmap='gray')
