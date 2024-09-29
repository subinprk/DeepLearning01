import numpy as np
from numpy.linalg import svd, norm
import matplotlib.pyplot as plt
from PIL import Image

# read the image and convert it to greyscale
X =  np.array(Image.open('harvey-saturday-goes7am.jpg').convert('L'))

# umcomment this block to check whether we read the image correctly
plt.imshow(X)
plt.show()

# perform SVD
U, s, V  = svd(X, full_matrices=False)
S = np.diag(s)

# reconstruct the image with k={2,10,40} singular vectors
for i in [2,10,40]:
    # ---------- make your implementation here -------------
    X_compressed = np.dot(U[:,:i], np.dot(S[:i,:i], V[:i,:]))
    Used_numbers = i
    # ---------- make your implementation here -------------
    plt.imshow(X_compressed)
    plt.show()
    print("Relative error is: {:.4f}".format(norm(X-X_compressed)/norm(X)))
    print("Used numbers: {}".format(Used_numbers))