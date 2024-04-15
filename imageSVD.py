import numpy as np
import cv2  

def svd_compress(image, k):
    # Perform SVD on the image matrix
    U, Sigma, Vt = np.linalg.svd(image)
    
    U_truncated = U[:, :k]
    Sigma_truncated = np.diag(Sigma[:k])
    Vt_truncated = Vt[:k, :]
    
    compressed_image = np.dot(U_truncated, np.dot(Sigma_truncated, Vt_truncated))
    
    return compressed_image

image = cv2.imread('Ann_Arbor_sunset_2018.jpg', cv2.IMREAD_GRAYSCALE)

image_array = np.array(image)
compression_level = 100
compressed_image = svd_compress(image_array, compression_level)
cv2.imwrite('compressed_image.jpg', compressed_image) 