# Local interest points

* Edges are perfect local interest points as they are invariant to translation, rotation and illumination

**Harris detector** - a combined corner and edge detector   - good repeatability under changing illumination, translation and rotation

    - Take an image path (x,y) and shifting it by (dx, dy)
    - Compute spatial derivatives of the image Ix and Iy (smoothen the image prior to the gradient calculation)
    - Construct the structure tensor M = \sum_x,y [Ix^2 Ixy ; Ixy Iy^2] = [L1 0; 0 L2]
    - L1 and L2 are the eigenvalues of matrix M
    - Compute Harris response as R = L1*L2 - k*(L1+L2)^2 = det(M) - k*tr(M)^2
    - Find local maxima within e.g. a 3x3 window
    
* Partial derivatives are *invariant w.r.t. changes in illumination*
* Edge or corner? - Depends on the scale! 

**Blob detection**
* Laplacian of Gaussian (LoG): max response if the LoG size is of the order of the blob's size
* How to find blobs of different sizes? - Compute a convolution of the image with LoG of different sizes (scale-normalized LoGs!) 
* Max LoG reposonse is for sigma = r / sqrt(2), where r is the blob's radius
