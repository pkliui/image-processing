# Local interest points

## Edges and corners detection

* Edges are perfect local interest points as they are invariant to translation, rotation and illumination

### **Harris detector** 
- a combined corner and edge detector   
- good repeatability under changing illumination, translation and rotation

    - Take an image path (x,y) and shifting it by (dx, dy)
    - Compute spatial derivatives of the image Ix and Iy (smoothen the image prior to the gradient calculation)
    - Construct the structure tensor M = \sum_x,y [Ix^2 Ixy ; Ixy Iy^2] = [L1 0; 0 L2]
    - L1 and L2 are the eigenvalues of matrix M
    - Compute Harris response as R = L1*L2 - k*(L1+L2)^2 = det(M) - k*tr(M)^2
    - Find local maxima within e.g. a 3x3 window
    
* Partial derivatives are *invariant w.r.t. changes in illumination*
* Edge or corner? - Depends on the scale! 


### Harris-Laplacian detector
* Harris detector is good at identifying the the corners under varying illumination conditions, rotation and translation. 
* However, it lacks scale invariance. 
* Harris-Laplace detector combines the classical Harris detector with the idea of Gaussian scale representation in order to create a scale-invariant detector. 



## Blob detection

### LoG
* Laplacian of Gaussian (LoG): max response if the LoG size is of the order of the blob's size
* How to find blobs of different sizes? - Compute a convolution of the image with LoG of different sizes (scale-normalized LoGs!) 
* For a fixed signa, max LoG repsonse is for sigma = r / sqrt(2), where r is the blob's radius

### DoG (difference of Gaussians)
* Lowe 2004

## Regions detection
## IBR (intensity-extrema-based) detector
## MSER (maximally stable extermal regions) detector



# Descriptors

* Descirbe local peculiarities of a point in an image

## SIFT - scale invariant feature transform
SIFT consists of:
 - DoG (position and scale features)
 - Orientation (dominant orientation using gradients)
 
