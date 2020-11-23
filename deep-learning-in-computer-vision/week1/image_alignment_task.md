# Image alignment

In this task, you will have to solve two image alignment problems: channel processing and face alignment. You can get **10 points** implementing all the passed functions (7.5 for the first part and 2.5 for the second one).

# Image channels processing and alignment (7.5 points)

## Problem review

Sergey Prokudin-Gorsky was the first color photographer in Russia, who made the color portrait of Leo Tolstoy. Each of his photographs is three black-and-white photo plates, corresponding to red, green, and blue color channels. Currently, the collection of his pictures is situated in the U.S. Library of Congress, the altered versions have proliferated online. In this task, you should make a programme which will align the images from the Prokudin-Gorsky plates and learn the basic image processing methods.

*The input image and the result of the alignment:*
<img src="http://cdn1.savepice.ru/uploads/2017/7/31/8e68237bfd49026d137f59283db18b29-full.png">


```python
%pylab inline
import matplotlib.pyplot as plt 
import numpy as np
import os
import cv2
```

    Populating the interactive namespace from numpy and matplotlib


## Problem description

#### Input image loading

The input image is the set of 3 plates, corresponding to B, G, and R channels (top-down). You should implement the function $\tt{load}$\_$\tt{data}$ that reads the data and returns the list of images of plates.
$\tt{dir}$\_$\tt{name}$ is the path to the directory with plate images. If this directory is located in the same directory as this notebook, then default arguments can be used.


```python
def load_data(dir_name = 'plates'):
    images = []
    for filename in os.listdir(dir_name):
        image = cv2.imread(os.path.join(dir_name,filename))
        if image is not None:
            images.append(image)
    return images

plates = load_data()

print(os.listdir('plates'))
```

    ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png']


The dataset is a list of 2-dimensional arrays.


```python
# The auxiliary function `visualize()` displays the images given as argument.
def visualize(imgs, format=None):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt_idx = i+1
        plt.subplot(3, 3, plt_idx)    
        plt.imshow(img)
        print(img.shape)
    plt.show()

visualize(plates, 'gray')



# Function `visualize_RGB()`to display RGB images correctly
def visualize_RGB(imgs, format=None):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt_idx = i+1
        plt.subplot(3, 3, plt_idx)    
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print(img.shape)
    plt.show()
```

    (1153, 448, 3)
    (1173, 457, 3)
    (1024, 398, 3)
    (1161, 449, 3)
    (1154, 444, 3)
    (1160, 448, 3)



![png](output_files/output_6_1.png)


#### The borders removal (1.5 points)
It is worth noting that there is a framing from all sides in most of the images. This framing can appreciably worsen the quality of channels alignment. Here, we suggest that you find the borders on the plates using Canny edge detector, and crop the images according to these edges. The example of using Canny detector implemented in skimage library can be found [here](http://scikit-image.org/docs/dev/auto_examples/edges/plot_canny.html).<br>

The borders can be removed in the following way:
* Apply Canny edge detector to the image.
* Find the rows and columns of the frame pixels. 
For example, in case of upper bound we will search for the row in the neighborhood of the upper edge of the image (e.g. 5% of its height). For each row let us count the number of edge pixels (obtained using Canny detector) it contains. Having these number let us find two maximums among them. Two rows corresponding to these maximums are edge rows. As there are two color changes in the frame (firstly, from light scanner background to the dark tape and then from the tape to the image), we need the second maximum that is further from the image border. The row corresponding to this maximum is the crop border. In order not to find two neighboring peaks, non-maximum suppression should be implemented: the rows next to the first maximum are set to zero, and after that, the second maximum is searched for.

#### Canny detector implementation (2.5 points)
You can write your own implementation of Canny edge detector to get extra points. <br>

Canny detection algorithm:
1. *Noise reduction.* To remove noise, the image is smoothed by Gaussian blur with the kernel of size $5 \times 5$ and $\sigma = 1.4$. Since the sum of the elements in the Gaussian kernel equals $1$, the kernel should be normalized before the convolution. <br><br>

2. *Calculating gradients.* When the image $I$ is smoothed, the derivatives $I_x$ and $I_y$ w.r.t. $x$ and $y$ are calculated. It can be implemented by convolving $I$ with Sobel kernels $K_x$ and $K_y$, respectively: 
$$ K_x = \begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix}, K_y = \begin{pmatrix} 1 & 2 & 1 \\ 0 & 0 & 0 \\ -1 & -2 & -1 \end{pmatrix}. $$ 
Then, the magnitude $G$ and the slope $\theta$ of the gradient are calculated:
$$ |G| = \sqrt{I_x^2 + I_y^2}, $$
$$ \theta(x,y) = arctan\left(\frac{I_y}{I_x}\right)$$<br><br>

3. *Non-maximum suppression.* For each pixel find two neighbors (in the positive and negative gradient directions, supposing that each neighbor occupies the angle of $\pi /4$, and $0$ is the direction straight to the right). If the magnitude of the current pixel is greater than the magnitudes of the neighbors, nothing changes, otherwise, the magnitude of the current pixel is set to zero.<br><br>

4. *Double threshold.* The gradient magnitudes are compared with two specified threshold values, the first one is less than the second. The gradients that are smaller than the low threshold value are suppressed; the gradients higher than the high threshold value are marked as strong ones and the corresponding pixels are included in the final edge map. All the rest gradients are marked as weak ones and pixels corresponding to these gradients are considered in the next step.<br><br>

5. *Edge tracking by hysteresis.* Since a weak edge pixel caused from true edges will be connected to a strong edge pixel, pixel $w$ with weak gradient is marked as edge and included in the final edge map if and only if it is involved in the same blob (connected component) as some pixel $s$ with strong gradient. In other words, there should be a chain of neighbor weak pixels connecting $w$ and $s$ (the neighbors are 8 pixels around the considered one). You are welcome to make up and implement an algorithm that finds all the connected components of the gradient map considering each pixel only once.  After that, you can decide which pixels will be included in the final edge map (this algorithm should be single-pass, as well).


```python
#################################################################
# TODO: implement Canny detector yourself.                      #
#       You can use methods from scipy.ndimage if you need.     #
#################################################################
from  skimage.feature import canny

from scipy import signal
import numpy as np


def gaussian_kernel(size, std):
    # create 2D Gaussian kernel
    kernel_1d = signal.gaussian(size, std).reshape(size, 1)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    #
    #normalise
    kernel_2d = kernel_2d  / (np.sum(kernel_2d))
    #
    return kernel_2d

def Sobel_gradients(img):
    #
    #Sobel kernels
    K_x = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]
    
    K_y = [[1, 2, 1],
          [0, 0, 0],
          [-1, -2,- 1]]
    #
    #Intensity gradients
    I_x = signal.convolve(img, K_x, 'same')
    I_y = signal.convolve(img, K_y, 'same')
    #
    #Gradient amplitude and phase
    gradient = np.sqrt(I_x**2 + I_y**2)
    slope = np.arctan(I_y / (I_x + 0.0001))
    #
    return gradient, slope

def nonmax_suppression(gradient, slope):
    #
    #initialise thresholded gradient
    thresholded_gradient = np.zeros(gradient.shape, dtype=np.float64)
    #
    #convert the phase to deg
    slope = slope * 180 / np.pi
    slope[slope<0] += 180
    #
    #loop through the slope to find the neighbours of each pixel
    for x in range(1,slope.shape[0]-1):
        for y in range(1, slope.shape[1]-1):
            #
            #in 0 deg direction and its opposite
            if (0.0 <= slope[x,y] < 22.5)  or (157.5 <= slope[x,y] < 180.0):
                neighbour1 = gradient[x,y-1]
                neighbour2 = gradient[x,y+1]
            #in 45 deg direction and its opposite
            elif (22.5 <= slope[x,y] < 67.5):
                neighbour1 = gradient[x+1,y-1]
                neighbour2 = gradient[x-1,y+1]
            #in 90 deg direction and its opposite
            elif (67.5 <= slope[x,y] < 112.5):
                neighbour1 = gradient[x+1,y]
                neighbour2 = gradient[x-1,y]
            #in 135 deg direction and its opposite
            elif (112.5 <= slope[x,y] < 157.5):
                neighbour1 = gradient[x+1,y+1]
                neighbour2 = gradient[x-1,y-1]
            #
            #check if the current pixel value is larger than its neighbour's value and change it
            #otherwise leave it as it is 
            if (gradient[x,y] >= neighbour1) and (gradient[x,y] >= neighbour2):
                thresholded_gradient[x,y] = gradient[x,y]
            #    
    return thresholded_gradient

def double_threshold(thresholded_gradient, low_threshold, high_threshold):
    #impose low and high threshold to obtain weakly and strongly thresholded gradients
    #initialise gradients
    gradient_weak = np.zeros(thresholded_gradient.shape)
    gradient_strong = np.zeros(thresholded_gradient.shape)
    #
    #obtain strong gradient (pixel values higher than the high threshold)
    gradient_strong[thresholded_gradient>=high_threshold] = thresholded_gradient[thresholded_gradient>=high_threshold]
    #
    #obtain weak gradient (pixel values between high and weak thresholds)
    gradient_weak[thresholded_gradient>=low_threshold] = thresholded_gradient[thresholded_gradient>=low_threshold]
    gradient_weak[thresholded_gradient<high_threshold] = 0
    #
    return gradient_weak, gradient_strong

def edge_tracking(gradient_weak, strong=255):
    # implement edge tracking 
    # find if the weak gradient's pixels are connected to the strong gradient's pixels
    # if not, set them to 0
    for x in range(1, gradient_weak.shape[0]-1):
        for y in range(1, gradient_weak.shape[1]-1):
            if ((gradient_weak[x,y+1] == strong) or (gradient_weak[x-1,y+1] == strong)
                or (gradient_weak[x-1,y] == strong) or (gradient_weak[x-1,y-1] == strong)
                or (gradient_weak[x,y-1] == strong) or (gradient_weak[x+1,y-1] == strong)
                or (gradient_weak[x+1,y] == strong) or (gradient_weak[x+1,y+1] == strong)):
                    gradient_weak[x,y] = strong
            else:
                gradient_weak[x,y] = 0
    #            
    return gradient_weak
                
def Canny_detector(img):
    # implemet Canny detector
    # convolve original image
    kernel_2d = gaussian_kernel(size=5, std=1.4)
    img_convolved = signal.convolve(img, kernel_2d, 'same') 
    #
    # compute its gradient amplitude and phase
    gradient, slope = Sobel_gradients(img_convolved)
    # threshold gradient amplitude by nonmax suppression
    thresholded_gradient = nonmax_suppression(gradient, slope)
    # generate (edge tracked) weak and strong gradients
    gradient_weak, gradient_strong = double_threshold(thresholded_gradient, 10, 50)
    gradient_weak = edge_tracking(gradient_weak, strong=255)
    # generate final image 
    canny_img = gradient_weak + gradient_strong
    
    return canny_img

canny_imgs = []
for img in plates:
    img = img[:,:,0]
    canny_img = Canny_detector(img)
    canny_imgs.append(canny_img)
    
visualize(canny_imgs, 'gray')
```

    (1153, 448)
    (1173, 457)
    (1024, 398)
    (1161, 449)
    (1154, 444)
    (1160, 448)



![png](output_files/output_9_1.png)



```python
#################################################################
# TODO: Implement the removal of the tape borders using         #
#       the output of Canny edge detector ("canny_img" list)    #
#################################################################

def remove_borders(img, canny_img):
    threshold = 0
    threshold_number = 0 
    #top
    for top in range(1, int(0.02*canny_img.shape[0])):
        upper_border = canny_img[top, :]
        counts = np.sum(upper_border)
        if counts > threshold:
            if top ==1:
                threshold = counts
            threshold_number += 1
            #at the second border threshold_number == 3:
            if threshold_number == 2:
                print('top=',top)
    
    threshold = 0
    threshold_number = 0
    #bottom
    for bottom in range(1, int(0.02*canny_img.shape[0])):
        bottom_border = canny_img[-bottom, :]
        counts = np.sum(bottom_border)
        if counts > threshold:
            if top ==1:
                threshold = counts
            threshold_number += 1
            #at the second border threshold_number == 3:
            if threshold_number == 2:
                print('bottom=',bottom)  
     
    threshold = 0
    threshold_number = 0
    #left
    for left in range(1, int(0.02*canny_img.shape[1])):
        left_border = canny_img[:, left]
        counts = np.sum(left_border)
        if counts > threshold:
            if top ==1:
                threshold = counts
            threshold_number += 1
            #at the second border threshold_number == 3:
            if threshold_number == 2:
                print('left=',left)
                
    threshold = 0
    threshold_number = 0           
    #right
    for right in range(1, int(0.02*canny_img.shape[1])):
        right_border = canny_img[:, -right]
        counts = np.sum(right_border)
        if counts > threshold:
            if top ==1:
                threshold = counts
            threshold_number += 1
            #at the second border threshold_number == 3:
            if threshold_number == 2:
                print('right=',right)
         
    return img[top+1:-bottom-1,left+1:-right-1]
                
cropped_imgs = []

#crop borders
for i, img in enumerate(plates):
    cropped_imgs.append(remove_borders(img, canny_imgs[i]))
    
visualize(cropped_imgs, 'gray')
```

    top= 12
    bottom= 8
    left= 7
    right= 3
    top= 12
    bottom= 11
    left= 4
    top= 6
    bottom= 10
    right= 3
    top= 12
    bottom= 9
    right= 3
    top= 8
    bottom= 8
    top= 14
    bottom= 10
    right= 3
    (1107, 432, 3)
    (1127, 439, 3)
    (984, 384, 3)
    (1115, 433, 3)
    (1108, 428, 3)
    (1114, 432, 3)



![png](output_files/output_10_1.png)


#### Channels separation  (0.5 points)

The next step is to separate the image into three channels (B, G, R) and make one colored picture. To get channels, you can divide each plate into three equal parts.


```python
#################################################################
# TODO: implement the function impose_components transforming   #
#       cropped black-and-white images cropped_imgs             #
#       into the list of RGB images rgb_imgs                    #
#################################################################
import numpy as np
from copy import deepcopy

def impose_components(img):
    #copy only one channel, because they are all the same anyway
    img0 = img[:,:,0]
    #
    #delete pixels at the bottom so that the image height is modulo 3
    img_mod3 = np.zeros((img0.shape[0]-img0.shape[0]%3, img0.shape[1]))
    img_mod3 = deepcopy(img0[0:img_mod3.shape[0],:])
    #
    #divide each image into three channels (R,G,B)
    r = np.zeros((img_mod3.shape[0]//3, img_mod3.shape[1]))
    g = np.zeros((img_mod3.shape[0]//3, img_mod3.shape[1])) 
    b = np.zeros((img_mod3.shape[0]//3, img_mod3.shape[1])) 
    #
    #separate channels
    r = img_mod3[0:img_mod3.shape[0]//3,:] 
    g = img_mod3[img_mod3.shape[0]//3:2*img_mod3.shape[0]//3,:]
    b = img_mod3[2*img_mod3.shape[0]//3:img_mod3.shape[0],:]
    #
    #create RGB image
    rgb = np.dstack((r,g,b))
    #
    return rgb

rgb_imgs = []
for cropped_img in cropped_imgs:
    rgb_img = impose_components(cropped_img)
    rgb_imgs.append(rgb_img)

#use custom visualise_RGB function (defined  in cell 3) to show truly RGB images    
visualize_RGB(rgb_imgs)
```

    (369, 432, 3)
    (375, 439, 3)
    (328, 384, 3)
    (371, 433, 3)
    (369, 428, 3)
    (371, 432, 3)



![png](output_files/output_12_1.png)


#### Search for the best shift for channel alignment (1 point for metrics implementation + 2 points for channel alignment)

In order to align two images, we will shift one image relative to another within some limits (e.g. from $-15$ to $15$ pixels). For each shift, we can calculate some metrics in the overlap of the images. Depending on the metrics, the best shift is the one the metrics achieves the greatest or the smallest value for. We suggest that you implement two metrics and choose the one that allows to obtain the better alignment quality:

* *Mean squared error (MSE):*<br><br>
$$ MSE(I_1, I_2) = \dfrac{1}{w * h}\sum_{x,y}(I_1(x,y)-I_2(x,y))^2, $$<br> where *w, h* are width and height of the images, respectively. To find the optimal shift you should find the minimum MSE over all the shift values.
    <br><br>
* *Normalized cross-correlation (CC):*<br><br>
    $$
    I_1 \ast I_2 = \dfrac{\sum_{x,y}I_1(x,y)I_2(x,y)}{\sum_{x,y}I_1(x,y)\sum_{x,y}I_2(x,y)}.
    $$<br>
    To find the optimal shift you should find the maximum CC over all the shift values.


```python
#################################################################
# TODO: implement the functions mse и cor calculating           #
#       mean squared error and normalized cross-correlation     #
#       for two input images, respectively (1 point)            #
#################################################################
def mse(X, Y):
    metrics =  np.sum((X/np.max(X)-Y/np.max(Y))**2 )/(X.shape[0]*X.shape[1])  
    return metrics

def cor(X, Y):
    metrics = np.sum((X-np.mean(X))*(Y-np.mean(Y)))/ np.sqrt(np.sum(((X-np.mean(X))**2) * ((Y-np.mean(Y))**2)))
    return metrics
```


```python
#################################################################
# TODO: implement the algorithm to find the best channel        #
#       shift and the alignment. Apply this algorithm for       #
#       rgb_imgs processing and get the final list of colored   #
#       pictures. These images will be used for the evaluation  #
#       of the quality of the whole algorithm.  (2 points)      #
#                                                               #
#       You can use the following interface or write your own.  #
#################################################################
from scipy import ndimage as ndi

import cv2


def get_best_shift_mse(channel1, channel2):
    #function to determine the best shift using MSE metric
    metrics_mse = []
    shifts_mse = []
    #shift images and look for the min of MSE metric
    for x_shift in range(-15,16):
        for y_shift in range(-15,16):
            channel2_shifted = ndi.shift(channel2, 
                                     shift=[x_shift, y_shift], 
                                     mode='nearest')            
            metric_mse = mse(channel1, channel2_shifted)
            #
            #record metric values and the shifts
            metrics_mse.append(metric_mse)
            shifts_mse.append([x_shift, y_shift])
    #        
    #find the minimum difference 
    idx = np.argmin(metrics_mse)
    #and the corresponding displacement
    best_shift = shifts_mse[idx]
    print(best_shift)
    #
    return best_shift

def get_best_shift_cor(channel1, channel2):
    #function to determine the best shift using cross-correlation metric    
    metrics_cor = []
    shifts_cor = []
    for x_shift in range(-15,16):
        for y_shift in range(-15,16):
            channel2_shifted = ndi.shift(channel2, 
                                     [x_shift, y_shift], 
                                     mode='nearest')
            metric_cor = cor(channel1, channel2_shifted)
            #record metric values and the shifts
            metrics_cor.append(metric_cor)
            shifts_cor.append([x_shift, y_shift])
    #        
    #find the maximum value of the correlation
    idx = np.argmax(metrics_cor)
    #and the corresponding displacement
    best_shift = shifts_cor[idx]
    print(best_shift)
    #
    return best_shift
        

def get_best_image(rgb_img, metric):
    image_aligned = np.zeros(rgb_img.shape)
    #
    #the first channel is our reference
    image_aligned[:,:,0] = rgb_img[:,:,0]
    #
    #MSE metric case
    if metric == 'mse':
        #align the second channel
        shift1 = get_best_shift_mse(rgb_img[:,:,0], rgb_img[:,:,1])
        image_aligned[:,:,1] = ndi.shift(rgb_img[:,:,1], 
                                     shift=shift1, 
                                     mode='nearest')
        #align the third channel
        shift2 = get_best_shift_mse(rgb_img[:,:,0], rgb_img[:,:,2])
        image_aligned[:,:,2] = ndi.shift(rgb_img[:,:,2], 
                                     shift=shift2, 
                                     mode='nearest')
    #
    # cross-correlation metric case    
    elif metric == 'cor':
        #align the second channel
        shift1 = get_best_shift_cor(rgb_img[:,:,0], rgb_img[:,:,1])
        image_aligned[:,:,1] = deepcopy(ndi.shift(rgb_img[:,:,1], 
                                     shift=shift1, 
                                     mode='nearest'))
        #align the third channel
        shift2 = get_best_shift_cor(rgb_img[:,:,0], rgb_img[:,:,2])
        image_aligned[:,:,2] = ndi.shift(rgb_img[:,:,2], 
                                     shift=shift2, 
                                     mode='nearest') 
    else:
        raise ValueError('Wrong metric type, must be either "mse" or "cor" !')    
    return image_aligned
```

### Image registration using MSE metrics


```python
final_imgs = []
for img in rgb_imgs:
    final_img = (get_best_image(img, metric='mse')* 1).astype(np.uint8)
    final_imgs.append(final_img)

visualize_RGB(final_imgs)
```

    [-9, 1]
    [-15, 2]
    [-13, -1]
    [-15, -2]
    [-10, 0]
    [-15, 0]
    [-9, 1]
    [-15, 1]
    [-13, 1]
    [15, 0]
    [-9, 3]
    [-15, 3]
    (369, 432, 3)
    (375, 439, 3)
    (328, 384, 3)
    (371, 433, 3)
    (369, 428, 3)
    (371, 432, 3)



![png](output_files/output_17_1.png)


### Image registration using cross-correlation metrics


```python
final_imgs = []
for img in rgb_imgs:
    final_img = (get_best_image(img, metric='cor')* 1).astype(np.uint8)
    final_imgs.append(final_img)

visualize_RGB(final_imgs)
```

    [-9, 1]
    [-15, 2]
    [-13, -1]
    [-15, -2]
    [-10, 0]
    [-15, -1]
    [-11, 0]
    [-15, 0]
    [-13, 1]
    [15, 0]
    [-9, 1]
    [-15, 3]
    (369, 432, 3)
    (375, 439, 3)
    (328, 384, 3)
    (371, 433, 3)
    (369, 428, 3)
    (371, 432, 3)



![png](output_files/output_19_1.png)


# Face Alignment (2.5 points)

In this task, you have to implement face normalization and alignment. Most of the face images deceptively seem to be aligned, but since many face recognition algorithms are very sensitive to shifts and rotations, we need not only to find a face on the image but also normalize it. Besides, the neural networks usually used for recognition have fixed input size, so, the normalized face images should be resized as well.

There are six images of faces you have to normalize. In addition, you have the coordinates of the eyes in each of the pictures. You have to rotate the image so that the eyes are on the same height, crop the square box containing the face and transform it to the size $224\times 224.$ The eyes should be located symmetrically and in the middle of the image (on the height).

Here is an example of how the transformation should look like.

<img src = "https://cdn1.savepice.ru/uploads/2017/12/13/286e475ef7a4f4e59005bcf7de78742f-full.jpg">

#### Get data
You get the images and corresponding eyes coordinates for each person. You should implement the  function $\tt{load}$\_$\tt{faces}$\_$\tt{and}$\_$\tt{eyes}$ that reads the data and returns two dictionaries: the dictionary of images and the dictionary of eyes coordinates. Eyes coordinates is a list of two tuples $[(x_1,y_1),(x_2,y_2)]$.
Both dictionaries should have filenames as the keys.

$\tt{dir}$\_$\tt{name}$ is the path to the directory with face images, $\tt{eye}$\_$\tt{path}$ is the path to .pickle file with eyes coordinates. If these directory and file are located in the same directory as this notebook, then default arguments can be used.


```python
import pickle

def load_faces_and_eyes(dir_name = 'faces_imgs', eye_path = './eyes.pickle'):
    images_dic = {}
    eyes_dic = {}
    images = []
    filenames = []
    #
    for filename in os.listdir(dir_name):
        image = cv2.imread(os.path.join(dir_name,filename))
        if image is not None:
            images.append(image)
            filenames.append(filename)
    #
    #write image data to a dictionary
    images_dic = dict(zip(filenames, images))
    #
    #write eyes data
    eyes_dic = pickle.load(open("./eyes.pickle","rb"))
    return images_dic, eyes_dic
    
faces, eyes = load_faces_and_eyes()
```

Here is how the input images look like:


```python
visualize_RGB(faces.values())
```

    (516, 504, 3)
    (345, 297, 3)
    (594, 431, 3)
    (574, 366, 3)
    (742, 582, 3)
    (621, 566, 3)



![png](output_files/output_26_1.png)


You may make the transformation using your own algorithm or by the following steps:
1. Find the angle between the segment connecting two eyes and horizontal line;
2. Rotate the image;
3. Find the coordinates of the eyes on the rotated image
4. Find the width and height of the box containing the face depending on the eyes coordinates
5. Crop the box and resize it to $224\times224$


```python
#################################################################
# TODO: implement the function transform_face that rotates      #
#       the image so that the eyes have equal ordinate,         #
#       crops the square box containing face and resizes it.    #
#       You can use methods from skimage library if you need.   #
#       (2.5 points)                                              #
#################################################################
from scipy import ndimage as ndi
from scipy import signal
from skimage import transform
from skimage.feature import peak_local_max

def transform_face(image, eyes):
    #
    #find original eyes' coordinates and an eyes' vector
    eye1 = np.array(eyes[0])
    eye2 = np.array(eyes[1])
    eyes_vec = eye2 - eye1
    #
    #create pseudo-image with eyes' coordinates
    eyes_image = np.zeros((image.shape[0], image.shape[1]))
    gkern1d = signal.gaussian(10, 3).reshape(10, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    eyes_image[eye1[1]-5:eye1[1]+5, eye1[0]-5:eye1[0]+5] = gkern2d
    eyes_image[eye2[1]-5:eye2[1]+5, eye2[0]-5:eye2[0]+5] = gkern2d
    #
    #create a reference vector 
    #pointing upwards, to avoid handling negative angles between the eyes' and the ref. vector
    horiz1 = np.array([1,10])
    horiz2 = np.array([1,1])
    horiz_vec = horiz2 - horiz1
    #
    #compute an angle between the eyes' vector and the reference vector
    dot_product = eyes_vec[0]*horiz_vec[0] + eyes_vec[1]*horiz_vec[1]
    alpha = arccos(dot_product / (np.sqrt(eyes_vec[0]**2+eyes_vec[1]**2)*np.sqrt(horiz_vec[0]**2+horiz_vec[1]**2)))
    #
    #rotate the original image and the pseudo-eyes image by this angle - 90° 
    #because the reference vector was pointing upwards
    print('Angle w.r.t. the horizontal axis is ', alpha*180/np.pi - 90)
    image_rot = ndi.rotate(image, alpha * 180/np.pi - 90, mode='constant')
    eyes_image_rot = ndi.rotate(eyes_image, alpha * 180/np.pi - 90, mode='constant')
    #
    #find new eyes' coordinates and the midpoint
    eyes_rot = peak_local_max(eyes_image_rot,
                                             num_peaks = 2)
    eye1_rot = np.array(eyes_rot[0])
    eye2_rot = np.array(eyes_rot[1])
    eyes_rot_centre = np.array((eye1_rot+eye2_rot)/2, dtype=int)
    #print(eyes_rot_centre)
    #
    #crop image to 200 (because the eyes in image 2 are located to close to the border, 
    #hence not possible to cut this image to 224x224)
    N = 200
    image_cropped = np.zeros((N,N,3))
    image_cropped = image_rot[int(eyes_rot_centre[0]-N/2):int(eyes_rot_centre[0]+N/2),
                              int(eyes_rot_centre[1]-N/2):int(eyes_rot_centre[1]+N/2),:]
    #print(image_cropped.shape)
    #plt.imshow(image_cropped)
    #plt.show()
    return image_cropped
```


```python
transformed_imgs = []
for i in faces:
    img = faces[i]
    eye = eyes[i]
    transformed = transform_face(img, eye)
    transformed_imgs.append(transformed)
visualize_RGB(transformed_imgs)
```

    Angle w.r.t. the horizontal axis is  -9.462322208025626
    Angle w.r.t. the horizontal axis is  -6.170175095029606
    Angle w.r.t. the horizontal axis is  -6.2540327439164685
    Angle w.r.t. the horizontal axis is  5.273895957351769
    Angle w.r.t. the horizontal axis is  -8.130102354155994
    Angle w.r.t. the horizontal axis is  -7.523820438638609
    (200, 200, 3)
    (200, 200, 3)
    (200, 200, 3)
    (200, 200, 3)
    (200, 200, 3)
    (200, 200, 3)



![png](output_files/output_29_1.png)



```python

```
