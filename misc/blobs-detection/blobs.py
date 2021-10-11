import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from skimage import io
from scipy.ndimage import gaussian_laplace
from copy import copy


class Blobs:
    """
    This is a class to find blobs in 2d tif images by multi-scale Laplacian of Gaussian detector
    """
    def __init__(self, pathtodata=None, filetoread=None, image=None, image_log=None, blobs_maxima=None, blobs_boxes=None):
        """
        Initializes the Blobs class
        ---
        Parameters
        ---
        pathtodata: str
            Path to a directory with data
            Default is None
        filetoread: str
            Filename to read, with extension
            Default is None
        image: numpy 2D array, float
            Input image
            Default is None
        image_log: a numpy array, float
            Input image after the application of LoG detector
            Default is None
        blobs_maxima: a numpy array
            Binary image where pixel values = 1 are the inferred locations of blobs' maxima in the input image
            Default is None
        blobs_boxes: list
            A list of rectangular boxes around the detected blobs
            Default is None
        """
        self.pathtodata = pathtodata
        self.filetoread = filetoread
        self.image = image
        self.image_log = image_log
        self.blobs_maxima = blobs_maxima
        self.blobs_boxes = blobs_boxes
        #
        if self.image is None and (pathtodata and filetoread) is not None:
            self.read_data()

    def __repr__(self):
        return "2D Fourier-domain image"

    def read_data(self):
        """
        reads tiff image
        ---
        Sets the following class variables:
        ---
        self.image
        """
        pathtofile = os.path.join(self.pathtodata, self.filetoread)
        if os.path.exists(pathtofile):
            #
            # read data from a tiff file and make sure to save it as a signed integer (important for LoG detector)
            self.image = io.imread(pathtofile, as_gray=True).astype(np.int32)
        else:
            raise ValueError("File path ", pathtofile, " does not exist! Please enter a valid path.")

    def detect_blobs_multiscale(self, diameter_min=None, diameter_max=None, log_thres=0.5):
        """
        Multi-scale Laplacian of Gaussian (LoG) blobs detector.
        The range of blobs' diameters the filter is most sensitive to is set by diameter_min and diameter_max.
        The step size between intermediate diameters is 1 unless diameter_max=diameter_min.
        ---
        Parameters
        ---
        diameter_min: int
            Min blobs' diameter we would like to detect
        diameter_max: int, >=diameter_min
            Max blobs' diameter we would like to detect.
        log_thres: 0 <= float < 1
            The lower bound for the scale space maxima. Reduce to detect blobs having lower intensities
            Default: 0.5
        ---
        Sets the following class variables:
        ---
        self.image_log, self.blobs_maxima, self.blobs_boxes

        """
        assert diameter_min <= diameter_max, "Max diameter value cannot be smaller than min diameter value"

        # initialise
        self.image_log = []
        self.blobs_boxes = []

        if diameter_min and diameter_max and log_thres is not None:
            #
            # set the range of diameters of interest to be used in a LoG scale space
            diameters = range(diameter_min - 1, diameter_max + 2, 1)
            #
            # loop through the scale space
            for scale_idx, diameter in enumerate(diameters):
                #
                # compute the LoG filter and scale-normalise it
                sigma = 0.5 * diameter / np.sqrt(2)
                if scale_idx == 0:
                    self.image_log = - sigma**2 * gaussian_laplace(self.image, sigma=sigma)
                else:
                    self.image_log = np.dstack((self.image_log, - sigma**2 * gaussian_laplace(self.image, sigma=sigma)))
            # save the square of the responses
            self.image_log = np.array(self.image_log)**2
            #
            # get a rough estimate of where the blobs' max are located
            image_log_thresholded = self.threshold_image(input_image=self.image_log, thres=log_thres)
            #
            # refine the location of blobs' maxima
            self.blobs_maxima = self.get_local_maxima_3d(image_log_thresholded * self.image_log)
            #
            # get the bounding boxes
            for scale_idx, diameter in enumerate(diameters):
                self.get_blobs_boxes(local_maxima_mask=self.blobs_maxima[:, :, scale_idx], linear_size=diameter)
        else:
            raise ValueError("Input values cannot be None. \
                            Current values: diamter_min={0},  diamter_max={1}, log_thres={2}"\
                             .format(diameter_min, diameter_max, log_thres))

    def threshold_image(self, input_image=None, thres=0.5):
        """
        Threshold an input image at provided threshold
        ---
        Parameters
        ---
        input_image: a numpy array, float
            input image to threshold
        thres: float between 0 and 1
            threshold value (in a fraction from the maximum input_image value)
            Default: 0.5
        ---
        Return
        ---
        image_thresholded: a numpy array, float
            Binary image with pixel values = 1 above the threshold and 0 below.
        """
        if input_image is not None:
            #
            # get rows and cols where pixel values are above the threshold
            #rows, cols = np.array(np.where(input_image > thres * input_image.max()))
            coords = list(np.where(input_image > thres * input_image.max()))
            #
            image_thresholded = np.zeros(input_image.shape)
            image_thresholded[tuple(coords)] = 1
        else:
            raise ValueError("Input image cannot be None")
        return image_thresholded

    def get_local_maxima(self, input_image):
        """
        Looks for local maxima in a 2D image by comparing pixel values with the values of their immediate neighbours.
        Assumes there are no local maxima at the edges
        Returns a binary image of local maxima in the input image excluding its edges
        ---
        Parameters
        ---
        input_image: 2d numpy array, float
            Input image whose local maxima are being looked for
            Minimum size is 3x3 image
        ---
        Return
        ---
        local_max_mask: 2d numpy array, int
            Binary image with pixel values = 1 at locations of blobs' maxima and 0 elsewhere.
        """
        if input_image.shape[0] >= 3 and input_image.shape[1] >= 3:
            # first find local max in each column
            # multiply by 1 to get a binary mask image from  boolean array
            local_max_mask_cols = 1 * (input_image[1:-1, :] > input_image[:-2, :]) & (input_image[1:-1, :] > input_image[2:, :])
            #
            # now search for local max in each row
            local_max_mask_rows = 1 * (input_image[:, 1:-1] > input_image[:, :-2]) & (input_image[:, 1:-1] > input_image[:, 2:])

            # add two rows of zeros to match the original size
            local_max_mask_cols = np.pad(local_max_mask_cols, ((1, 1), (0, 0)))
            local_max_mask_rows = np.pad(local_max_mask_rows, ((0, 0), (1, 1)))
            local_maxima_mask = local_max_mask_cols * local_max_mask_rows
        else:
            raise ValueError("Linear input image size cannot be less than 3 pixels!")
        return local_maxima_mask

    def get_local_maxima_3d(self, input_image):
        """
        Looks for local maxima in a 3D image by comparing pixel values with the values of their immediate neighbours.
        Assumes there are no local maxima at the edges
        Returns a binary image of local maxima in the input image excluding its edges.
        ---
        Parameters
        ---
        input_image: 3d numpy array, float
            Input image whose local maxima are being looked for
        ---
        Return
        ---
        local_max_mask: 3d numpy array, int
            Binary image with pixel values = 1 at locations of blobs' maxima and 0 elsewhere.
        """
        #
        # first look for local max in 2D channels of the input_image
        local_max_mask_2d = np.zeros(input_image.shape)
        for ii in range(input_image.shape[2]):
            local_max_mask_2d[:, :, ii] = self.get_local_maxima(input_image[:, :, ii])
        #
        # now do the same for each channel
        local_max_mask_cnls = 1 * (input_image[:, :, 1:-1] > input_image[:, :, :-2]) & (input_image[:, :, 1:-1] > input_image[:, :, 2:])
        #
        # match to the original size
        local_max_mask_cols = np.pad(local_max_mask_2d, ((1, 1), (1, 1), (0, 0)))
        local_max_mask_cnls = np.pad(local_max_mask_cnls, ((0, 0), (0, 0), (1, 1)))
        #
        # locate maxima in 3D
        local_maxima_mask = local_max_mask_2d * local_max_mask_cnls

        return local_maxima_mask

    def get_blobs_boxes(self, local_maxima_mask=None, linear_size=None):
        """
        Return the list of rectangular boxes whose linear size is set by diameter and locations by local_maxima_mask
        ---
        Parameters
        ---
        local_maxima_mask: 2d numpy array, int
            Binary image with pixel values = 1 and 0.
        linear_size: int
            Linear size of the bounding boxes
        """
        # get the locations of blobs
        coords = np.array(np.where(local_maxima_mask == 1))
        assert len(coords[0]) == len(coords[1]), "Rows and columns lengths are not the same"
        #
        # get their bounding boxes
        for ii in range(len(coords[0])):
            #print("linear size", linear_size)
            rect = patches.Rectangle((coords[1][ii] - linear_size // 2 - 0.5, coords[0][ii] - linear_size // 2 - 0.5),
                                     linear_size, linear_size, linewidth=1, edgecolor='r', facecolor='none')
            self.blobs_boxes.append(rect)

    def plot_image(self, image_mode="raw", boxes=False, roi_x0=0, roi_y0=0, roi_x=None, roi_y=None):
        """
        Plots raw or LoG-filtered images
        There is an option to display a portion of the image by setting the region of interest (ROI)
        ---
        Parameters
        ---
        image_mode : str, optional
            What kind of image to plot.
            The input must be one of the following: "raw", "log" for raw and LoG-filtered image
            Default is "raw"
        boxes: bool, optional
            If True, bounding  boxes around the blobs' maxima are drawn
            (only if detect_blobs_multiscale was called and the blobs_maxima is not empty or None)
            Default: False
        roi_x0: int, optional
            Lower left horizontal coordinate of the ROI
        roi_y0: int, optional
            Lower left vertical coordinate of the ROI
        roi_x: int, optional
            Horizontal width of the ROI
        roi_y: int, optional
            Vertical width of the ROI
        """
        fig, ax = plt.subplots(1, figsize=(10, 10))
        #
        # select what kind of image to plot
        if image_mode == "raw":
            if self.image is not None:
                image2plot = self.image
            else:
                raise ValueError('Read the image data first')
        elif image_mode == "log":
            if self.image_log is not None:
                image2plot = self.image_log
            else:
                raise ValueError('Find the blobs maxima first by calling detect_blobs method')
        #
        # plot bounding boxes
        if boxes:
            for box in self.blobs_boxes:
                #
                # this piece is needed to prevent add_patch from crashing
                # if called again without initialisation of the class
                box_copy = copy(box)
                box_copy.axes = None
                box_copy.figure = None
                box_copy.set_transform(ax.transData)
                # add bounding box to the plot
                ax.add_patch(box_copy)
                #
            ax.plot(np.array(np.where(self.blobs_maxima == 1))[1, :],
                    np.array(np.where(self.blobs_maxima == 1))[0, :],
                    'r.')
        #
        # plot
        if roi_x is None and roi_y is None:
            roi_y, roi_x = image2plot.shape[0], image2plot.shape[1]
        plt.axis([roi_x0,
                  roi_x0 + roi_x,
                  roi_y0,
                  roi_y0 + roi_y])
        plt.gca().invert_yaxis()
        plt.imshow(image2plot)
        plt.colorbar()
