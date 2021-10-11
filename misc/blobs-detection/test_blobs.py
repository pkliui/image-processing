import unittest

import numpy as np
from scipy.ndimage import gaussian_filter
import shutil, tempfile
from ddt import ddt

from blobs import Blobs

@ddt
class TestBlobsClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        #create an instance of the Blobs class
        self.blobs = Blobs()
        #
        #create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        #
        # create a test image with blobs
        test_image = np.zeros((40, 40))
        test_image[20:25, 20:25] = 1
        test_image[31:36, 18:23] = 1
        test_image[10, 10] = 10
        test_image[10, 20] = 10
        self.test_image = gaussian_filter(test_image, sigma=0.5 * 5/np.sqrt(2))
        #
        # test_image's maxima
        test_image_max = np.zeros((40, 40))
        test_image_max[22, 22] = 1
        test_image_max[33, 20] = 1
        test_image_max[10, 10] = 1
        test_image_max[10, 20] = 1
        self.test_image_max = test_image_max
        #
        # scale-space version of test_image
        self.test_image_3d = np.zeros((40, 40, 3))
        self.test_image_3d[:, :, 1] = self.test_image
        #
        # scale-space version of test_image's maxima
        self.test_image_3d_max = np.zeros((40, 40, 3))
        self.test_image_3d_max[:, :, 1] = self.test_image_max

    def tearDown(self):
        #remove temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_empty_arguments(self):
        """
        test the input arguments are existing and are all None
        :return:
        """
        for var in ["pathtodata", "filetoread", "image", "image_log", "blobs_maxima", "blobs_boxes"]:
            self.assertIn(var, self.blobs.__dict__)
            self.assertEqual(self.blobs.__dict__[var], None)

    def test_read_data(self):
        """
        test missing positional arguments
        test reading some non-existing data
        """
        with self.assertRaises(TypeError):
            self.blobs.read_data()
        with self.assertRaises(ValueError):
            bs = Blobs(pathtodata="Nonexisting_path", filetoread="Nonexisting_file")

    def test_threshold_image(self):
        """
        test thresholding a 2d array
        """
        image = np.array([[0.0, 1.0, 2.0], [2.3, 8.5, 5.5], [6.0, 7.0, 8.0]])
        threshold = 0.3
        # hence all pixels  below 0.3 * 8.0 = 2.4 should be set to 0, the rest to 1
        image_after_thres_expected = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        #
        self.blobs.image = image
        image_after_thres = self.blobs.threshold_image(input_image=self.blobs.image, thres=threshold)
        print(image_after_thres)
        #
        self.assertTrue(np.array_equal(image_after_thres_expected, image_after_thres))

    def test_get_local_maxima(self):
        """
        test getting local maxima
        """
        test_image_thres = self.blobs.threshold_image(input_image=self.test_image)
        test_image_max = self.blobs.get_local_maxima(test_image_thres * self.test_image)
        self.assertTrue(np.array_equal(test_image_max, self.test_image_max))

    def test_get_local_maxima_3d(self):
        """
        test getting local maxima in 3d
        """
        test_image_3d_thres = self.blobs.threshold_image(input_image=self.test_image_3d)
        test_image_3d_max = self.blobs.get_local_maxima_3d(test_image_3d_thres * self.test_image_3d)
        self.assertTrue(np.array_equal(test_image_3d_max, self.test_image_3d_max))

    def test_get_blobs_boxes(self):
        """
        test getting blobs' boxes (single scale & box size)
        """
        # set the size  of the box to draw and initialise a list to keep the boxes
        linear_size = 5
        self.blobs.blobs_boxes = []
        #
        # threshold the image and get its local maxima
        test_image_thres = self.blobs.threshold_image(input_image=self.test_image)
        test_image_max = self.blobs.get_local_maxima(test_image_thres * self.test_image)
        max_coords = np.array(np.where(test_image_max == 1))
        #print("max_coords", max_coords)
        #
        # draw boxes of the chosen size
        self.blobs.get_blobs_boxes(local_maxima_mask=test_image_max[:, :], linear_size=linear_size)
        #
        for ii in range(0, 4):
            # check the box height and width for all four blobs
            self.assertEqual(linear_size, self.blobs.blobs_boxes[ii].get_height())
            self.assertEqual(linear_size, self.blobs.blobs_boxes[ii].get_width())
            # get the low left corner of the boxes
            self.assertEqual((max_coords[1, ii] - linear_size // 2 - 0.5, max_coords[0, ii] - linear_size // 2 - 0.5),
                             self.blobs.blobs_boxes[ii].get_xy())

    def test_detect_blobs_multiscale(self):
        """
        test detecting blobs
        """
        diameter_min = 5
        diameter_max = 7
        self.blobs.image = self.test_image
        self.blobs.detect_blobs_multiscale(diameter_min=diameter_min, diameter_max=diameter_max, log_thres=0.3)
        self.blobs.plot_image(image_mode="raw", boxes=True, roi_x0=0, roi_y0=0)
        #
        #expected height and width
        blobs_boxes_height = [5, 5, 7, 7]
        blobs_boxes_width = [5, 5, 7, 7]
        # expected lower left corner coordinates
        max_coords = np.array(np.where(self.blobs.blobs_maxima == 1))
        for ii in range(len(self.blobs.blobs_boxes)):
            #
            # check the box height and width for all four blobs are as expected
            self.assertEqual(blobs_boxes_height[ii], self.blobs.blobs_boxes[ii].get_height())
            self.assertEqual(blobs_boxes_width[ii], self.blobs.blobs_boxes[ii].get_width())
            self.assertEqual((max_coords[1, ii] - blobs_boxes_height[ii] // 2 - 0.5, max_coords[0, ii] - blobs_boxes_width[ii] // 2 - 0.5), self.blobs.blobs_boxes[ii].get_xy())
