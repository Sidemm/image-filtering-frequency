
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

Mat myFilter(Mat, Mat, int);
Mat hybrid_image_visualize(Mat);
Mat DFT_Spectrum(Mat);



enum border { Border_Replicate, Border_Reflect, Border_Constant };

Mat myFilter(Mat im, Mat filter, int borderType = Border_Constant)
{
	/*This function is intended to behave like the built in function filter2D()

	Mat outI;
	Mat patiti; //matrix for bordered image

	im.copyTo(outI);
	int brd;

	switch (borderType) { // conforming enum borders to copymakeborder method 
	case 0: brd = 1; break;
	case 1: brd = 2; break;
	case 2: brd = 0; break;
	}

	copyMakeBorder(im, patiti, (filter.rows - 1) / 2, (filter.rows - 1) / 2, (filter.cols - 1) / 2, (filter.cols - 1) / 2, brd, 0);

	for (int m = 0; m < im.rows; m++) { // traversing the original image 
		for (int n = 0; n < im.cols; n++) {
			Vec3d sum = 0;
			for (int i = 0; i < filter.rows; i++) { // traversing the filter 
				for (int j = 0; j < filter.cols; j++) {
					sum = sum + filter.at<double>(i, j)*patiti.at<Vec3d>(m + i, n + j); // adding up the multiplied values for one filter matrix interval
				}
				outI.at<Vec3d>(m, n) = sum; 
			}
		}
	}


	return outI;
}


Mat hybrid_image_visualize(Mat hybrid_image)
{
	//visualize a hybrid image by progressively downsampling the image and
	//concatenating all of the images together.		
	int scales = 5; //how many downsampled versions to create		
	double scale_factor = 0.5; //how much to downsample each time		
	int padding = 5; //how many pixels to pad.
	int original_height = hybrid_image.rows; // height of the image
	int num_colors = hybrid_image.channels(); //counting how many color channels the input has
	Mat output = hybrid_image;
	Mat cur_image = hybrid_image;

	for (int i = 2; i <= scales; i++)
	{
		//add padding
		hconcat(output, Mat::ones(original_height, padding, CV_8UC3), output);

		//dowsample image;
		resize(cur_image, cur_image, Size(0, 0), scale_factor, scale_factor, INTER_LINEAR);

		//pad the top and append to the output
		Mat tmp;
		vconcat(Mat::ones(original_height - cur_image.rows, cur_image.cols, CV_8UC3), cur_image, tmp);
		hconcat(output, tmp, output);
	}

	return output;
}

Mat DFT_Spectrum(Mat img)
{
	/*
	This function is intended to return the spectrum of an image in a displayable form. Displayable form
	means that once the complex DFT is calculated, the log magnitude needs to be determined from the real
	and imaginary parts. Furthermore the center of the resultant image needs to correspond to the origin of the spectrum.
	*/

	vector<Mat> im_channels(3);
	split(img, im_channels);
	img = im_channels[0];

	
	//pad the input image to optimal size using getOptimalDFTSize()

	Mat padded;
	int m = getOptimalDFTSize(img.rows);
	int n = getOptimalDFTSize(img.cols);
	copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, 0); // padded image with extra borders


	
	//Determine complex DFT of the image. 
	// The first dimension represents the real part and second dimesion represents the complex part of the DFT 

	Mat complex;
		
 	dft(padded, complex, DFT_COMPLEX_OUTPUT); 

	
	//compute the magnitude and switch to logarithmic scale
	
	vector<Mat> planes; // vector plane for keeping the real and imaginary parts
	split(complex, planes);
	Mat magI;
	magnitude(planes[0], planes[1], magI); // magnitude calculation

	log(Scalar::all(1) + magI, magI);

	
	/* For visualization purposes the quadrants of the spectrum are rearranged so that the
	origin (zero, zero) corresponds to the image center. To achieve this swap the top left
	quadrant with bottom right quadrant, and swap the top right quadrant with bottom left quadrant
	*/

	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	Mat quad1(magI, Rect(0, 0, magI.cols / 2, magI.rows / 2)); // top left 
	Mat quad2(magI, Rect(magI.cols / 2, 0, magI.cols / 2, magI.rows / 2)); // top right
	Mat quad3(magI, Rect(0, magI.rows / 2, magI.cols / 2, magI.rows / 2)); // bottom left
	Mat quad4(magI, Rect(magI.cols / 2, magI.rows / 2, magI.cols / 2, magI.rows / 2)); // bottom right

	Mat temp;
	quad4.copyTo(temp);
	quad1.copyTo(quad4); // swaps top left and bottom right
	temp.copyTo(quad1);

	quad3.copyTo(temp);
	quad2.copyTo(quad3); // swaps top right and bottom left
	temp.copyTo(quad2);

	// Transform the matrix with float values into a viewable image form (float between values 0 and 1).
	normalize(magI, magI, 0, 1, CV_MINMAX);
	return magI;
}

int main()
{
	//Read images
	Mat image1 = imread("../data/dog.bmp");
	if (!image1.data)                              // Check for invalid image
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	Mat image2 = imread("../data/cat.bmp");
	if (!image2.data)                              // Check for invalid image
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	image1.convertTo(image1, CV_64FC3);
	image2.convertTo(image2, CV_64FC3);

		//  FILTERING AND HYBRID IMAGE CONSTRUCTION  ////

	int cutoff_frequency = 6;
	/*This is the standard deviation, in pixels, of the
	Gaussian blur that will remove the high frequencies from one image and
	remove the low frequencies from another image (by subtracting a blurred
	version from the original version). You will want to tune this for every
	image pair to get the best results.*/

	Mat filter = getGaussianKernel(cutoff_frequency * 4 + 1, cutoff_frequency, CV_64F);
	filter = filter*filter.t();



	Mat low_freq_img = myFilter(image1, filter); // low pass filtering

	Mat high_freq_img = image2 - myFilter(image2, filter); // high pass filtering 

	Mat hybrid_image = low_freq_img + high_freq_img; 

	
	//added a scalar to high frequency image because it is centered around zero and is mostly black	
	high_freq_img = high_freq_img + Scalar(0.5, 0.5, 0.5) * 255;
	//Converted the resulting images type to the 8 bit unsigned integer matrix with 3 channels
	high_freq_img.convertTo(high_freq_img, CV_8UC3);
	low_freq_img.convertTo(low_freq_img, CV_8UC3);
	hybrid_image.convertTo(hybrid_image, CV_8UC3);

	Mat vis = hybrid_image_visualize(hybrid_image);

	imshow("Low frequencies", low_freq_img); waitKey(50);
	imshow("High frequencies", high_freq_img);	waitKey(50);
	imshow("Hybrid image", vis); waitKey(50);


	imwrite("low_frequencies.jpg", low_freq_img);
	imwrite("high_frequencies.jpg", high_freq_img);
	imwrite("hybrid_image.jpg", hybrid_image);
	imwrite("hybrid_image_scales.jpg", vis);



	//DFT_Spectrum() method

	Mat img1_DFT = DFT_Spectrum(image1);
	imshow("Image 1 DFT", img1_DFT); waitKey(50);
	imwrite("Image1_DFT.jpg", img1_DFT * 255);

	low_freq_img.convertTo(low_freq_img, CV_64FC3);
	Mat low_freq_DFT = DFT_Spectrum(low_freq_img);
	imshow("Low Frequencies DFT", low_freq_DFT); waitKey(50);
	imwrite("Low_Freq_DFT.jpg", low_freq_DFT * 255);

	Mat img2_DFT = DFT_Spectrum(image2);
	imshow("Image 2 DFT", img2_DFT); waitKey(50);
	imwrite("Image2_DFT.jpg", img2_DFT * 255);

	high_freq_img.convertTo(high_freq_img, CV_64FC3);
	Mat high_freq_DFT = DFT_Spectrum(high_freq_img);
	imshow("High Frequencies DFT", high_freq_DFT); waitKey(50);
	imwrite("High_Freq_DFT.jpg", high_freq_DFT * 255);
	waitKey(0);
}