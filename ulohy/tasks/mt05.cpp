#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <opencv2/core/core.hpp>        // basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // OpenCV image processing
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

// Support functions 
void checkDifferences( const cv::Mat test, const cv::Mat ref, std::string tag, bool save = false);


/*---------------------------------------------------------------------------
TASK 5
	!! Only add code to the reserved places.
	!! The resulting program must not print anything extra on any output (nothing more than the prepared program framework).
*/

/* TASK 5.0 - PSNR ... use solution from TASK 4
*/
double getPSNR(const Mat& I1, const Mat& I2)
{
	double psnr = 0.0;
	

	/* ***** Working area - begin ***** */
	Mat s1;
    absdiff(I1, I2, s1);
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);
    Scalar s = sum(s1);

    double mse = s.val[0] / (double)(I1.size().height * I1.size().width);
    if (mse > 0) {
        psnr = 10.0 * log10((255 * 255) / mse);
    }

	/* ***** Working area - end ***** */

 	return psnr; 
}


/*	TASK 5.1 - Median filter
	The median filter is a non-linear digital filtering technique, often used to remove noise from an image or signal.
	The main idea of the median filter is to run through the signal entry by entry, replacing each entry with the median of neighboring entries. The pattern of neighbors is called the "window", which slides, entry by entry, over the entire signal. 
*/

int medianFilter( const cv::Mat& src, cv::Mat& dst, int size = 3 )
{
	if( src.empty() || size < 3)
		return 1;

	// size - odd number >= 3
	int center = size/2;
	center = MAX(1,center);
	size = 2*center+1;
	
	// we simplify the calculation at the edge of the image by expanding the content by half the size of the filter
	cv::Mat srcBorder;
	copyMakeBorder( src, srcBorder, center, center, center, center, cv::BORDER_REPLICATE );

	// output image
	dst = cv::Mat( src.size(), src.type() );

	/*
		For each pixel of the output image
		1. use pixels in the neighbourhood according to the sice of the filter
		2. sort the values
		3. use the value at the middle position of the sorted list as the output value

		Allowed Mat attributes, methods and OpenCV functions for task solution are:
			Mat:: rows, cols, step(), size(), at<>(), zeros(), ones(), eye(), cvRound()
			any sorting function for an array, vector etc.
	*/

	/*  Working area - begin */
	int size2 = size * size;
    int balance[size2];

    for (int i = 0; i < dst.rows; ++i) {
        for (int j = 0; j < dst.cols; ++j) {

            for (int k = 0; k < size2; ++k) {
                int rowOffset = k / size;
                int colOffset = k % size;
                balance[k] = srcBorder.at<unsigned char>(i + rowOffset, j + colOffset);
            }

            std::nth_element(balance, balance + size2 / 2, balance + size2);
            dst.at<unsigned char>(i, j) = balance[size2 / 2];
        }
    }
    /*  Working area - end */

	return 0;
}



/*	TASK 5.1 - Gaussian "separable" filter 

*/

//  Funkce pro filtraci obrazu Gaussovým filtrem.
//     - size filtru v pixelech
//	   - sigma je rozptyl pro výpočet hodnot Gaussovy funkce
int gaussianFilter( const cv::Mat& src, cv::Mat& dst, int size = 3, double sigma = 1. )
{
	if( src.empty() || size < 3 || sigma <= 0. )
		return 1;

	// size - odd number >= 3
	int center = size/2;
	center = MAX(1,center);
	size = 2*center+1;

	// output image
	dst = cv::Mat( src.size(), src.type() );

	/*
		Create 1D Gaussian kernel
		compute values using Gaussian function
		normalize the values
		e.g. size=3, kernel values should be: []0.05448813464528447, 0.2441988775028802, 0.4026158836200937, 0.2441988775028802, 0.05448813464528447

		Allowed Mat attributes, methods and OpenCV functions for task solution are:
			Mat:: rows, cols, step(), size(), at<>(), zeros(), ones(), eye(), sum()
			sqrt, exp(), 
	*/

	cv::Mat gauss1D = cv::Mat::zeros( 1, size, CV_64FC1 );

	/* ***** Working area - begin ***** */
    double sum = 0.0;
    double sigma_squared = sigma * sigma;
    for (int i = 0; i < size; i++) {
        double x = i - center;
        double value = exp(-(x * x) / (2 * sigma_squared)) / (sqrt(2 * CV_PI) * sigma);
        gauss1D.at<double>(0, i) = value;
        sum += value;
    }

    gauss1D /= sum;
    /* ***** Working area - end ***** */


	/*
		2D Gaussian image filter
		use cv::filter2D function for convolution
	*/

	/* ***** Working area - begin ***** */
    cv::Mat temp;
    cv::filter2D(src, temp, -1, gauss1D, cv::Point(-1, -1));
    cv::transpose(gauss1D, gauss1D);
    cv::filter2D(temp, dst, -1, gauss1D, cv::Point(-1, -1));
    /* ***** Working area - end ***** */

	return 0;
}



/*--------------------------------------------------------------------------- */

/* 	Examples of input parameters
	mt05 image_path fiter_type size [sigma = 0.0]
	eg. ./mt-05 ../../data/lena-sp.png mf 3
	    ./mt-05 ../../data/lena-gn.png gf 5

	- filter type
	  mf - median filter, size of the filter
	  gf - Gaussian filter, size of the filter and standard deviation
*/

int main(int argc, char* argv[])
{
 	std::string img_path = "";
    std::string filter_type = "";
    int size = 3;
    double sigma = 1.;

	if (argc < 4)
	{
		cout << "Not enough parameters." << endl;
		cout << "Usage: mt05 image_path filter_type size [sigma = 0.0]" << endl;
		return -1;
	}

	// check input parameters
	if( argc > 1 ) img_path = std::string( argv[1] );
	if( argc > 2 ) filter_type = std::string( argv[2] );
	if( argc > 3 ) size  = atoi( argv[3] );
	if( argc > 4 ) sigma = atof( argv[4] );

	// load testing images
	cv::Mat rgb = cv::imread( img_path );

	// check testing images
	if( rgb.empty() ) {
		std::cout << "Failed to load image: " << img_path << std::endl;
		return -1;
	}

	cv::Mat gray;
	cv::cvtColor( rgb, gray, CV_BGR2GRAY );

	//---------------------------------------------------------------------------

	Mat test, ref;

	if (filter_type == "mf")
	{
		medianFilter( gray, test, size );
		cv::medianBlur( gray, ref, size );
		checkDifferences( test, ref, "mf", true );	// to store output images 
		std::cout << ", PSNR(mf-test), " << getPSNR( gray, test );
		std::cout << ", PSNR(mf-ref), " << getPSNR( gray, ref );
		cv::GaussianBlur( gray, ref, cv::Size(size,size), sigma, sigma );
		std::cout << ", PSNR(gf-ref), " << getPSNR( gray, ref ) << std::endl;
		std::cout << std::endl;
	}
	else if (filter_type == "gf")
	{
		gaussianFilter( gray, test, size, sigma );
		cv::GaussianBlur( gray, ref, cv::Size(size,size), sigma, sigma );
		checkDifferences( test, ref, "gf", true );	// to store output images 
		std::cout << ", PSNR(gf-test), " << getPSNR( gray, test );
		std::cout << ", PSNR(gf-ref), " << getPSNR( gray, ref );
		cv::medianBlur( gray, ref, size );
		std::cout << ", PSNR(mf-ref), " << getPSNR( gray, ref ) << std::endl;
		std::cout << std::endl;		
	}

    return 0;
}
//---------------------------------------------------------------------------



void checkDifferences( const cv::Mat test, const cv::Mat ref, std::string tag, bool save )
{
	double mav = 255., err = 255., nonzeros = 1000.;

	if( !test.empty() ) {
		cv::Mat diff;
		cv::absdiff( test, ref, diff );
		cv::minMaxLoc( diff, NULL, &mav );
		nonzeros = 1. * cv::countNonZero( diff ); // / (diff.rows*diff.cols);
		err = (nonzeros > 0 ? ( cv::sum(diff).val[0] / nonzeros ) : 0);
		nonzeros /= (diff.rows*diff.cols);

		if( save ) {
			diff *= 255;
			cv::imwrite( (tag+".0.ref.png").c_str(), ref );
			cv::imwrite( (tag+".1.test.png").c_str(), test );
			cv::imwrite( (tag+".2.diff.png").c_str(), diff );
		}
	}

	printf( "%s, avg, %.1f, perc, %2.2f, max, %.0f, ", tag.c_str(), err, 100.*nonzeros, mav );

	return;
}
 
