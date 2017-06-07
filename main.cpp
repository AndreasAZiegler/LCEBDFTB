#include <chrono>
#include "barcode_localization.h"

#include <mgl2/mgl.h>

// Barcode standard, ID, contour
using Barcode = std::tuple<std::string, std::string, std::vector<cv::Point>>;

int main(int argc, char** argv) {
  // Parameters
  int minLineLength = 30;
  int maxLengthToLineLengthRatio = 8;
  int minLengthToLineLengthRatio = 2;
  int support_candidates_threshold = 7;
  int delta = 125;
  int inSegmentXDistance = 300;
  int inSegmentYDistance = 100;

  if(argc != 2) {
    std::cout << "Usage: display_image ImageToLoadAndDisplay" << std::endl;
    return(-1);
  }

  cv::Mat image_color;
  cv::Mat image_greyscale;
  image_color = cv::imread(argv[1], cv::IMREAD_COLOR);
  image_greyscale = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  //cv::resize(image, image, cv::Size(1080, 960));
  //cv::resize(image, image, cv::Size(), 0.4, 0.4);
  //cv::resize(image, image, cv::Size(), 1.0, 1.0);

  if(!image_greyscale.data) {
    std::cout << "No image data" << std::endl;
    return(-1);
  }

  // Create LSDDetector
  cv::line_descriptor::LSDDetector LSD;

  // Create keylines vector
  std::vector<cv::line_descriptor::KeyLine> keylines;
  // Detect lines with the LSD
  auto start = std::chrono::steady_clock::now();
  auto total_start = start;
  LSD.detect(image_greyscale, keylines, 2, 1);
  auto end = std::chrono::steady_clock::now();
  std::cout << "LSD: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	// Draw the detected lines on the image
	cv::Mat image_lines;
	image_color.copyTo(image_lines);

	std::cout << "Number of lines detected: " << keylines.size() << std::endl;

	start = std::chrono::steady_clock::now();
	std::vector<std::vector<cv::Point>> contours_lineSegments = getLineSegmentsContours(keylines, image_lines, minLineLength);
	end = std::chrono::steady_clock::now();
	std::cout << "Creating bounding boxes: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	cv::drawContours(image_lines, contours_lineSegments, -1, cv::Scalar(0, 0, 255));

	std::cout << "Number of bounding boxes: " << contours_lineSegments.size() << std::endl;

	cv::imwrite("debug-line-segments.jpg", image_lines);
	/*
	cv::namedWindow("Image with line segments", cv::WINDOW_NORMAL);
	cv::resizeWindow("Image with line segments", 600, 600);
	cv::imshow("Image with line segments", image_lines);
	*/

	// Find for every bounding box the containing segments
	std::vector<std::vector<std::shared_ptr<cv::line_descriptor::KeyLine>>> keylinesInContours(contours_lineSegments.size());
	int contours_lineSegments_size = contours_lineSegments.size();

	start = std::chrono::steady_clock::now();
	findContainingSegments(keylinesInContours, keylines, contours_lineSegments, contours_lineSegments_size);
	end = std::chrono::steady_clock::now();
	std::cout << "Find segments in bounding boxes: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	std::vector<std::vector<int>> support_scores(keylinesInContours.size());

	// Calculate support score of every segment for every bounding box
	int keylinesInContours_size = keylinesInContours.size();
	start = std::chrono::steady_clock::now();
	calculateSupportScores(keylinesInContours, support_scores, keylinesInContours_size);
	end = std::chrono::steady_clock::now();
	std::cout << "Calculate support scores: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	// Select s_cand
	start = std::chrono::steady_clock::now();

	std::vector<int> support_candidates(keylinesInContours_size);
	std::vector<int> support_candidates_pos(keylinesInContours_size);
	cv::Mat image_candidates;
	image_color.copyTo(image_candidates);

	selectSCand(support_scores,
							support_candidates,
							support_candidates_pos,
							keylinesInContours,
							keylinesInContours_size,
							image_candidates,
							support_candidates_threshold);
	end = std::chrono::steady_clock::now();
	std::cout << "Select s_cand: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	// Create vectors of intensities
	start = std::chrono::steady_clock::now();

	std::vector<bool> deletedContours(keylinesInContours_size);
	for(bool dc : deletedContours) {
		dc = false;
	}

	std::vector<std::vector<cv::Point>> perpendicularLineStartEndPoints(keylinesInContours_size, std::vector<cv::Point>(2));
	std::vector<std::vector<std::vector<uchar>>> intensities(keylinesInContours_size, std::vector<std::vector<uchar>>(5));
	std::vector<std::vector<int>> startStopIntensitiesPosition(keylinesInContours_size, std::vector<int>(2));

	int intensities_size = intensities.size();
	int image_cols = image_greyscale.cols;
	int image_rows = image_greyscale.rows;

	createVectorsOfIntensities(support_candidates,
														 support_candidates_pos,
														 keylinesInContours,
														 startStopIntensitiesPosition,
														 perpendicularLineStartEndPoints,
														 intensities,
														 image_greyscale,
														 image_cols,
														 image_rows,
														 intensities_size,
														 support_candidates_threshold,
														 deletedContours);

	end = std::chrono::steady_clock::now();
	std::cout << "Compute intensities: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	// Compute phis
	start = std::chrono::steady_clock::now();
	std::vector<std::vector<std::vector<int>>> phis(keylinesInContours_size, std::vector<std::vector<int>>(6));
	std::vector<int> start_barcode_pos(keylinesInContours_size);
	std::vector<int> end_barcode_pos(keylinesInContours_size);

	computePhis(delta,
							intensities,
							intensities_size,
							phis,
							startStopIntensitiesPosition,
							start_barcode_pos,
							end_barcode_pos,
							deletedContours);
	end = std::chrono::steady_clock::now();
	std::cout << "Compute phis: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;


	// Select a good line segment example
	int index = 0;
	int nmb = 0;
	bool finish = false;
	for(unsigned int i = 0; (i < intensities.size()) && (!finish); i++) {
		if(0 < intensities[i][3].size()) {
			nmb++;
			//if(250 < nmb) {
			if(290 < nmb) {
				finish = true;
				index = i;
			}
		}
	}
	index = 0;
	std::cout << "index = " << index << ", size() = " << intensities[index][2].size() << std::endl;

	// Calculate bounding boxes
	start = std::chrono::steady_clock::now();
	std::vector<std::vector<cv::Point>> contours_barcodes(keylinesInContours_size, std::vector<cv::Point>(4));
	calculateBoundingBoxes(keylinesInContours_size,
												 start_barcode_pos,
												 end_barcode_pos,
												 keylines,
												 contours_barcodes,
												 perpendicularLineStartEndPoints,
												 image_candidates,
												 deletedContours,
												 index,
												 maxLengthToLineLengthRatio,
												 minLengthToLineLengthRatio);

	end = std::chrono::steady_clock::now();
	std::cout << "Calculated bounding boxes: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	// Filtering bounding boxes
	start = std::chrono::steady_clock::now();

	filterContours(keylinesInContours_size,
								 deletedContours,
								 start_barcode_pos,
								 end_barcode_pos,
								 keylines,
								 support_scores,
								 contours_barcodes,
								 inSegmentXDistance,
								 inSegmentYDistance);

	cv::drawContours(image_candidates, contours_barcodes, -1, cv::Scalar(255, 0, 0), 1);

	for(int i = 0; i < keylinesInContours_size; i++) {
		if(false == deletedContours[i]) {
			cv::putText(image_candidates, std::to_string(i), contours_barcodes[i][0], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));
		}
	}

	end = std::chrono::steady_clock::now();
	std::cout << "Filtering bounding boxes: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	// barcode decoding with ZBar
	start = std::chrono::steady_clock::now();

	cv::Mat image_barcodes;
	image_color.copyTo(image_barcodes);

	decodeBarcode(keylinesInContours_size, deletedContours, contours_barcodes, image_greyscale, image_barcodes);

  end = std::chrono::steady_clock::now();
  std::cout << "Barcode decoding: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	std::cout << "Total time: " << std::chrono::duration <double, std::milli> (end - total_start).count() << " ms" << std::endl;


	// Plot intensity and phi of selected line segment
	std::vector<std::vector<double>> intensity(5, std::vector<double>(intensities[index][2].size()));
	std::vector<std::vector<double>> phi(6, std::vector<double>(phis[index][2].size()));
	for(int j = 0; j < 5; j++) {
		for(unsigned int i = 0; i < intensities[index][2].size(); i++) {
			intensity[j][i] = static_cast<double>(intensities[index][j][i]);
			phi[j][i] = static_cast<double>(phis[index][j][i]);
		}
	}
	for(unsigned int i = 0; i < intensities[index][2].size(); i++) {
		phi[5][i] = static_cast<double>(phis[index][5][i]);
	}

	// Draw line segment and perpendicular line
	std::shared_ptr<cv::line_descriptor::KeyLine> kl = keylinesInContours[index][support_candidates_pos[index]];

	cv::Point start_point_m30(0, kl->pt.y + kl->pt.x*std::tan(M_PI_2 - kl->angle) - 16);
	cv::Point start_point_m15(0, kl->pt.y + kl->pt.x*std::tan(M_PI_2 - kl->angle) - 8);
	cv::Point start_point_0(0, kl->pt.y + kl->pt.x*std::tan(M_PI_2 - kl->angle));
	cv::Point start_point_15(0, kl->pt.y + kl->pt.x*std::tan(M_PI_2 - kl->angle) + 8);
	cv::Point start_point_30(0, kl->pt.y + kl->pt.x*std::tan(M_PI_2 - kl->angle) + 16);

	cv::Point end_point_m30(image_greyscale.cols, kl->pt.y - (image_greyscale.cols - kl->pt.x)*std::tan(M_PI_2 - kl->angle) - 16);
	cv::Point end_point_m15(image_greyscale.cols, kl->pt.y - (image_greyscale.cols - kl->pt.x)*std::tan(M_PI_2 - kl->angle) - 8);
	cv::Point end_point_0(image_greyscale.cols, kl->pt.y - (image_greyscale.cols - kl->pt.x)*std::tan(M_PI_2 - kl->angle));
	cv::Point end_point_15(image_greyscale.cols, kl->pt.y - (image_greyscale.cols - kl->pt.x)*std::tan(M_PI_2 - kl->angle) + 8);
	cv::Point end_point_30(image_greyscale.cols, kl->pt.y - (image_greyscale.cols - kl->pt.x)*std::tan(M_PI_2 - kl->angle) + 16);

	cv::line(image_candidates, start_point_m30, end_point_m30, cv::Scalar(0, 255, 0), 2);
	cv::line(image_candidates, start_point_m15, end_point_m15, cv::Scalar(0, 255, 0), 2);
	cv::line(image_candidates, start_point_0, end_point_0, cv::Scalar(0, 255, 0), 2);
	cv::line(image_candidates, start_point_15, end_point_15, cv::Scalar(0, 255, 0), 2);
	cv::line(image_candidates, start_point_30, end_point_30, cv::Scalar(0, 255, 0), 2);

	cv::imwrite("debug-candidate-segments.jpg", image_candidates);

	/*
	cv::namedWindow("Image with candidate segments", cv::WINDOW_NORMAL);
	//cv::resizeWindow("Image with candidate segments", 1080, 960);
	cv::imshow("Image with candidate segments", image_candidates);
	*/

	/*
	cv::imshow("Plot intensity", plot_result_intensity);
	cv::imshow("Plot phi", plot_result_phi);
	*/

	//cv::waitKey(0);

	/*
	mglGraph gr_int;
	mglGraph gr_phi;
	std::vector<mglData> mgl_phi(6);
	std::vector<mglData> mgl_intensities(5);

	//mgls_prepare1d(&y);
	for(int i = 0; i < 5; i++) {
		mgl_phi[i].Set(phi[i].data(), phi[i].size());
		mgl_intensities[i].Set(intensity[i].data(), intensity[i].size()-1);


		gr_phi.SubPlot(1, 6, i);
		std::string str = "Phi " + std::to_string(i);
		gr_phi.Title(str.c_str());
		gr_phi.SetOrigin(0,0,0);
		gr_phi.SetRanges(0, phis[index][2].size(), -2900, 2900);
		gr_phi.Axis();
		gr_phi.Grid();
		gr_phi.Plot(mgl_phi[i]);

		gr_int.SubPlot(1, 5, i);
		str = "Intensities" + std::to_string(i);
		gr_int.Title(str.c_str());
		gr_int.SetOrigin(0,0,0);
		gr_int.SetRanges(0, phis[index][2].size(), 0, 300);
		//gr.SetRanges(0, 3320, -5000, 5000);
		gr_int.Axis();
		gr_int.Grid();
		gr_int.Plot(mgl_intensities[i]);
	}

	mgl_phi[5].Set(phi[5].data(), phi[5].size());
	gr_phi.SubPlot(1, 6, 5);
	std::string str = "Phi average";
	gr_phi.Title(str.c_str());
	gr_phi.SetOrigin(0,0,0);
	gr_phi.SetRanges(0, phis[index][2].size(), -2900, 2900);
	gr_phi.Axis();
	gr_phi.Grid();
	gr_phi.Plot(mgl_phi[5]);


	//gr.Box();
	gr_int.WriteFrame("plot-intensities.png");
	gr_phi.WriteFrame("plot-phi.png");
	*/

	return(0);
}
