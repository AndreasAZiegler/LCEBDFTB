#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/plot.hpp>
#include <iostream>
#include <chrono>
#include <memory>

#include <mgl2/mgl.h>

int main(int argc, char** argv) {
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

  //cv::imshow("Original image", image);

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

  std::vector<cv::Point2f> lookup;
  std::vector<std::vector<cv::Point>> contours;

	// Draw the detected lines on the image
	cv::Mat image_lines;
	image_color.copyTo(image_lines);

	std::cout << "Number of lines detected: " << keylines.size() << std::endl;

	start = std::chrono::steady_clock::now();
	for(cv::line_descriptor::KeyLine kl : keylines) {
		cv::line(image_lines, kl.getStartPoint(), kl.getEndPoint(), cv::Scalar(255, 0, 0));
		//std::vector<cv::Point2f> point = {kl.getStartPoint(), kl.getEndPoint()};

		float linelength = kl.lineLength;
		float angle = kl.angle;
		float cos_angle = std::cos(angle);
		float sin_angle = std::sin(angle);
		float start_x = kl.startPointX;
		float start_y = kl.startPointY;
		float end_x = kl.endPointX;
		float end_y = kl.endPointY;

		std::vector<cv::Point> contour;
		contour.push_back(cv::Point2f(start_x - linelength*sin_angle, start_y - linelength*cos_angle));
		contour.push_back(cv::Point2f(start_x + linelength*sin_angle, start_y + linelength*cos_angle));
		contour.push_back(cv::Point2f(end_x + linelength*sin_angle, end_y + linelength*cos_angle));
		contour.push_back(cv::Point2f(end_x + linelength*sin_angle, end_y + linelength*cos_angle));
		contour.push_back(cv::Point2f(end_x - linelength*sin_angle, end_y - linelength*cos_angle));
		contour.push_back(cv::Point2f(end_x - linelength*sin_angle, end_y - linelength*cos_angle));
		contour.push_back(cv::Point2f(start_x - linelength*sin_angle, start_y - linelength*cos_angle));
		contours.push_back(contour);
		lookup.push_back(kl.pt);
	}
	end = std::chrono::steady_clock::now();
	std::cout << "Creating bounding boxes: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	cv::drawContours(image_lines, contours, -1, cv::Scalar(0, 0, 255));

	std::cout << "Number of bounding boxes: " << contours.size() << std::endl;

	cv::imwrite("debug-line-segments.jpg", image_lines);
	cv::namedWindow("Image with line segments", cv::WINDOW_NORMAL);
	cv::resizeWindow("Image with line segments", 600, 600);
	cv::imshow("Image with line segments", image_lines);

	std::vector<std::vector<bool>> lookup2d(contours.size(), std::vector<bool>(keylines.size()));

	/*
	for(std::vector<bool> v : lookup2d) {
		for(bool b : v) {
			b = false;
		}
	}
	*/

	// Find for every bounding box the containing segments
	std::vector<std::vector<std::shared_ptr<cv::line_descriptor::KeyLine>>> keylinesInContours(contours.size());

	start = std::chrono::steady_clock::now();
	int contours_size = contours.size();
	#pragma omp parallel for
	for(int i = 0; i < contours_size; i++) {
		register float px = keylines[i].pt.x;
		register float py = keylines[i].pt.y;
		register float ll_2 = keylines[i].lineLength * 2;

		int keylines_size = keylines.size();
		#pragma omp parallel for
		for(int j = 0; j < keylines_size; j++) {
			if(i == j) {
				keylinesInContours[i].push_back(std::make_shared<cv::line_descriptor::KeyLine>(keylines[j]));
			}
			else{
				register cv::line_descriptor::KeyLine kl_j = keylines[j];
				if(std::abs(kl_j.pt.x - px) < ll_2) {
					if(std::abs(kl_j.pt.y - py) < ll_2) {
						if((0 < cv::pointPolygonTest(contours[i], kl_j.getStartPoint(), false)) &&
							 (0 < cv::pointPolygonTest(contours[i], kl_j.getEndPoint(), false))) {
							keylinesInContours[i].push_back(std::make_shared<cv::line_descriptor::KeyLine>(keylines[j]));
						}
					}
				}
			}
		}
	}
	end = std::chrono::steady_clock::now();
	std::cout << "Find segments in bounding boxes: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	std::vector<std::vector<int>> support_scores(keylinesInContours.size());

	// Calculate support score of every segment for every bounding box
	int keylinesInContours_size = keylinesInContours.size();
	start = std::chrono::steady_clock::now();
	#pragma omp parallel for
	for(int i = 0; i < keylinesInContours_size; i++) {
		int keylinesInContours_i_size = keylinesInContours[i].size();
		support_scores[i] = std::vector<int>(keylinesInContours[i].size());
		#pragma omp parallel for
		for(int j = 0; j < keylinesInContours_i_size; ++j) {
			support_scores[i][j] = 0;
		}
	}

	std::vector<float> diff_angle_vec;
	std::vector<float> diff_length_vec;
	std::vector<float> diff_norm_pt_vec;
	register float diff_length;
	register float diff_angle;
	register float diff_norm_pt;
	#pragma omp parallel for
	for(int i = 0; i < keylinesInContours_size; i++) {
		int keylinesInContours_i_size = keylinesInContours[i].size();
		#pragma omp parallel for
		for(int j = 0; j < keylinesInContours_i_size; j++) {
			register std::shared_ptr<cv::line_descriptor::KeyLine> kl_j = keylinesInContours[i][j];

			for(int k = j+1; k < keylinesInContours_i_size; k++) {
				register std::shared_ptr<cv::line_descriptor::KeyLine> kl_k = keylinesInContours[i][k];

				diff_length = std::abs(kl_j->lineLength - kl_k->lineLength);
				//diff_length_vec.push_back(diff_length);

				if((diff_length) < 5.0) {
					diff_angle = std::abs(kl_j->angle - kl_k->angle);
					//diff_angle_vec.push_back(diff_angle);

					if((diff_angle) < 3.0) {
						diff_norm_pt = cv::norm(kl_j->pt - kl_k->pt);
						//diff_norm_pt_vec.push_back(diff_norm_pt);

						if((diff_norm_pt) < 15.0) {
							/*
							std::cout << "Under the thresholds" << std::endl;
							std::cout << "diff_angle = " << diff_angle << ", diff_length = " << diff_length << ", diff_norm_pt = " << diff_norm_pt << std::endl;
							*/
							support_scores[i][j] += 1;
							support_scores[i][k] += 1;
						}
					}
				}
			}
		}
	}
	end = std::chrono::steady_clock::now();
	std::cout << "Calculate support scores: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	/*
	float diff_angle_min = diff_angle_vec[std::distance(diff_angle_vec.begin(), std::min_element(diff_angle_vec.begin(), diff_angle_vec.end()))];
	float diff_length_min = diff_length_vec[std::distance(diff_length_vec.begin(), std::min_element(diff_length_vec.begin(), diff_length_vec.end()))];
	float diff_norm_pt_min = diff_norm_pt_vec[std::distance(diff_norm_pt_vec.begin(), std::min_element(diff_norm_pt_vec.begin(), diff_norm_pt_vec.end()))];
	std::cout << "min diff angle = " << diff_angle_min << ", min length = " << diff_length_min << ", min norm pt = " << diff_norm_pt_min << std::endl;
	float diff_angle_max = diff_angle_vec[std::distance(diff_angle_vec.begin(), std::max_element(diff_angle_vec.begin(), diff_angle_vec.end()))];
	float diff_length_max = diff_length_vec[std::distance(diff_length_vec.begin(), std::max_element(diff_length_vec.begin(), diff_length_vec.end()))];
	float diff_norm_pt_max = diff_norm_pt_vec[std::distance(diff_norm_pt_vec.begin(), std::max_element(diff_norm_pt_vec.begin(), diff_norm_pt_vec.end()))];
	std::cout << "max diff angle = " << diff_angle_max << ", max length = " << diff_length_max << ", max norm pt = " << diff_norm_pt_max << std::endl;
	*/

	// Select s_cand

	std::vector<int> support_candidates(keylinesInContours_size);
	std::vector<int> support_candidates_pos(keylinesInContours_size);

	cv::Mat image_candidates;
	image_color.copyTo(image_candidates);

	start = std::chrono::steady_clock::now();
	#pragma omp parallel for
	for(int i = 0; i < keylinesInContours_size; i++) {
		support_candidates_pos[i] = std::distance(support_scores[i].begin(), std::max_element(support_scores[i].begin(), support_scores[i].end()));
		//std::cout << "support_candidate_pos[" << i << "] = " << support_candidates_pos[i] << std::endl;
		support_candidates[i] = support_scores[i][std::distance(support_scores[i].begin(), std::max_element(support_scores[i].begin(), support_scores[i].end()))];
		//std::cout << "max_support_candidates[" << i << "] = " << support_candidates[i] << std::endl;
		//std::cout << "min_support_candidates[" << i << "] = " << support_scores[i][std::distance(support_scores[i].begin(), std::min_element(support_scores[i].begin(), support_scores[i].end()))] << std::endl;

		if(0 < support_candidates[i]) {
			std::shared_ptr<cv::line_descriptor::KeyLine> kl = keylinesInContours[i][support_candidates_pos[i]];
			cv::line(image_candidates, kl->getStartPoint(), kl->getEndPoint(), cv::Scalar(0, 0, 255));
		}
	}
	end = std::chrono::steady_clock::now();
	std::cout << "Select s_cand: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	cv::imwrite("debug-candidate-segments.jpg", image_candidates);

	// Create vectors of intensities
	start = std::chrono::steady_clock::now();
	std::vector<std::vector<cv::Point>> perpencidularLineStartEndPoints(keylinesInContours_size, std::vector<cv::Point>(2));
	std::vector<std::vector<std::vector<uchar>>> intensities(keylinesInContours_size, std::vector<std::vector<uchar>>(5));
	for(int i = 0; i < intensities.size(); i++) {
		if(0 < support_candidates[i]) {
			std::shared_ptr<cv::line_descriptor::KeyLine> kl = keylinesInContours[i][support_candidates_pos[i]];
			std::vector<cv::Point> pt1s;
			pt1s.push_back(cv::Point(0, kl->pt.y + kl->pt.x*std::tan(M_PI_2 - kl->angle) - 16));
			pt1s.push_back(cv::Point(0, kl->pt.y + kl->pt.x*std::tan(M_PI_2 - kl->angle) - 8));
			pt1s.push_back(cv::Point(0, kl->pt.y + kl->pt.x*std::tan(M_PI_2 - kl->angle)));
			pt1s.push_back(cv::Point(0, kl->pt.y + kl->pt.x*std::tan(M_PI_2 - kl->angle) + 8));
			pt1s.push_back(cv::Point(0, kl->pt.y + kl->pt.x*std::tan(M_PI_2 - kl->angle) + 16));

			std::vector<cv::Point> pt2s;
			pt2s.push_back(cv::Point(image_greyscale.cols, kl->pt.y - (image_greyscale.cols - kl->pt.x)*std::tan(M_PI_2 - kl->angle) - 16));
			pt2s.push_back(cv::Point(image_greyscale.cols, kl->pt.y - (image_greyscale.cols - kl->pt.x)*std::tan(M_PI_2 - kl->angle) - 8));
			pt2s.push_back(cv::Point(image_greyscale.cols, kl->pt.y - (image_greyscale.cols - kl->pt.x)*std::tan(M_PI_2 - kl->angle)));
			pt2s.push_back(cv::Point(image_greyscale.cols, kl->pt.y - (image_greyscale.cols - kl->pt.x)*std::tan(M_PI_2 - kl->angle) + 8));
			pt2s.push_back(cv::Point(image_greyscale.cols, kl->pt.y - (image_greyscale.cols - kl->pt.x)*std::tan(M_PI_2 - kl->angle) + 16));

			perpencidularLineStartEndPoints[i][0] = cv::Point(0, kl->pt.y + kl->pt.x*std::tan(M_PI_2 - kl->angle));
			perpencidularLineStartEndPoints[i][1] = cv::Point(image_greyscale.cols, kl->pt.y - (image_greyscale.cols - kl->pt.x)*std::tan(M_PI_2 - kl->angle));

			std::vector<cv::LineIterator> lineIterators;
			for(int j = 0; j < intensities[i].size(); j++) {
				lineIterators.push_back(cv::LineIterator(image_greyscale, pt1s[j], pt2s[j], 8, true));
				//cv::line(image_candidates, pt1s[j], pt2s[j], cv::Scalar(0, 255, 0), 1);
			}

			for(int j = 0; j < lineIterators.size(); j++) {
				intensities[i][j] = std::vector<uchar>(lineIterators[j].count);
				for(int k = 0; k < lineIterators[j].count; k++, ++lineIterators[j]) {
					intensities[i][j][k] = image_greyscale.at<uchar>(lineIterators[j].pos());
				}
			}
		}
	}
	end = std::chrono::steady_clock::now();
	std::cout << "Compute intensities: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	// Compute phis
	start = std::chrono::steady_clock::now();
	std::vector<std::vector<std::vector<int>>> phis(keylinesInContours_size, std::vector<std::vector<int>>(5));
	std::vector<std::vector<int>> start_barcode_pos(keylinesInContours_size, std::vector<int>(5));
	std::vector<std::vector<int>> end_barcode_pos(keylinesInContours_size, std::vector<int>(5));

	for(int i = 0; i < intensities.size(); i++) {
		for(int j = 0; j < intensities[i].size(); j++) {
			phis[i][j] = std::vector<int>(intensities[i][j].size());
			int max = 0;
			int min = 0;
			for(int k = 0; k < intensities[i][j].size(); k++) {
				int phi_1 = 0;
				int phi_2 = 0;
				for(int l = 0; l < 150; l++) {
					phi_1 += std::abs(intensities[i][j][k - l] - intensities[i][j][k - l - 1]);
					phi_2 += std::abs(intensities[i][j][k + l] - intensities[i][j][k + l + 1]);
				}
				phis[i][j][k] = phi_1 - phi_2;

				if(phis[i][j][k] > max) {
					max = phis[i][j][k];
					end_barcode_pos[i][j] = k;
				}
				if(phis[i][j][k] < min) {
					min = phis[i][j][k];
					start_barcode_pos[i][j] = k;
				}
			}
		}
	}
	end = std::chrono::steady_clock::now();
	std::cout << "Compute phis: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;


	std::cout << "Total time: " << std::chrono::duration <double, std::milli> (end - total_start).count() << " ms" << std::endl;

	// Select a good line segment example
	int index = 0;
	int nmb = 0;
	bool finish = false;
	for(int i = 0; (i < intensities.size()) && (!finish); i++) {
		if(0 < intensities[i][3].size()) {
			nmb++;
			//if(250 < nmb) {
			if(290 < nmb) {
				finish = true;
				index = i;
			}
		}
	}
	std::cout << "index = " << index << ", size() = " << intensities[index][2].size() << std::endl;

	// Calculate bounding boxes
	std::vector<std::vector<cv::Point>> contour(keylinesInContours_size, std::vector<cv::Point>(4));
	for(int i = 0; i < keylinesInContours_size; i++) {
		if(0 < support_candidates[i]) {
		//if(i == index) {
			int diff_1 = std::abs(start_barcode_pos[i][2] - start_barcode_pos[i][0]);
			int diff_2 = std::abs(start_barcode_pos[i][2] - start_barcode_pos[i][1]);
			int diff_3 = std::abs(start_barcode_pos[i][2] - start_barcode_pos[i][3]);
			int diff_4 = std::abs(start_barcode_pos[i][2] - start_barcode_pos[i][4]);
			std::cout << "diff_1 = " << diff_1 << ", diff_2 = " << diff_2 << ", diff_3 = " << diff_3 << ", diff_4 = " << diff_4 << std::endl;
			int angle = 0;
			int sign = 1;
			/*
			if((diff_2 < 4) &&
				 (diff_3 < 4)) {
			*/
			if(15 > diff_1 + diff_2 + diff_3 + diff_4) {
				std::cout << "Add one bounding box contour!" << std::endl;
				std::cout << "start_barcode_pos[" << i << "][2] = " << start_barcode_pos[i][2] << " , end_barcode_pos[" << i << "][2] = " << end_barcode_pos[i][2] << ", end_pos = " << phis[i][2].size() << ", angle = " << keylines[i].angle << std::endl;
				if(0 < keylines[i].angle) {
					angle = M_PI_2 - keylines[i].angle;
					sign = 1;
				} else if( 0 > keylines[i].angle) {
					angle = M_PI_2 + keylines[i].angle;
					sign = -1;
				}

				contour[i][0] = cv::Point(perpencidularLineStartEndPoints[i][0].x + std::cos(angle)*start_barcode_pos[i][2] - sign*keylines[i].lineLength*std::sin(angle)*0.5,
																	perpencidularLineStartEndPoints[i][0].y + std::sin(angle)*start_barcode_pos[i][2] - keylines[i].lineLength*std::cos(angle)*0.5);
				contour[i][1] = cv::Point(perpencidularLineStartEndPoints[i][0].x + std::cos(angle)*end_barcode_pos[i][2] - sign*keylines[i].lineLength*std::sin(angle)*0.5,
																	perpencidularLineStartEndPoints[i][0].y + std::sin(angle)*(end_barcode_pos[i][2]-2500) - keylines[i].lineLength*std::cos(angle)*0.5);
				contour[i][2] = cv::Point(perpencidularLineStartEndPoints[i][0].x + std::cos(angle)*end_barcode_pos[i][2] + sign*keylines[i].lineLength*std::sin(angle)*0.5,
																	perpencidularLineStartEndPoints[i][0].y + std::sin(angle)*(end_barcode_pos[i][2]-2500) + keylines[i].lineLength*std::cos(angle)*0.5);
				contour[i][3] = cv::Point(perpencidularLineStartEndPoints[i][0].x + std::cos(angle)*start_barcode_pos[i][2] + sign*keylines[i].lineLength*std::sin(angle)*0.5,
																	perpencidularLineStartEndPoints[i][0].y + std::sin(angle)*start_barcode_pos[i][2] + keylines[i].lineLength*std::cos(angle)*0.5);
				/*
				std::cout << "perpencidularLineStartEndPoints[" << i << "][0].x = " << perpencidularLineStartEndPoints[i][0].x << ", perpencidularLineStartEndPoints[" << i << "][0].y = " << perpencidularLineStartEndPoints[i][0].y << std::endl;
				std::cout << "std::cos(M_PI_2 + keylines[" << i << "].angle)*start_barcode_pos[" << i << "][2] = " << std::cos(M_PI_2 + keylines[i].angle)*start_barcode_pos[i][2] << std::endl;
				std::cout << "std::sin(M_PI_2 + keylines[" << i << "].angle)*start_barcode_pos[" << i << "][2] = " << std::sin(M_PI_2 + keylines[i].angle)*start_barcode_pos[i][2] << std::endl;
				*/
				std::cout << "keylines[" << i << "].lineLength*std::sin(M_PI_2 + keylines[" << i << "].angle)*0.5 = " << keylines[i].lineLength*std::sin(M_PI_2 + keylines[i].angle)*0.5 << ", keylines[" << i << "].lineLength*std::cos(M_PI_2 + keylines[" << i << "].angle)*0.5 = " << keylines[i].lineLength*std::cos(M_PI_2 + keylines[i].angle)*0.5 << std::endl;
				std::cout << "contour[" << i << "][0] = " << contour[i][0] << ", contour[" << i << "][1] = " << contour[i][1] <<
										 ", contour[" << i << "][2] = " << contour[i][2] << ", contour[" << i << "][3] = " << contour[i][3] << std::endl;
			}
		}
	}

	cv::drawContours(image_candidates, contour, -1, cv::Scalar(255, 0, 0), 1);

	// Plot intensity and phi of selected line segment
	std::vector<double> intensity(intensities[index][2].size());
	std::vector<double> phi(phis[index][2].size());
	for(int i = 0; i < intensities[index][2].size(); i++) {
		intensity[i] = static_cast<double>(intensities[index][2][i]);
		phi[i] = static_cast<double>(phis[index][2][i]);
	}

	cv::Mat data_intensity(intensities[index][2].size(), 1, CV_64F);
	cv::Mat data_phi(phis[index][2].size(), 1, CV_64F);
	//cv::randu(data, 0, 500);
	memcpy(data_intensity.data, intensity.data(), intensity.size()*sizeof(double));
	memcpy(data_phi.data, phi.data(), phi.size()*sizeof(double));

	cv::Mat plot_result_intensity;
	cv::Mat plot_result_phi;

	cv::Ptr<cv::plot::Plot2d> plot_intensity = cv::plot::createPlot2d(data_intensity);
	plot_intensity->setPlotBackgroundColor(cv::Scalar(50, 50, 50));
	plot_intensity->setPlotLineColor(cv::Scalar(50, 50, 255));
	plot_intensity->render(plot_result_intensity);

	cv::Ptr<cv::plot::Plot2d> plot_phi = cv::plot::createPlot2d(data_phi);
	plot_phi->setPlotBackgroundColor(cv::Scalar(50, 50, 50));
	plot_phi->setPlotLineColor(cv::Scalar(50, 50, 255));
	plot_phi->render(plot_result_phi);


	// Draw line segment and perpendicular line
	std::shared_ptr<cv::line_descriptor::KeyLine> kl = keylinesInContours[index][support_candidates_pos[index]];
	cv::line(image_candidates, kl->getStartPoint(), kl->getEndPoint(), cv::Scalar(255, 0, 0), 10);

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


	cv::namedWindow("Image with candidate segments", cv::WINDOW_NORMAL);
	//cv::resizeWindow("Image with candidate segments", 1080, 960);
	cv::imshow("Image with candidate segments", image_candidates);

	/*
	cv::imshow("Plot intensity", plot_result_intensity);
	cv::imshow("Plot phi", plot_result_phi);
	*/

	cv::waitKey(0);

	mglGraph gr;
	mglData y;

	//mgls_prepare1d(&y);
	y.Set(phi.data(), phi.size());
	gr.SetOrigin(0,0,0);
	//gr.SubPlot(2,2,0,"");
	gr.Title("Plot plot (default)");
	gr.SetRanges(0, 1121, -5000, 5000);
	//gr.SetRanges(0, 3320, -5000, 5000);
	gr.Box();
	gr.Axis();
	gr.Grid();
	gr.Plot(y);
	gr.WriteFrame("plot.png");

	return(0);
}
