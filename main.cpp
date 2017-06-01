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
	//for(cv::line_descriptor::KeyLine kl : keylines) {
	for (auto it = keylines.begin(); it != keylines.end();) {
		auto kl = *it;
		if(30 < kl.lineLength) {
			cv::line(image_lines, kl.getStartPoint(), kl.getEndPoint(), cv::Scalar(255, 0, 0));
			//std::vector<cv::Point2f> point = {kl.getStartPoint(), kl.getEndPoint()};

			float linelength = kl.lineLength;
			float angle = kl.angle;
			float cos_angle = std::abs(std::cos(angle));
			float sin_angle = std::abs(std::sin(angle));
			float start_x;
			float start_y;
			float end_x;
			float end_y;
			if(kl.startPointY > kl.endPointY) {
				start_x = kl.startPointX;
				start_y = kl.startPointY;
				end_x = kl.endPointX;
				end_y = kl.endPointY;
			} else {
				start_x = kl.endPointX;
				start_y = kl.endPointY;
				end_x = kl.startPointX;
				end_y = kl.startPointY;
			}

			std::vector<cv::Point> contour;
			contour.push_back(cv::Point2f(start_x - 2*linelength*sin_angle, start_y + 5.0*linelength*cos_angle));
			contour.push_back(cv::Point2f(start_x + 2*linelength*sin_angle, start_y + 5.0*linelength*cos_angle));
			contour.push_back(cv::Point2f(end_x + 2*linelength*sin_angle, end_y - 5.0*linelength*cos_angle));
			contour.push_back(cv::Point2f(end_x - 2*linelength*sin_angle, end_y - 5.0*linelength*cos_angle));
			contour.push_back(cv::Point2f(start_x - 2*linelength*sin_angle, start_y + 5.0*linelength*cos_angle));
			contours.push_back(contour);
			lookup.push_back(kl.pt);

			++it;
		} else {
			keylines.erase(it);
		}
	}
	end = std::chrono::steady_clock::now();
	std::cout << "Creating bounding boxes: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	cv::drawContours(image_lines, contours, -1, cv::Scalar(0, 0, 255));

	std::cout << "Number of bounding boxes: " << contours.size() << std::endl;

	cv::imwrite("debug-line-segments.jpg", image_lines);
	/*
	cv::namedWindow("Image with line segments", cv::WINDOW_NORMAL);
	cv::resizeWindow("Image with line segments", 600, 600);
	cv::imshow("Image with line segments", image_lines);
	*/

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

				//if((diff_length) < 5.0) {
				if((diff_length) < 4.0) {
					diff_angle = std::abs(kl_j->angle - kl_k->angle);
					//diff_angle_vec.push_back(diff_angle);

					//if((diff_angle) < 3.0) {
					if((diff_angle) < 0.26) {
						diff_norm_pt = cv::norm(kl_j->pt - kl_k->pt);
						//diff_norm_pt_vec.push_back(diff_norm_pt);

						if((diff_norm_pt) < 300.0) {
						//if((diff_norm_pt) < 400.0) {
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

		if(7 < support_candidates[i]) {
			std::shared_ptr<cv::line_descriptor::KeyLine> kl = keylinesInContours[i][support_candidates_pos[i]];
			cv::line(image_candidates, kl->getStartPoint(), kl->getEndPoint(), cv::Scalar(0, 0, 255));
		}
	}
	end = std::chrono::steady_clock::now();
	std::cout << "Select s_cand: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	// Create vectors of intensities
	start = std::chrono::steady_clock::now();
	std::vector<std::vector<cv::Point>> perpencidularLineStartEndPoints(keylinesInContours_size, std::vector<cv::Point>(2));
	std::vector<std::vector<std::vector<uchar>>> intensities(keylinesInContours_size, std::vector<std::vector<uchar>>(5));
	std::vector<std::vector<int>> startStopIntensitiesPosition(keylinesInContours_size, std::vector<int>(2));
	/**
		* @todo Set line iterators from start to stop intensities position, keep intensities vector and initialize it with zero
		* values and than run intensities over shorter line iterators.
		*/
	for(unsigned int i = 0; i < intensities.size(); i++) {
		if(7 < support_candidates[i]) {
			std::shared_ptr<cv::line_descriptor::KeyLine> kl = keylinesInContours[i][support_candidates_pos[i]];

			float angle = kl->angle;
			if(0 < angle) {
				angle	= (M_PI_2 - angle);
			} else {
				angle = -(M_PI_2 - std::abs(angle));
			}

			std::vector<cv::Point> pt1s;
			pt1s.push_back(cv::Point(0, kl->pt.y + kl->pt.x*std::tan(angle) - 16));
			pt1s.push_back(cv::Point(0, kl->pt.y + kl->pt.x*std::tan(angle) - 8));
			pt1s.push_back(cv::Point(0, kl->pt.y + kl->pt.x*std::tan(angle)));
			pt1s.push_back(cv::Point(0, kl->pt.y + kl->pt.x*std::tan(angle) + 8));
			pt1s.push_back(cv::Point(0, kl->pt.y + kl->pt.x*std::tan(angle) + 16));

			startStopIntensitiesPosition[i][0] = kl->pt.x - 600*std::cos(angle);

			std::vector<cv::Point> pt2s;
			pt2s.push_back(cv::Point(image_greyscale.cols, kl->pt.y - (image_greyscale.cols - kl->pt.x)*std::tan(angle) - 16));
			pt2s.push_back(cv::Point(image_greyscale.cols, kl->pt.y - (image_greyscale.cols - kl->pt.x)*std::tan(angle) - 8));
			pt2s.push_back(cv::Point(image_greyscale.cols, kl->pt.y - (image_greyscale.cols - kl->pt.x)*std::tan(angle)));
			pt2s.push_back(cv::Point(image_greyscale.cols, kl->pt.y - (image_greyscale.cols - kl->pt.x)*std::tan(angle) + 8));
			pt2s.push_back(cv::Point(image_greyscale.cols, kl->pt.y - (image_greyscale.cols - kl->pt.x)*std::tan(angle) + 16));

			startStopIntensitiesPosition[i][1] = kl->pt.x + 600*std::cos(angle);

			perpencidularLineStartEndPoints[i][0] = cv::Point(0, kl->pt.y + kl->pt.x*std::tan(angle));
			perpencidularLineStartEndPoints[i][1] = cv::Point(image_greyscale.cols, kl->pt.y - (image_greyscale.cols - kl->pt.x)*std::tan(angle));

			std::vector<cv::LineIterator> lineIterators;
			for(unsigned int j = 0; j < intensities[i].size(); j++) {
				lineIterators.push_back(cv::LineIterator(image_greyscale, pt1s[j], pt2s[j], 8, true));
				//cv::line(image_candidates, pt1s[j], pt2s[j], cv::Scalar(0, 255, 0), 1);
			}

			for(unsigned int j = 0; j < lineIterators.size(); j++) {
				intensities[i][j] = std::vector<uchar>(lineIterators[j].count);

				for(int k = 0; k < lineIterators[j].count; k++, ++lineIterators[j]) {
					/*
					std::cout << "Angle = " << 180*angle/M_PI << std::endl;
					std::cout << "Start taking intensities at: " << startStopIntensitiesPosition[i][0] << std::endl;
					std::cout << "End taking intensities at: " << startStopIntensitiesPosition[i][1] << std::endl;
					std::cout << "pos.x: " << lineIterators[j].pos().x << std::endl;
					*/

					if((startStopIntensitiesPosition[i][0] < lineIterators[j].pos().x) && (startStopIntensitiesPosition[i][1] > lineIterators[j].pos().x)) {
						intensities[i][j][k] = image_greyscale.at<uchar>(lineIterators[j].pos());
					} else {
						intensities[i][j][k] = 0;
					}
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

	//#pragma omp parallel for
	for(unsigned int i = 0; i < intensities.size(); i++) {
		if(7 < support_candidates[i]) {
			//#pragma omp parallel for
			for(unsigned int j = 0; j < intensities[i].size(); j++) {
				phis[i][j] = std::vector<int>(intensities[i][j].size());
				int max = 0;
				int min = 0;
				//#pragma omp parallel for
				for(int k = 0; k < intensities[i][j].size(); k++) {
					int phi_1 = 0;
					int phi_2 = 0;
					for(int l = 0; l < 150; l++) {
						if(0 < (k - l - 1)) {
							phi_1 += std::abs(intensities[i][j][k - l] - intensities[i][j][k - l - 1]);
						}
						if(intensities[i][j].size() > (k + l + 1)) {
							phi_2 += std::abs(intensities[i][j][k + l] - intensities[i][j][k + l + 1]);
						}
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
	}
	end = std::chrono::steady_clock::now();
	std::cout << "Compute phis: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;


	std::cout << "Total time: " << std::chrono::duration <double, std::milli> (end - total_start).count() << " ms" << std::endl;

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
	index = 3655;
	std::cout << "index = " << index << ", size() = " << intensities[index][2].size() << std::endl;

	// Calculate bounding boxes
	std::vector<std::vector<cv::Point>> contour(keylinesInContours_size, std::vector<cv::Point>(4));
	for(int i = 0; i < keylinesInContours_size; i++) {
		int length = end_barcode_pos[i][2] - start_barcode_pos[i][2];
		if(7 < support_candidates[i]) {
		//if(i == index) {
			if((0 < length) && ((length / keylines[i].lineLength) < 10)) {
				int diff_1 = std::abs(start_barcode_pos[i][2] - start_barcode_pos[i][0]);
				int diff_2 = std::abs(start_barcode_pos[i][2] - start_barcode_pos[i][1]);
				int diff_3 = std::abs(start_barcode_pos[i][2] - start_barcode_pos[i][3]);
				int diff_4 = std::abs(start_barcode_pos[i][2] - start_barcode_pos[i][4]);
				float angle = keylines[i].angle;
				if(0 < angle) {
					angle	= (M_PI_2 - angle);
				} else {
					angle = -(M_PI_2 - std::abs(angle));
				}

				//if(50 > diff_1 + diff_2 + diff_3 + diff_4) {
				if(true) {
					std::cout << "support_candidates[" << i << "] = " << support_candidates[i] << std::endl;
					std::cout << "diff_1 = " << diff_1 << ", diff_2 = " << diff_2 << ", diff_3 = " << diff_3 << ", diff_4 = " << diff_4 << std::endl;
					std::cout << "Add one bounding box contour!" << std::endl;
					std::cout << "keylines[" << i << "].lineLength = " << keylines[i].lineLength << std::endl;
					std::cout << "start_barcode_pos[" << i << "][2] = " << start_barcode_pos[i][2] << " , end_barcode_pos[" << i << "][2] = " << end_barcode_pos[i][2] << ", end_pos = " << phis[i][2].size() << ", angle = " << 180*angle/M_PI << std::endl;

					contour[i][0] = cv::Point(perpencidularLineStartEndPoints[i][0].x + std::cos(angle)*start_barcode_pos[i][2] - keylines[i].lineLength*std::sin(angle)*0.5,
																		perpencidularLineStartEndPoints[i][0].y - std::sin(angle)*start_barcode_pos[i][2] - keylines[i].lineLength*std::cos(angle)*0.5);
					contour[i][1] = cv::Point(perpencidularLineStartEndPoints[i][0].x + std::cos(angle)*end_barcode_pos[i][2] - keylines[i].lineLength*std::sin(angle)*0.5,
																		perpencidularLineStartEndPoints[i][0].y - std::sin(angle)*end_barcode_pos[i][2] - keylines[i].lineLength*std::cos(angle)*0.5);
					contour[i][2] = cv::Point(perpencidularLineStartEndPoints[i][0].x + std::cos(angle)*end_barcode_pos[i][2] + keylines[i].lineLength*std::sin(angle)*0.5,
																		perpencidularLineStartEndPoints[i][0].y - std::sin(angle)*end_barcode_pos[i][2] + keylines[i].lineLength*std::cos(angle)*0.5);
					contour[i][3] = cv::Point(perpencidularLineStartEndPoints[i][0].x + std::cos(angle)*start_barcode_pos[i][2] + keylines[i].lineLength*std::sin(angle)*0.5,
																		perpencidularLineStartEndPoints[i][0].y - std::sin(angle)*start_barcode_pos[i][2] + keylines[i].lineLength*std::cos(angle)*0.5);

					//cv::putText(image_candidates, std::to_string(i), keylines[i].getEndPoint(), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));
					cv::putText(image_candidates, std::to_string(i), contour[i][0], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));
					//cv::line(image_candidates, keylines[i].getStartPoint(), keylines[i].getEndPoint(), cv::Scalar(255, 0, 0), 2);
					/*
					std::cout << "perpencidularLineStartEndPoints[" << i << "][0].x = " << perpencidularLineStartEndPoints[i][0].x << ", perpencidularLineStartEndPoints[" << i << "][0].y = " << perpencidularLineStartEndPoints[i][0].y << std::endl;
					std::cout << "std::cos(M_PI_2 + keylines[" << i << "].angle)*start_barcode_pos[" << i << "][2] = " << std::cos(M_PI_2 + keylines[i].angle)*start_barcode_pos[i][2] << std::endl;
					std::cout << "std::sin(M_PI_2 + keylines[" << i << "].angle)*start_barcode_pos[" << i << "][2] = " << std::sin(M_PI_2 + keylines[i].angle)*start_barcode_pos[i][2] << std::endl;
					*/
					std::cout << "contour[" << i << "][0] = " << contour[i][0] << ", contour[" << i << "][1] = " << contour[i][1] <<
											 ", contour[" << i << "][2] = " << contour[i][2] << ", contour[" << i << "][3] = " << contour[i][3] << std::endl;
				}
			}
		}
	}

	cv::drawContours(image_candidates, contour, -1, cv::Scalar(255, 0, 0), 1);

	// Plot intensity and phi of selected line segment
	std::vector<std::vector<double>> intensity(5, std::vector<double>(intensities[index][2].size()));
	std::vector<std::vector<double>> phi(5, std::vector<double>(phis[index][2].size()));
	for(int j = 0; j < 5; j++) {
		for(unsigned int i = 0; i < intensities[index][2].size(); i++) {
			intensity[j][i] = static_cast<double>(intensities[index][j][i]);
			phi[j][i] = static_cast<double>(phis[index][j][i]);
		}
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

	mglGraph gr_int;
	mglGraph gr_phi;
	std::vector<mglData> mgl_phi(5);
	std::vector<mglData> mgl_intensities(5);

	//mgls_prepare1d(&y);
	for(int i = 0; i < 5; i++) {
		mgl_phi[i].Set(phi[i].data(), phi[i].size());
		mgl_intensities[i].Set(intensity[i].data(), intensity[i].size());


		gr_phi.SubPlot(1, 5, i);
		std::string str = "Phi " + std::to_string(i);
		gr_phi.Title(str.c_str());
		gr_phi.SetOrigin(0,0,0);
		gr_phi.SetRanges(0, phis[index][2].size(), -5800, 5800);
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


	//gr.Box();
	gr_int.WriteFrame("plot-intensities.png");
	gr_phi.WriteFrame("plot-phi.png");

	return(0);
}
