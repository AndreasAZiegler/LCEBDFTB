#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/plot.hpp>
#include <iostream>
#include <chrono>
#include <memory>

#include <mgl2/mgl.h>

std::vector<std::vector<cv::Point>> getLineSegmentsContours(std::vector<cv::line_descriptor::KeyLine> &keylines,
																														cv::Mat& image_lines) {
	std::vector<std::vector<cv::Point>> contours;
	for (auto it = keylines.begin(); it != keylines.end();) {
		auto kl = *it;
		if(30 < kl.lineLength) {
			// For debug
			//cv::line(image_lines, kl.getStartPoint(), kl.getEndPoint(), cv::Scalar(255, 0, 0));

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

			float temp_1 = 2*linelength*sin_angle;
			float temp_2 = 5.0*linelength*cos_angle;

			std::vector<cv::Point> contour(5);
			contour[0] = (cv::Point2f(start_x - temp_1, start_y + temp_2));
			contour[1] = (cv::Point2f(start_x + temp_1, start_y + temp_2));
			contour[2] = (cv::Point2f(end_x + temp_1, end_y - temp_2));
			contour[3] = (cv::Point2f(end_x - temp_1, end_y - temp_2));
			contour[4] = (cv::Point2f(start_x - temp_1, start_y + temp_2));
			contours.push_back(contour);

			++it;
		} else {
			keylines.erase(it);
		}
	}

	return(contours);
}

void findContainingSegments(std::vector<std::vector<std::shared_ptr<cv::line_descriptor::KeyLine>>> &keylinesInContours,
														std::vector<cv::line_descriptor::KeyLine> keylines,
														std::vector<std::vector<cv::Point>> contours,
														int contours_size) {
	#pragma omp parallel for
	for(int i = 0; i < contours_size; i++) {
		register float px = keylines[i].pt.x;
		register float py = keylines[i].pt.y;
		register float ll_2 = keylines[i].lineLength * 2;

		register int keylines_size = keylines.size();
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
}

void calculateSupportScores(std::vector<std::vector<std::shared_ptr<cv::line_descriptor::KeyLine>>> &keylinesInContours,
														std::vector<std::vector<int>> &support_scores,
														int keylinesInContours_size) {
	#pragma omp parallel for
	for(int i = 0; i < keylinesInContours_size; i++) {
		int keylinesInContours_i_size = keylinesInContours[i].size();
		support_scores[i] = std::vector<int>(keylinesInContours[i].size());
		#pragma omp parallel for
		for(int j = 0; j < keylinesInContours_i_size; ++j) {
			support_scores[i][j] = 0;
		}
	}

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
}

void selectSCand(std::vector<std::vector<int>> &support_scores,
								 std::vector<int> &support_candidates,
								 std::vector<int> &support_candidates_pos,
								 std::vector<std::vector<std::shared_ptr<cv::line_descriptor::KeyLine>>> &keylinesInContours,
								 cv::Mat &image_candidates,
								 int keylinesInContours_size) {
	#pragma omp parallel for
	for(int i = 0; i < keylinesInContours_size; i++) {
		support_candidates_pos[i] = std::distance(support_scores[i].begin(),
																							std::max_element(support_scores[i].begin(),
																							support_scores[i].end()));
		//std::cout << "support_candidate_pos[" << i << "] = " << support_candidates_pos[i] << std::endl;

		support_candidates[i] = support_scores[i][std::distance(support_scores[i].begin(),
																														std::max_element(support_scores[i].begin(),
																														support_scores[i].end()))];
		//std::cout << "max_support_candidates[" << i << "] = " << support_candidates[i] << std::endl;
		//std::cout << "min_support_candidates[" << i << "] = " << support_scores[i][std::distance(support_scores[i].begin(), std::min_element(support_scores[i].begin(), support_scores[i].end()))] << std::endl;

		// For debug
		/*
		if(support_candidates_threshold < support_candidates[i]) {
			std::shared_ptr<cv::line_descriptor::KeyLine> kl = keylinesInContours[i][support_candidates_pos[i]];
			cv::line(image_candidates, kl->getStartPoint(), kl->getEndPoint(), cv::Scalar(0, 0, 255));
		}
		*/
	}
}

/**
	* @todo Set line iterators from start to stop intensities position, keep intensities vector and initialize it with zero
	* values and than run intensities over shorter line iterators.
	*/
void createVectorsOfIntensities(std::vector<int> &support_candidates,
																std::vector<int> &support_candidates_pos,
																std::vector<std::vector<std::shared_ptr<cv::line_descriptor::KeyLine>>> &keylinesInContours,
																std::vector<std::vector<int>> &startStopIntensitiesPosition,
																std::vector<std::vector<cv::Point>> &perpendicularLineStartEndPoints,
																std::vector<std::vector<std::vector<uchar>>> &intensities,
																cv::Mat &image_greyscale,
																int image_cols,
																int intensities_size,
																int support_candidates_threshold) {
	float angle;
	float kl_pt_x;
	float kl_pt_y;
	float temp_0;
	float temp_1;
	float temp_2;
	int temp_start;
	int temp_end;
	std::vector<cv::Point> pt1s(6);
	float temp_3;
	float temp_4;
	std::vector<cv::Point> pt2s(6);
	int pt_size;
	int start;
	int end;
	int lineIterators_size_2;
	int lineIterators_5_count;

	for(int i = 0; i < intensities_size; i++) {
		if(support_candidates_threshold < support_candidates[i]) {
			std::shared_ptr<cv::line_descriptor::KeyLine> kl = keylinesInContours[i][support_candidates_pos[i]];

			angle = kl->angle;
			if(0 < angle) {
				angle	= (M_PI_2 - angle);
			} else {
				angle = -(M_PI_2 - std::abs(angle));
			}

			kl_pt_y = kl->pt.y;
			kl_pt_x = kl->pt.x;

			temp_0= kl_pt_y + kl_pt_x*std::tan(angle);
			temp_1 = 600*std::cos(angle);
			temp_2= kl_pt_y + temp_1*std::tan(angle);

			temp_start = kl_pt_x - temp_1;
			temp_end = kl_pt_x + temp_1;
			startStopIntensitiesPosition[i][0] = temp_start;
			startStopIntensitiesPosition[i][1] = temp_end;

			pt1s[0] = (cv::Point(temp_start, temp_2 - 16));
			pt1s[1] = (cv::Point(temp_start, temp_2 - 8));
			pt1s[2] = (cv::Point(temp_start, temp_2));
			pt1s[3] = (cv::Point(temp_start, temp_2 + 8));
			pt1s[4] = (cv::Point(temp_start, temp_2 + 16));
			pt1s[5] = cv::Point(0, temp_0);

			temp_3 = kl_pt_y - (image_cols - kl_pt_x)*std::tan(angle);
			temp_4 = kl_pt_y - temp_1*std::tan(angle);

			pt2s[0] = (cv::Point(temp_end, temp_4 - 16));
			pt2s[1] = (cv::Point(temp_end, temp_4 - 8));
			pt2s[2] = (cv::Point(temp_end, temp_4));
			pt2s[3] = (cv::Point(temp_end, temp_4 + 8));
			pt2s[4] = (cv::Point(temp_end, temp_4 + 16));
			pt2s[5] = cv::Point(image_cols, temp_3);

			perpendicularLineStartEndPoints[i][0] = cv::Point(0, temp_0);
			perpendicularLineStartEndPoints[i][1] = cv::Point(image_cols, temp_3);


			pt_size = pt1s.size();
			std::vector<cv::LineIterator> lineIterators;
			for(int j = 0; j < pt_size; j++) {
				lineIterators.push_back(cv::LineIterator(image_greyscale, pt1s[j], pt2s[j], 8, true));
				//cv::line(image_candidates, pt1s[j], pt2s[j], cv::Scalar(0, 255, 0), 1);
			}

			for(start = 0; temp_start > lineIterators[5].pos().x; ++lineIterators[5], start++);
			for(end = start; temp_end > lineIterators[5].pos().x; ++lineIterators[5], end++);

			lineIterators_size_2 = lineIterators.size() - 1;
			for(int j = 0; j < lineIterators_size_2; j++) {
				lineIterators_5_count = lineIterators[5].count;
				intensities[i][j] = std::vector<uchar>(lineIterators_5_count);

				for(uchar &intensity : intensities[i][j]) {
					intensity = 0;
				}

				for(int k = start; k < end; k++, ++lineIterators[j]) {
						intensities[i][j][k] = image_greyscale.at<uchar>(lineIterators[j].pos());
				}
			}
		}
	}
}

void computePhis(int delta,
								 std::vector<int> &support_candidates,
								 int support_candidates_threshold,
								 std::vector<std::vector<std::vector<uchar>>> &intensities,
								 int intensities_size,
								 std::vector<std::vector<std::vector<int>>> &phis,
								 std::vector<std::vector<int>> &startStopIntensitiesPosition,
								 std::vector<int> &start_barcode_pos,
								 std::vector<int> &end_barcode_pos) {


	int phis_i_5_k;
	//#pragma omp parallel for
	for(int i = 0; i < intensities_size; i++) {
		int intensities_i_size = intensities[i].size();
		int startStopIntensitiesPosition_i_0 = startStopIntensitiesPosition[i][0];
		int startStopIntensitiesPosition_i_1 = startStopIntensitiesPosition[i][1];

		if(support_candidates_threshold < support_candidates[i]) {
			#pragma omp parallel for
			for(int j = 0; j < intensities_i_size; j++) {
				int intensities_i_j_size = intensities[i][j].size();
				phis[i][j] = std::vector<int>(intensities_i_j_size);
				int max = 0;
				int min = 0;
				#pragma omp parallel for
				for(int k = 0; k < intensities_i_j_size; k++) {
					if(startStopIntensitiesPosition_i_0 - delta < k) {
						if(startStopIntensitiesPosition_i_1 + delta > k) {
							int phi_1 = 0;
							int phi_2 = 0;

							int start_1 = k - delta - 1;
							int end_1 = k;

							if(startStopIntensitiesPosition_i_1 < k) {
								if (start_1 < startStopIntensitiesPosition_i_1) {
									end_1 = startStopIntensitiesPosition_i_1;
								}
							}

							if(end_1 > startStopIntensitiesPosition_i_0) {
								if(startStopIntensitiesPosition_i_0 > start_1) {
									start_1 = startStopIntensitiesPosition_i_0;
								}
							}

							#pragma omp parallel for
							for(int l = start_1; l < end_1; l++) {
								phi_1 += std::abs(intensities[i][j][l + 1] - intensities[i][j][l]);
							}

							int start_2 = k;
							int end_2 = intensities_i_j_size;
							if(startStopIntensitiesPosition_i_0 > start_2) {
								if(end_1 > startStopIntensitiesPosition_i_0) {
									start_2 = startStopIntensitiesPosition_i_0;
								}
							}

							if(intensities_i_j_size > (k + delta + 1)) {
								end_2 = k + delta + 1;
							}

							if(startStopIntensitiesPosition_i_1 < end_2) {
								if(start_2 < startStopIntensitiesPosition_i_1) {
									end_2 = startStopIntensitiesPosition_i_1;
								}
							}

							#pragma omp parallel for
							for(int l = start_2; l < end_2; l++) {
								phi_2 += std::abs(intensities[i][j][l] - intensities[i][j][l + 1]);
							}

							phis[i][j][k] = phi_1 - phi_2;
						}
					}
				}
			}

			phis[i][5] = std::vector<int>(phis[i][0].size());
			int phis_i_5_size = phis[i][5].size();
			for(int k = 0; k < phis_i_5_size; k++) {
				phis_i_5_k = phis[i][0][k] + phis[i][1][k] + phis[i][2][k] + phis[i][3][k] + phis[i][4][k];
				phis[i][5][k] = phis_i_5_k / 5;
			}

			start_barcode_pos[i] = std::distance(phis[i][5].begin(),
																					 std::min_element(phis[i][5].begin(),
																					 phis[i][5].end()));
			end_barcode_pos[i] = std::distance(phis[i][5].begin(),
																				 std::max_element(phis[i][5].begin(),
																				 phis[i][5].end()));
		}
	}
}

void calculateBoundingBoxes(int keylinesInContours_size,
														std::vector<int> &start_barcode_pos,
														std::vector<int> &end_barcode_pos,
														std::vector<int> &support_candidates,
														int support_candidates_threshold,
														std::vector<cv::line_descriptor::KeyLine> &keylines,
														std::vector<std::vector<cv::Point>> &contour,
														std::vector<std::vector<cv::Point>> &perpendicularLineStartEndPoints,
														cv::Mat &image_candidates) {

	int length;

	#pragma omp parallel for
	for(int i = 0; i < keylinesInContours_size; i++) {
		length = end_barcode_pos[i] - start_barcode_pos[i];
		if(support_candidates_threshold < support_candidates[i]) {
		//if(i == index) {
			if(0 < length) {
				if((length / keylines[i].lineLength) < 10) {
					/*
					int diff_1 = std::abs(start_barcode_pos[i][2] - start_barcode_pos[i][0]);
					int diff_2 = std::abs(start_barcode_pos[i][2] - start_barcode_pos[i][1]);
					int diff_3 = std::abs(start_barcode_pos[i][2] - start_barcode_pos[i][3]);
					int diff_4 = std::abs(start_barcode_pos[i][2] - start_barcode_pos[i][4]);
					*/
					float angle = keylines[i].angle;
					if(0 < angle) {
						angle	= (M_PI_2 - angle);
					} else {
						angle = -(M_PI_2 - std::abs(angle));
					}

					/*
					std::cout << "support_candidates[" << i << "] = " << support_candidates[i] << std::endl;
					std::cout << "diff_1 = " << diff_1 << ", diff_2 = " << diff_2 << ", diff_3 = " << diff_3 << ", diff_4 = " << diff_4 << std::endl;
					std::cout << "Add one bounding box contour!" << std::endl;
					std::cout << "keylines[" << i << "].lineLength = " << keylines[i].lineLength << std::endl;
					std::cout << "start_barcode_pos = " << start_barcode_pos[i] << " , end_barcode_pos = " << end_barcode_pos[i] << std::endl;// ", end_pos = " << phis[i][2].size() << ", angle = " << 180*angle/M_PI << std::endl;
					*/

					contour[i][0] = cv::Point(perpendicularLineStartEndPoints[i][0].x + std::cos(angle)*start_barcode_pos[i] - keylines[i].lineLength*std::sin(angle)*0.5,
																		perpendicularLineStartEndPoints[i][0].y - std::sin(angle)*start_barcode_pos[i] - keylines[i].lineLength*std::cos(angle)*0.5);
					contour[i][1] = cv::Point(perpendicularLineStartEndPoints[i][0].x + std::cos(angle)*end_barcode_pos[i] - keylines[i].lineLength*std::sin(angle)*0.5,
																		perpendicularLineStartEndPoints[i][0].y - std::sin(angle)*end_barcode_pos[i] - keylines[i].lineLength*std::cos(angle)*0.5);
					contour[i][2] = cv::Point(perpendicularLineStartEndPoints[i][0].x + std::cos(angle)*end_barcode_pos[i] + keylines[i].lineLength*std::sin(angle)*0.5,
																		perpendicularLineStartEndPoints[i][0].y - std::sin(angle)*end_barcode_pos[i] + keylines[i].lineLength*std::cos(angle)*0.5);
					contour[i][3] = cv::Point(perpendicularLineStartEndPoints[i][0].x + std::cos(angle)*start_barcode_pos[i] + keylines[i].lineLength*std::sin(angle)*0.5,
																		perpendicularLineStartEndPoints[i][0].y - std::sin(angle)*start_barcode_pos[i] + keylines[i].lineLength*std::cos(angle)*0.5);

					cv::putText(image_candidates, std::to_string(i), contour[i][0], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));
					/*
					cv::line(image_candidates, keylines[i].getStartPoint(), keylines[i].getEndPoint(), cv::Scalar(255, 0, 0), 2);
					std::cout << "perpencidularLineStartEndPoints[" << i << "][0].x = " << perpencidularLineStartEndPoints[i][0].x << ", perpencidularLineStartEndPoints[" << i << "][0].y = " << perpencidularLineStartEndPoints[i][0].y << std::endl;
					std::cout << "contour[" << i << "][0] = " << contour[i][0] << ", contour[" << i << "][1] = " << contour[i][1] <<
											 ", contour[" << i << "][2] = " << contour[i][2] << ", contour[" << i << "][3] = " << contour[i][3] << std::endl;
					*/
				}
			}
		}
	}
}

int main(int argc, char** argv) {
  int support_candidates_threshold = 7;
  int delta = 125;

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
	std::vector<std::vector<cv::Point>> contours = getLineSegmentsContours(keylines, image_lines);
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

	// Find for every bounding box the containing segments
	std::vector<std::vector<std::shared_ptr<cv::line_descriptor::KeyLine>>> keylinesInContours(contours.size());
	int contours_size = contours.size();

	start = std::chrono::steady_clock::now();
	findContainingSegments(keylinesInContours, keylines, contours, contours_size);
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

	selectSCand(support_scores, support_candidates, support_candidates_pos, keylinesInContours, image_candidates, keylinesInContours_size);
	end = std::chrono::steady_clock::now();
	std::cout << "Select s_cand: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	// Create vectors of intensities
	std::vector<std::vector<cv::Point>> perpendicularLineStartEndPoints(keylinesInContours_size, std::vector<cv::Point>(2));
	std::vector<std::vector<std::vector<uchar>>> intensities(keylinesInContours_size, std::vector<std::vector<uchar>>(5));
	std::vector<std::vector<int>> startStopIntensitiesPosition(keylinesInContours_size, std::vector<int>(2));

	int intensities_size = intensities.size();
	int image_cols = image_greyscale.cols;

	start = std::chrono::steady_clock::now();
	createVectorsOfIntensities(support_candidates,
														 support_candidates_pos,
														 keylinesInContours,
														 startStopIntensitiesPosition,
														 perpendicularLineStartEndPoints,
														 intensities,
														 image_greyscale,
														 image_cols,
														 intensities_size,
														 support_candidates_threshold);

	end = std::chrono::steady_clock::now();
	std::cout << "Compute intensities: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	// Compute phis
	start = std::chrono::steady_clock::now();
	std::vector<std::vector<std::vector<int>>> phis(keylinesInContours_size, std::vector<std::vector<int>>(6));
	std::vector<int> start_barcode_pos(keylinesInContours_size);
	std::vector<int> end_barcode_pos(keylinesInContours_size);

	computePhis(delta,
							support_candidates,
							support_candidates_threshold,
							intensities,
							intensities_size,
							phis,
							startStopIntensitiesPosition,
							start_barcode_pos,
							end_barcode_pos);
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
	index = 3965;
	std::cout << "index = " << index << ", size() = " << intensities[index][2].size() << std::endl;

	// Calculate bounding boxes
	start = std::chrono::steady_clock::now();
	std::vector<std::vector<cv::Point>> contour(keylinesInContours_size, std::vector<cv::Point>(4));
	calculateBoundingBoxes(keylinesInContours_size,
												 start_barcode_pos,
												 end_barcode_pos,
												 support_candidates,
												 support_candidates_threshold,
												 keylines,
												 contour,
												 perpendicularLineStartEndPoints,
												 image_candidates);

	end = std::chrono::steady_clock::now();
	std::cout << "Calculated bounding boxes: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	std::cout << "Total time: " << std::chrono::duration <double, std::milli> (end - total_start).count() << " ms" << std::endl;

	cv::drawContours(image_candidates, contour, -1, cv::Scalar(255, 0, 0), 1);

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

	return(0);
}
