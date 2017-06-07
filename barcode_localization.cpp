#include "barcode_localization.h"
#include <zbar.h>

std::vector<std::vector<cv::Point>> getLineSegmentsContours(std::vector<cv::line_descriptor::KeyLine> &keylines,
																														cv::Mat& image_lines,
																														int minLineLength) {
	std::vector<std::vector<cv::Point>> contours;
	for (auto it = keylines.begin(); it != keylines.end();) {
		auto kl = *it;
		if(minLineLength < kl.lineLength) {
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
								 int keylinesInContours_size,
								 cv::Mat &image_candidates,
								 int support_candidates_threshold) {

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
		if(support_candidates_threshold < support_candidates[i]) {
			std::shared_ptr<cv::line_descriptor::KeyLine> kl = keylinesInContours[i][support_candidates_pos[i]];
			cv::line(image_candidates, kl->getStartPoint(), kl->getEndPoint(), cv::Scalar(0, 0, 255));
		}
	}
}

void createVectorsOfIntensities(std::vector<int> &support_candidates,
																std::vector<int> &support_candidates_pos,
																std::vector<std::vector<std::shared_ptr<cv::line_descriptor::KeyLine>>> &keylinesInContours,
																std::vector<std::vector<int>> &startStopIntensitiesPosition,
																std::vector<std::vector<cv::Point>> &perpendicularLineStartEndPoints,
																std::vector<std::vector<std::vector<uchar>>> &intensities,
																cv::Mat &image_greyscale,
																int image_cols,
																int image_rows,
																int intensities_size,
																int support_candidates_threshold,
																std::vector<bool> &deletedContours) {
	float angle;
	float kl_pt_x;
	float kl_pt_y;
	float temp_0;
	float temp_1;
	float temp_start_y;
	int temp_start_x;
	int temp_end_x;
	int temp_start_mock_x;
	int temp_start_mock_y;
	int temp_end_mock_x;
	int temp_end_mock_y;
	std::vector<cv::Point> pt1s(6);
	float temp_3;
	float temp_end_y;
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
			//std::cout << "angle = " << 180*angle/M_PI << std::endl;
			if(M_PI_2 < angle) {
				angle -= M_PI_2;
			}

			kl_pt_y = kl->pt.y;
			kl_pt_x = kl->pt.x;

			if(M_PI_4 > std::abs(angle)) {
				if(0 < angle) {
					angle	= (M_PI_2 - angle);
				} else {
					angle = -(M_PI_2 - std::abs(angle));
				}

				temp_1 = 600*std::sin(angle);
				temp_start_x = kl_pt_x - temp_1;
				temp_start_y = kl_pt_y - temp_1*(1/std::tan(angle));
				temp_end_x = kl_pt_x + temp_1;
				temp_end_y = kl_pt_y + temp_1*(1/std::tan(angle));

				temp_start_mock_x = kl_pt_x - kl_pt_y*std::tan(angle);
				temp_start_mock_y = 0;
				temp_end_mock_x = kl_pt_x + (image_rows - kl_pt_y)*std::tan(angle);
				temp_end_mock_y = image_rows;
			} else {
				if(0 < angle) {
					angle	= (M_PI_2 - angle);
				} else {
					angle = -(M_PI_2 - std::abs(angle));
				}

				temp_1 = 600*std::cos(angle);
				temp_start_x = kl_pt_x - temp_1;
				temp_start_y= kl_pt_y + temp_1*std::tan(angle);
				temp_end_x = kl_pt_x + temp_1;
				temp_end_y = kl_pt_y - temp_1*std::tan(angle);

				temp_start_mock_x = 0;
				temp_start_mock_y = kl_pt_y + kl_pt_x*std::tan(angle);
				temp_end_mock_x = image_cols;
				temp_end_mock_y = kl_pt_y - (image_cols - kl_pt_x)*std::tan(angle);
			}

			startStopIntensitiesPosition[i][0] = temp_start_x;
			startStopIntensitiesPosition[i][1] = temp_end_x;

			pt1s[0] = (cv::Point(temp_start_x, temp_start_y - 16));
			pt1s[1] = (cv::Point(temp_start_x, temp_start_y - 8));
			pt1s[2] = (cv::Point(temp_start_x, temp_start_y));
			pt1s[3] = (cv::Point(temp_start_x, temp_start_y + 8));
			pt1s[4] = (cv::Point(temp_start_x, temp_start_y + 16));
			pt1s[5] = cv::Point(temp_start_mock_x, temp_start_mock_y);

			pt2s[0] = (cv::Point(temp_end_x, temp_end_y - 16));
			pt2s[1] = (cv::Point(temp_end_x, temp_end_y - 8));
			pt2s[2] = (cv::Point(temp_end_x, temp_end_y));
			pt2s[3] = (cv::Point(temp_end_x, temp_end_y + 8));
			pt2s[4] = (cv::Point(temp_end_x, temp_end_y + 16));
			pt2s[5] = cv::Point(temp_end_mock_x, temp_end_mock_y);

			temp_0 = kl_pt_y + kl_pt_x*std::tan(angle);
			temp_3 = kl_pt_y - (image_cols - kl_pt_x)*std::tan(angle);

			perpendicularLineStartEndPoints[i][0] = cv::Point(temp_start_mock_x, temp_start_mock_y);
			perpendicularLineStartEndPoints[i][1] = cv::Point(temp_end_mock_x, temp_end_mock_y);


			pt_size = pt1s.size();
			std::vector<cv::LineIterator> lineIterators;
			for(int j = 0; j < pt_size; j++) {
				lineIterators.push_back(cv::LineIterator(image_greyscale, pt1s[j], pt2s[j], 8, true));
				//cv::line(image_candidates, pt1s[j], pt2s[j], cv::Scalar(0, 255, 0), 1);
			}

			for(start = 0; (temp_start_x > lineIterators[5].pos().x) && (start < lineIterators[5].count); ++lineIterators[5], start++);
			for(end = start; (temp_end_x > lineIterators[5].pos().x) && (end < lineIterators[5].count); ++lineIterators[5], end++);

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
		} else{
			deletedContours[i] = true;
		}
	}
}

void computePhis(int delta,
								 std::vector<std::vector<std::vector<uchar>>> &intensities,
								 int intensities_size,
								 std::vector<std::vector<std::vector<int>>> &phis,
								 std::vector<std::vector<int>> &startStopIntensitiesPosition,
								 std::vector<int> &start_barcode_pos,
								 std::vector<int> &end_barcode_pos,
								 std::vector<bool> &deletedContours) {

	int phis_i_5_k;
	#pragma omp parallel for
	for(int i = 0; i < intensities_size; i++) {
		int intensities_i_size = intensities[i].size();
		int startStopIntensitiesPosition_i_0 = startStopIntensitiesPosition[i][0];
		int startStopIntensitiesPosition_i_1 = startStopIntensitiesPosition[i][1];

		//if(support_candidates_threshold < support_candidates[i]) {
		if(false == deletedContours[i]) {
			#pragma omp parallel for
			for(int j = 0; j < intensities_i_size; j++) {
				int intensities_i_j_size = intensities[i][j].size();
				phis[i][j] = std::vector<int>(intensities_i_j_size);
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
														std::vector<cv::line_descriptor::KeyLine> &keylines,
														std::vector<std::vector<cv::Point>> &contours,
														std::vector<std::vector<cv::Point>> &perpendicularLineStartEndPoints,
														cv::Mat &image_candidates,
														std::vector<bool> &deletedContours,
														int index,
														int maxLengthToLineLengthRatio,
														int minLengthToLineLengthRatio) {

	/*
	int length;
	float angle;
	float sin_angle;
	float cos_angle;
	float keylines_i_lineLength;
	int perpendicularLineStartEndPoints_i_0_x;
	int perpendicularLineStartEndPoints_i_0_y;
	int start_barcode_pos_i;
	int end_barcode_pos_i;
	int tmp_0;
	int tmp_1;
	int tmp_2;
	int tmp_3;
	float tmp_4;
	float tmp_5;
	*/

	#pragma omp parallel for
	for(int i = 0; i < keylinesInContours_size; i++) {
		int length = end_barcode_pos[i] - start_barcode_pos[i];
		float keylines_i_lineLength = keylines[i].lineLength;
		//if(support_candidates_threshold < support_candidates[i]) {
		if(false == deletedContours[i]) {
			//if(i == index) {
			if(0 < length) {
				if((length / keylines_i_lineLength) < maxLengthToLineLengthRatio) {
					if((length / keylines_i_lineLength) > minLengthToLineLengthRatio) {
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
						float sin_angle = std::sin(angle);
						float cos_angle = std::cos(angle);

						/*
						std::cout << "support_candidates[" << i << "] = " << support_candidates[i] << std::endl;
						std::cout << "diff_1 = " << diff_1 << ", diff_2 = " << diff_2 << ", diff_3 = " << diff_3 << ", diff_4 = " << diff_4 << std::endl;
						std::cout << "Add one bounding box contour!" << std::endl;
						std::cout << "start_barcode_pos = " << start_barcode_pos[i] << " , end_barcode_pos = " << end_barcode_pos[i] << std::endl;// ", end_pos = " << phis[i][2].size() << ", angle = " << 180*angle/M_PI << std::endl;
						std::cout << "keylines[" << i << "].lineLength = " << keylines[i].lineLength << std::endl;
						*/
						int perpendicularLineStartEndPoints_i_0_x = perpendicularLineStartEndPoints[i][0].x;
						int perpendicularLineStartEndPoints_i_0_y = perpendicularLineStartEndPoints[i][0].y;
						int start_barcode_pos_i = start_barcode_pos[i];
						int end_barcode_pos_i = end_barcode_pos[i];

						int tmp_0 = perpendicularLineStartEndPoints_i_0_x + cos_angle*start_barcode_pos_i;
						int tmp_1 = perpendicularLineStartEndPoints_i_0_y - sin_angle*start_barcode_pos_i;
						int tmp_2 = perpendicularLineStartEndPoints_i_0_x + cos_angle*end_barcode_pos_i;
						int tmp_3 = perpendicularLineStartEndPoints_i_0_y - sin_angle*end_barcode_pos_i;
						float tmp_4 = keylines_i_lineLength*sin_angle*0.5;
						float tmp_5 = keylines_i_lineLength*cos_angle*0.5;

						contours[i][0] = cv::Point(tmp_0 - tmp_4,
																			tmp_1 - tmp_5);
						contours[i][1] = cv::Point(tmp_2 - tmp_4,
																			tmp_3 - tmp_5);
						contours[i][2] = cv::Point(tmp_2 + tmp_4,
																			tmp_3 + tmp_5);
						contours[i][3] = cv::Point(tmp_0 + tmp_4,
																			tmp_1 + tmp_5);

						cv::line(image_candidates, keylines[i].getStartPoint(), keylines[i].getEndPoint(), cv::Scalar(255, 0, 0), 2);
						/*
						cv::putText(image_candidates, std::to_string(i), contours[i][0], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));
						std::cout << "perpencidularLineStartEndPoints[" << i << "][0].x = " << perpencidularLineStartEndPoints[i][0].x << ", perpencidularLineStartEndPoints[" << i << "][0].y = " << perpencidularLineStartEndPoints[i][0].y << std::endl;
						std::cout << "contour[" << i << "][0] = " << contour[i][0] << ", contour[" << i << "][1] = " << contour[i][1] <<
												 ", contour[" << i << "][2] = " << contour[i][2] << ", contour[" << i << "][3] = " << contour[i][3] << std::endl;
						*/
					} else {
						deletedContours[i] = true;
					}
				} else {
					deletedContours[i] = true;
				}
			} else {
				deletedContours[i] = true;
			}
			//} // End index else
		}
	}
}

void filterContours(int keylinesInContours_size,
										std::vector<bool> &deletedContours,
										std::vector<int> &start_barcode_pos,
										std::vector<int> &end_barcode_pos,
										std::vector<cv::line_descriptor::KeyLine> &keylines,
										std::vector<std::vector<int>> &support_scores,
										std::vector<std::vector<cv::Point>> &contours_barcodes,
										int inSegmentXDistance,
										int inSegmentYDistance) {

	int length;
	int keylines_i_lineLength;
	cv::Point2f pt_i;
	cv::Point2f pt_j;

	for(int i = 0; i < keylinesInContours_size; i++) {
		if(true == deletedContours[i]) {
			continue;
		}

		length = end_barcode_pos[i] - start_barcode_pos[i];
		keylines_i_lineLength = keylines[i].lineLength;
		if(false == deletedContours[i]) {
			//if(0 < length) {
			//if((length / keylines_i_lineLength) < 10) {
			for(int j = 0; j < keylinesInContours_size; j++) {
				if(i == j) {
					continue;
				}
				if(true == deletedContours[j]) {
					continue;
				}

				pt_i = keylines[i].pt;
				pt_j = keylines[j].pt;
				if(std::abs(pt_i.x - pt_j.x) < inSegmentXDistance) {
					if(std::abs(pt_i.y - pt_j.y) < inSegmentYDistance) {
						if(support_scores[i] >= support_scores[j]) {
							// Remove contour j
							contours_barcodes[j].clear();
							deletedContours[j] = true;
						} else {
							// Remove contour i
							contours_barcodes[i].clear();
							deletedContours[i] = true;
						}
					}
				}
			}
		}
	}
}

cv::Point contourCenter(const std::vector<cv::Point>& contour) {
	if (0 == contour.size()) {
		return(cv::Point(-1, -1));
	}

	cv::Point contourCenter(0, 0);
	for(const auto& point : contour) {
		contourCenter += point;
	}
	contourCenter = cv::Point(contourCenter.x / contour.size(), contourCenter.y / contour.size());

	return(contourCenter);
}

std::vector<cv::Point> scaleContour(double scalingFactor,
																		const std::vector<cv::Point>& contour,
																		const cv::Mat &image) {
	cv::Point center = contourCenter(contour);

	std::vector<cv::Point> scaledContour(contour.size());
	std::transform(contour.begin(), contour.end(), scaledContour.begin(),
		[&](const cv::Point& point) {
			return scalingFactor * (point - center) + center;
		}
	);

	return(scaledContour);
}

cv::Rect clamRoiToImage(cv::Rect roi, const cv::Mat& image) {
	cv::Rect clampedRoi = roi;

	if(0 > clampedRoi.x) {
		clampedRoi.x = 0;
	}
	if(image.cols < clampedRoi.y) {
		clampedRoi.y = image.cols;
	}
	if(image.cols < clampedRoi.x + clampedRoi.width) {
		clampedRoi.width = image.cols - clampedRoi.x;
	}

	if(0 > clampedRoi.y) {
		clampedRoi.y = 0;
	}
	if(image.rows < clampedRoi.y) {
		clampedRoi.y = image.rows;
	}
	if(image.rows < clampedRoi.y + clampedRoi.height) {
		clampedRoi.height = image.rows - clampedRoi.y;
	}

	return(clampedRoi);
}

void decodeBarcode(int keylinesInContours_size,
									 std::vector<bool> &deletedContours,
									 std::vector<std::vector<cv::Point>> &contours_barcodes,
									 cv::Mat & image_greyscale,
									 cv::Mat & image_barcodes) {
	zbar::ImageScanner scanner;
	std::vector<cv::Point> scaledContour;
	cv::Rect roi;
	cv::Mat croppedImage;
	std::string barcode;

	scanner.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 1);

	std::vector<std::vector<cv::Point>> scaledCroppedContours;
	for(int i = 0; i < keylinesInContours_size; i++) {
		if(true == deletedContours[i]) {
			continue;
		}

		scaledContour = scaleContour(1.5, contours_barcodes[i], image_barcodes);
		roi = cv::boundingRect(scaledContour);
		roi = clamRoiToImage(roi, image_barcodes);
		std::vector<cv::Point> scaledCroppedContour = {cv::Point(roi.x, roi.y),
																									 cv::Point(roi.x + roi.width, roi.y),
																									 cv::Point(roi.x + roi.width, roi.y + roi.height),
																									 cv::Point(roi.x, roi.y + roi.height)};
		scaledCroppedContours.push_back(scaledCroppedContour);

		image_greyscale(roi).copyTo(croppedImage);
		zbar::Image zbar_image(croppedImage.cols, croppedImage.rows, "Y800", croppedImage.data, croppedImage.cols * croppedImage.rows);
		scanner.scan(zbar_image);

    // Use first detected barcode reading from image
    zbar::Image::SymbolIterator symbol = zbar_image.symbol_begin();
    barcode = symbol->get_data();
    cv::putText(image_barcodes, barcode, contours_barcodes[i][0], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));
  }
  cv::drawContours(image_barcodes, contours_barcodes, -1, cv::Scalar(255, 0, 0));
  cv::drawContours(image_barcodes, scaledCroppedContours, -1, cv::Scalar(0, 0, 255), 1);
  cv::imwrite("debug-barcodes.jpg", image_barcodes);
}
