#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <iostream>
#include <chrono>
#include <memory>

int main(int argc, char** argv) {
  if(argc != 2) {
    std::cout << "Usage: display_image ImageToLoadAndDisplay" << std::endl;
    return(-1);
  }

  cv::Mat image;
  image = cv::imread(argv[1], cv::IMREAD_COLOR);
  //cv::resize(image, image, cv::Size(1080, 960));
  //cv::resize(image, image, cv::Size(), 0.4, 0.4);
  //cv::resize(image, image, cv::Size(), 1.0, 1.0);

  if(!image.data) {
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
  LSD.detect(image, keylines, 2, 1);
  auto end = std::chrono::steady_clock::now();
  std::cout << "LSD: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

  std::vector<cv::Point2f> lookup;
  std::vector<std::vector<cv::Point>> contours;

	// Draw the detected lines on the image
	cv::Mat image_lines;
	image.copyTo(image_lines);

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
	image.copyTo(image_candidates);

	start = std::chrono::steady_clock::now();
	#pragma omp parallel for
	for(int i = 0; i < keylinesInContours_size; i++) {
		support_candidates_pos[i] = std::distance(support_scores[i].begin(), std::max_element(support_scores[i].begin(), support_scores[i].end()));
		//std::cout << "support_candidate_pos[" << i << "] = " << support_candidates_pos[i] << std::endl;
		support_candidates[i] = support_scores[i][std::distance(support_scores[i].begin(), std::max_element(support_scores[i].begin(), support_scores[i].end()))];
		//std::cout << "max_support_candidates[" << i << "] = " << support_candidates[i] << std::endl;
		//std::cout << "min_support_candidates[" << i << "] = " << support_scores[i][std::distance(support_scores[i].begin(), std::min_element(support_scores[i].begin(), support_scores[i].end()))] << std::endl;

		if(0 < support_candidates[i]) {
			register std::shared_ptr<cv::line_descriptor::KeyLine> kl = keylinesInContours[i][support_candidates_pos[i]];
			cv::line(image_candidates, kl->getStartPoint(), kl->getEndPoint(), cv::Scalar(0, 0, 255));
		}
	}
	end = std::chrono::steady_clock::now();
	std::cout << "Select s_cand: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;

	std::cout << "Total time: " << std::chrono::duration <double, std::milli> (end - total_start).count() << " ms" << std::endl;

	cv::imwrite("debug-candidate-segments.jpg", image_candidates);
	cv::namedWindow("Image with candidate segments", cv::WINDOW_NORMAL);
	cv::resizeWindow("Image with candidate segments", 600, 600);
	cv::imshow("Image with candidate segments", image_candidates);

	cv::waitKey(0);

	return(0);
}
