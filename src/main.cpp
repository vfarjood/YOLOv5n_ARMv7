#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <math.h>
#include <map>
#include <ctime>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.5;
const float CONFIDENCE_THRESHOLD = 0.6;

class Timer {
private:
    std::chrono::time_point<std::chrono::steady_clock> begin, end;
    std::chrono::duration<float> duration;
public:
    void start() {
        begin = std::chrono::steady_clock::now();
    }
    float stop(){
        end = std::chrono::steady_clock::now();
        duration = end - begin; 
        float second = duration.count();
        return second;
    }
    std::string getCurrentTime(){
        std::stringstream ss;
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
        return ss.str();
    }
};

struct Centroid {
    int id;
    float conf;
    float area;
    cv::Point center;
    cv::Rect box;
    std::vector<cv::Rect> box_history;
    std::vector<cv::Point> position_history;
    cv::Point next_position;

    std::string name;
    int lane_num = 0;
    float distance{0.0};
    float speed{0.0};
};

class Lane {
public:
    std::vector<std::pair<cv::Point, cv::Point>> line;
    Lane(){
        line.emplace_back(std::make_pair(cv::Point(250,0), cv::Point(45 ,360))); // First line from left side of the picture
        line.emplace_back(std::make_pair(cv::Point(267,0), cv::Point(172,360))); // Second line
        line.emplace_back(std::make_pair(cv::Point(282,0), cv::Point(330,360))); // ...
        line.emplace_back(std::make_pair(cv::Point(300,0), cv::Point(475,360)));
        line.emplace_back(std::make_pair(cv::Point(313,0), cv::Point(620,360)));
        line.emplace_back(std::make_pair(cv::Point(330,0), cv::Point(640,270)));
    };
    void draw(cv::Mat const& image){
        for(int i = 0; i < line.size(); i++){
            cv::line(image, line[i].first, line[i].second, cv::Scalar(173.0, 255.0, 47.0), 2);
        }
    }
};

void drawResultOnImage(std::vector<std::string>& input_files, std::vector<std::vector<Centroid>>& vector_of_centeroids)
{
    // std::default_random_engine generator;
    std::mt19937 generator(2019);
    // set uniform distribution for each R,G,B color:
    std::uniform_int_distribution<int> distribution(0, 255);

    Lane line;
    int j=0;
    for(auto const& centroids: vector_of_centeroids)
    {    
        cv::Mat image = cv::imread(input_files[j], 1);
        std::string name = "image-" + std::to_string(j+1) + ".jpg";
        std::string path = "../media/result/" + name;

        // cv::resize(image, image, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
        for(int i = 0; i < centroids.size(); i++)
        {
            cv::Scalar color = cv::Scalar(0.0, 255.0, 255.0);
            color = cv::Scalar(distribution(generator), distribution(generator), distribution(generator));
            std::string label = "[" + centroids[i].name + " : " + cv::format("%.2f",centroids[i].conf) + "]"; // + " L:" + std::to_string(centroids[i].lane_num);
    
            cv::rectangle(image, cv::Point(centroids[i].box.x, centroids[i].box.y), cv::Point(centroids[i].box.x+centroids[i].box.width, centroids[i].box.y+centroids[i].box.height), color, 2);
            cv::putText(image, label, cv::Point(centroids[i].box.x, centroids[i].box.y-5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
        line.draw(image);
        cv::imwrite(path, image);
        ++j;
    }
};
double distanceBetweenPoints(const cv::Point& point1, const cv::Point& point2) 
{
    double distance = sqrt(pow((point2.x - point1.x), 2) + pow((point2.y - point1.y), 2));
    return distance;
};

bool checkTheLine(const cv::Point& center, 
                    const std::pair<cv::Point, cv::Point>& left_line, 
                        const std::pair<cv::Point, cv::Point>& right_line,
                            const cv::Size& image_size)
{
    bool left  = false;
    bool right = false;
    cv::LineIterator left_it(image_size,left_line.first, left_line.second, 8);
    cv::LineIterator rigth_it(image_size,right_line.first, right_line.second, 8);
    for(int i = 0; i < left_it.count; i++, ++left_it)
    {
        cv::Point pt= left_it.pos();
        if((center.y == pt.y) && (center.x >= pt.x))
            left = true;
    }
    for(int i = 0; i < rigth_it.count; i++, ++rigth_it)
    {
        cv::Point pt= rigth_it.pos();
        if((center.y == pt.y) && (center.x <= pt.x))
            right = true;
    }
    if(left && right)
        return true;
    else
        return false;
}

void findLineNumber(std::vector<Centroid> &vehicles, const cv::Size& image_size) 
{
    Lane lanes;
    for (auto &vehicle : vehicles) 
    {
        for(int i = 0; i < lanes.line.size(); i++)
        {
            if(checkTheLine(vehicle.center, lanes.line[i], lanes.line[i+1], image_size))
            {
                vehicle.lane_num = i+1;
            }
        }
    }
};

void findVehiclesTrajectory(std::vector<Centroid> &existingVehicles, std::vector<Centroid> &currentFrameVehicles, const cv::Size& image_size) 
{
    findLineNumber(currentFrameVehicles, image_size);

    for (auto &currentFrameVehicle : currentFrameVehicles) 
    {
        int least_distance_index = 0;
        double least_distance = 100000.0;
        for (int i = 0; i < existingVehicles.size(); i++) 
        {
            if ((existingVehicles[i].lane_num == currentFrameVehicle.lane_num) &&
                (existingVehicles[i].area > currentFrameVehicle.area))
            {
                double distance = distanceBetweenPoints(currentFrameVehicle.center, existingVehicles[i].position_history.back());
                if (distance < least_distance) 
                {
                    least_distance = distance;
                    least_distance_index = i;
                }
            }
        }
        if (least_distance < currentFrameVehicle.box.height * 5)
        {
            const int fps = 1;
            const float speed_factor = (existingVehicles[least_distance_index].center.y / image_size.height) + 1.3;
            existingVehicles[least_distance_index].distance = least_distance;
            existingVehicles[least_distance_index].speed = (least_distance/speed_factor)/fps;
            existingVehicles[least_distance_index].box_history.push_back(currentFrameVehicle.box);
            existingVehicles[least_distance_index].position_history.push_back(currentFrameVehicle.center);
        } 
        else { 
            if(currentFrameVehicle.position_history.size() == 0){
                currentFrameVehicle.position_history.push_back(currentFrameVehicle.center);
            }
            existingVehicles.push_back(currentFrameVehicle);
        }
    }
};

int main(int argc, char** argv )
{
    std::cout << "OpenCV version : " << CV_VERSION << std::endl;

    Timer time, total_time;
    float total_image_load_time, 
          model_loading_time,
          detection_time,
          tracking_time = 0;

    total_time.start();
    

    std::vector<std::vector<Centroid>> vector_of_centeroids; // keeping all vehicles from all input images.
    vector_of_centeroids.reserve(5);

    // Load class names:
    std::vector<std::string> class_names;
    class_names.reserve(5);
    std::ifstream ifs(std::string("../models/classes.txt").c_str());
    std::string line;
    while (getline(ifs, line)) {class_names.push_back(line);} 
    
    // load the neural network model:
    time.start();
    cv::dnn::Net model = cv::dnn::readNetFromONNX("../models/best.onnx");
    model_loading_time = time.stop();

    // load input image list:
    std::vector<std::string> input_files= {"../media/traffic/image2.jpg", "../media/traffic/image3.jpg"};
    cv::Size image_size;

    // Object Detection:
    for (std::string const& file : input_files)
    {
        // Load the image:
        time.start();
        cv::Mat image = cv::imread(file, 1);
        total_image_load_time += time.stop();
        image_size = image.size();

        time.start();
        // create blob from image: 
        cv::Mat input;   
        cv::dnn::blobFromImage(image, input, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
        std::vector<std::string> output_layer_names = model.getUnconnectedOutLayersNames();
    
        // set the blob to the model:
        model.setInput(input);
    
        // forward pass through the model to carry out the detection:
        cv::Mat output;
        model.forward(output);
    
        float x_factor = image.cols / INPUT_WIDTH;
        float y_factor = image.rows / INPUT_HEIGHT;
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        std::vector<Centroid> centroids;
        centroids.reserve(5);

        cv::Mat detectedMat(output.size[1], output.size[2], CV_32F, output.ptr<float>());

        for (int i = 0; i < detectedMat.rows; ++i) 
        {
            float confidence = detectedMat.at<float>(i, 4);
            if (confidence >= CONFIDENCE_THRESHOLD) {
    
                float* classes_scores = &detectedMat.at<float>(i, 5);
                cv::Point class_id;
                double max_class_score;
                cv::Mat scores(1, class_names.size(), CV_32FC1, classes_scores);
                cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                
                if (max_class_score > SCORE_THRESHOLD) {
    
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);
    
                    int x = static_cast<int>(detectedMat.at<float>(i, 0));
                    int y = static_cast<int>(detectedMat.at<float>(i, 1));
                    int w = static_cast<int>(detectedMat.at<float>(i, 2));
                    int h = static_cast<int>(detectedMat.at<float>(i, 3));
    
                    boxes.push_back(cv::Rect((x - w / 2)*x_factor,
                                             (y - h / 2)*y_factor,
                                             (w*x_factor),
                                             (h*y_factor)));
                }
            }
        }
    
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
        for (int i = 0; i < nms_result.size(); i++) 
        {
            int idx = nms_result[i];
            Centroid object;
            object.id = class_ids[idx];
            object.conf = confidences[idx];
            object.name = class_names[class_ids[idx]];
            object.box  = boxes[idx];
            object.area = (object.box.width * object.box.height);
            object.center     = cv::Point((object.box.x + (object.box.width/2)), (object.box.y + (object.box.height/2)));
            centroids.push_back(object);
        }
        vector_of_centeroids.push_back(centroids);

    }
    detection_time = time.stop();

    // Find Vehicles Trajectory:
    time.start();
    std::vector<Centroid> tracking_vehicles;
    bool first_image = true;
    for(auto & current_vehicles: vector_of_centeroids)
    {
        if (first_image == true) 
        {
            for (auto &vehicle : current_vehicles) 
            {
                vehicle.position_history.push_back(vehicle.center);
                tracking_vehicles.push_back(vehicle);
            }
            findLineNumber(tracking_vehicles, image_size);
            first_image = false;
        } 
        else 
        {
            findVehiclesTrajectory(tracking_vehicles, current_vehicles, image_size);
        }
    }
    tracking_time = time.stop();
    

    // create result file:
    std::ofstream result_file ("result.txt");
    if (!result_file.is_open())
    {
        std::cout << "Unable to open file";
    }
    Timer current;
    result_file << "*** "<< current.getCurrentTime() << " ***\n";

    // Calculate average traffic flow speed:
    float average_flow_speed = [&]() {
        float avg = 0;
        int count = 0;
        for(auto it = tracking_vehicles.begin(); it != tracking_vehicles.end(); it++){
            if(it->position_history.size() > 1){
                avg += it->speed;
                count++;
            }
        }
        return (avg/count);
    }();

    // Find the lane with the highest flow:
    float highest_lane_flow = [&]() {
        std::map<int, int> frequency_map;
        int max_frequency = 0;
        int most_frequent_element = 0;
        for(auto it = tracking_vehicles.begin(); it != tracking_vehicles.end(); it++){
            int value = ++frequency_map[it->lane_num];
            if (value > max_frequency)
            {
                max_frequency = value;
                most_frequent_element = it->lane_num;
            }
        }
        return most_frequent_element;
    }();

    // check the existance of the traffic:
    if(tracking_vehicles.size()<10)
        result_file << "Traffic Status: "  << "Low" << "\n";
    if( (tracking_vehicles.size()>=10) && (tracking_vehicles.size()<=20) )
        result_file << "Traffic Status: "  << "Medium" << "\n";
    if(tracking_vehicles.size()>20)
        result_file << "Traffic Status: "  << "High" << "\n";

    result_file << "Number of cars:     "  << tracking_vehicles.size() << "\n";
    result_file << "Average flow speed: "  << average_flow_speed << "\n";
    result_file << "Highest flow in lane number: "  << highest_lane_flow << "\n";

    result_file << "*****************" << "\n";


    // sort the vecotr of vehicle based on their lane number:
    std::sort(tracking_vehicles.begin(), tracking_vehicles.end(), 
                [](const Centroid& v1, const Centroid& v2) 
                    {return v1.lane_num < v2.lane_num;});

    // write quantitative information into result file:
    for(int num=0; num < tracking_vehicles.size(); num++)
    {
        result_file << "Vehicle Number: "<< num << "\n";
        result_file << "Vehicle Type: "  << tracking_vehicles[num].name << "\n";
        result_file << "Lane Number: "   << tracking_vehicles[num].lane_num << "\n";
        result_file << "Speed: "         << tracking_vehicles[num].speed << "\n";
        result_file << "Positions: "     << "\n";
        for(int i = 0; i < tracking_vehicles[num].position_history.size(); i++){
            result_file << "    frame["<< (i+1) << "]: "<<tracking_vehicles[num].position_history[i] << "\n";
        }
    result_file << "---------------------" << "\n";
    }

    std::cout << "Computation Time: " << "\n";
    std::cout << "  -Image Loading: " << total_image_load_time << "\n";
    std::cout << "  -Model Loading: " << model_loading_time << "\n";
    std::cout << "  -Detection    : " << detection_time << "\n";
    std::cout << "  -Tracking     : " << tracking_time << "\n";
    std::cout << "  -Total        : " << total_time.stop() << "\n";

    drawResultOnImage(input_files, vector_of_centeroids);

    return 0;
}
