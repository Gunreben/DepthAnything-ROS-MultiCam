#include "depth_anything/multi_depth_anything_node.hpp"

namespace
{
template <class T>
bool update_param(
  const std::vector<rclcpp::Parameter> & params, const std::string & name, T & value)
{
  const auto itr = std::find_if(
    params.cbegin(), params.cend(),
    [&name](const rclcpp::Parameter & p) { return p.get_name() == name; });

  if (itr == params.cend()) {
    return false;
  }

  value = itr->template get_value<T>();
  return true;
}
}  // namespace

namespace depth_anything
{

MultiDepthAnythingNode::MultiDepthAnythingNode(const rclcpp::NodeOptions & options)
: Node("multi_depth_anything_node", options)
{
  using std::placeholders::_1;

  // Parameters
  set_param_res_ =
    this->add_on_set_parameters_callback(std::bind(&MultiDepthAnythingNode::onSetParam, this, _1));

  node_param_.onnx_path = declare_parameter<std::string>("onnx_path");
  node_param_.precision = declare_parameter<std::string>("precision", "fp16");

  // Subscribers (example: front-left & front-right)
  sub_image_fl_ = image_transport::create_subscription(
    this, "/camera_image/Cam_FL", std::bind(&MultiDepthAnythingNode::onDataFL, this, _1),
    "raw", rmw_qos_profile_sensor_data);

  sub_image_fr_ = image_transport::create_subscription(
    this, "/camera_image/Cam_FR", std::bind(&MultiDepthAnythingNode::onDataFR, this, _1),
    "raw", rmw_qos_profile_sensor_data);

  // Publishers
  pub_depth_image_fl_ = create_publisher<sensor_msgs::msg::Image>("/camera_image/Cam_FL_depth", 1);
  pub_depth_image_fr_ = create_publisher<sensor_msgs::msg::Image>("/camera_image/Cam_FR_depth", 1);

  // Initialize the TensorRT model once (instead of once per camera)
  initModel();
  RCLCPP_INFO(get_logger(), "MultiDepthAnythingNode initialized for multiple camera streams");
}

void MultiDepthAnythingNode::initModel()
{
  // Adjust parameters to suit your build
  std::string calibType = "MinMax";
  int dla = -1;
  bool first = false, last = false, prof = false;
  double clip = 0.0;
  tensorrt_common::BuildConfig build_config(calibType, dla, first, last, prof, clip);

  // Example: batch size = 1
  int batch = 1;
  tensorrt_common::BatchConfig batch_config{1, batch / 2, batch};

  bool use_gpu_preprocess = false;
  std::string calibration_images = "calibration_images.txt";
  const size_t workspace_size = (1 << 30);  // 1 GiB

  // Create your TRT engine once
  node_param_.precision = "fp16";
  RCLCPP_INFO(get_logger(), "precision: %s", node_param_.precision.c_str());

  trt_depth_anything_ = std::make_unique<tensorrt_depth_anything::TrtDepth_Anything>(
    node_param_.onnx_path, node_param_.precision, build_config, use_gpu_preprocess,
    calibration_images, batch_config, workspace_size);

  is_initialized_ = false;  // Will set to true after first inference
  RCLCPP_INFO(get_logger(), "TensorRT model loaded from: %s", node_param_.onnx_path.c_str());
}

void MultiDepthAnythingNode::onDataFL(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  runDepthInference(msg, pub_depth_image_fl_, "Cam_FL");
}

void MultiDepthAnythingNode::onDataFR(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  runDepthInference(msg, pub_depth_image_fr_, "Cam_FR");
}

void MultiDepthAnythingNode::runDepthInference(
  const sensor_msgs::msg::Image::ConstSharedPtr & msg,
  const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr & publisher,
  const std::string & camera_name
)
{
  // Convert ROS Image to OpenCV
  cv_bridge::CvImagePtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "[%s] cv_bridge exception: %s", camera_name.c_str(), e.what());
    return;
  }

  const auto width = in_image_ptr->image.cols;
  const auto height = in_image_ptr->image.rows;

  // Initialize TRT buffers on the first run
  if (!is_initialized_) {
    trt_depth_anything_->initPreprocessBuffer(width, height);
    is_initialized_ = true;
  }

  // Run inference
  std::vector<cv::Mat> input_images;
  input_images.push_back(in_image_ptr->image);
  trt_depth_anything_->doInference(input_images);

  // Get depth result and resize to original resolution
  cv::Mat depth_image = trt_depth_anything_->getDepthImage();
  cv::resize(depth_image, depth_image, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);

  // Convert CV to ROS image
  cv_bridge::CvImage cv_img;
  cv_img.image = depth_image;
  cv_img.encoding = "mono8";

  sensor_msgs::msg::Image depth_msg;
  cv_img.toImageMsg(depth_msg);
  depth_msg.header = msg->header;

  // Publish
  publisher->publish(depth_msg);
}

rcl_interfaces::msg::SetParametersResult MultiDepthAnythingNode::onSetParam(
  const std::vector<rclcpp::Parameter> & params)
{
  rcl_interfaces::msg::SetParametersResult result;
  try {
    auto & p = node_param_;
    // Update parameters if provided
    update_param(params, "onnx_path", p.onnx_path);
    update_param(params, "precision", p.precision);

  } catch (const rclcpp::exceptions::InvalidParameterTypeException & e) {
    result.successful = false;
    result.reason = e.what();
    return result;
  }

  // Re-initialize model if relevant params changed
  // (Optional: you could check if onnx_path or precision changed before re-initializing)
  initModel();

  result.successful = true;
  result.reason = "success";
  return result;
}

}  // namespace depth_anything

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(depth_anything::MultiDepthAnythingNode)
