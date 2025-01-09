#ifndef DEPTH_ANYTHING__MULTI_DEPTH_ANYTHING_NODE_HPP_
#define DEPTH_ANYTHING__MULTI_DEPTH_ANYTHING_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <memory>
#include <string>
#include <vector>

#include "tensorrt_depth_anything/tensorrt_depth_anything.hpp"


namespace depth_anything
{

class MultiDepthAnythingNode : public rclcpp::Node
{
public:
  explicit MultiDepthAnythingNode(const rclcpp::NodeOptions & options);

private:
  // Parameter struct
  struct NodeParam
  {
    std::string onnx_path;
    std::string precision;
  } node_param_;

  // Callbacks
  void onDataFL(const sensor_msgs::msg::Image::ConstSharedPtr msg);
  void onDataFR(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  // Parameter callback
  rcl_interfaces::msg::SetParametersResult onSetParam(const std::vector<rclcpp::Parameter> & params);

  // Subscribers
  image_transport::Subscriber sub_image_fl_;
  image_transport::Subscriber sub_image_fr_;

  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_image_fl_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_image_fr_;

  // TensorRT Inference
  std::unique_ptr<tensorrt_depth_anything::TrtDepth_Anything> trt_depth_anything_;
  bool is_initialized_ {false};

  // Helper to init model
  void initModel();

  // Reusable inference function
  void runDepthInference(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg,
    const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr & publisher,
    const std::string & camera_name
  );

  // Param handling
  OnSetParametersCallbackHandle::SharedPtr set_param_res_;
};

}  // namespace depth_anything

#endif  // DEPTH_ANYTHING__MULTI_DEPTH_ANYTHING_NODE_HPP_
