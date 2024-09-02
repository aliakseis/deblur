// deblur.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    try {
    // Initialize OpenVINO core
    ov::Core ie;

    // Read the model
    std::shared_ptr<ov::Model> model = ie.read_model("D:/solutions/DeblurGANv2/model/deblurgan-v2.xml");

    // Compile the model for a specific device
    ov::CompiledModel compiled_model = ie.compile_model(model, "CPU");

    // Get input and output layers
    ov::Output<const ov::Node> model_input_layer = compiled_model.input(0);
    //ov::Output<const ov::Node> model_output_layer = compiled_model.output(0);

    // Image filename (local path or URL)
    //std::string filename = "https://raw.githubusercontent.com/VITA-Group/DeblurGANv2/master/test_img/000027.png";
    std::string filename = "D:/solutions/DeblurGANv2/000027.png";

    // Load the input image
    cv::Mat image = cv::imread(filename);

    // Convert the image to RGB format if needed
    if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // Get input shape
    auto input_shape = model_input_layer.get_shape();
    //size_t N = input_shape[0];
    size_t C = input_shape[1];
    size_t H = input_shape[2];
    size_t W = input_shape[3];

    // Resize the image to meet network expected input sizes
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(W, H));

    // Convert image to float32 precision and normalize in [-1, 1] range
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 127.5, -1.0);

    cv::Mat s[3];

    split(resized_image, s);

    // Add batch dimension to input image tensor
    std::vector<float> input_image(C * H * W);
    for (int i = 0; i < C; i++) {
        std::memcpy(input_image.data() + i * H * W, s[i].data, H * W * sizeof(float));
    }
    // Inference
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    infer_request.set_input_tensor(ov::Tensor(ov::element::from<float>(), input_shape,
        //resized_image.data
        input_image.data()
    ));
    infer_request.infer();

    // Get the result
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    float* result_data = output_tensor.data<float>();

    for (int i = 0; i < C; i++) {
        s[i] = cv::Mat(H, W, CV_32F, result_data + i * H * W);
    }

    // Convert the result to an image shape and [0, 255] range
    //cv::Mat result_image(H, W, CV_32FC3, result_data);
    cv::Mat result_image;
    merge(s, 3, result_image);
    result_image.convertTo(result_image, CV_8U, 255.0);

    // Resize to the original image size and convert to original u8 precision
    cv::Mat resized_result_image;
    cv::resize(result_image, resized_result_image, cv::Size(image.cols, image.rows));

    cv::cvtColor(resized_result_image, resized_result_image, cv::COLOR_RGB2BGR);

    // Save the result image
    std::string savename = "deblurred.png";
    cv::imwrite(savename, resized_result_image);
    }
    catch (const std::exception& ex) {
        std::cerr << "Exception " << typeid(ex).name() << ": " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
}
