#pragma once
#include <filesystem>
#include <spdlog/spdlog.h>

template <typename T>
void Engine<T>::transformOutput(std::vector<std::vector<std::vector<T>>> &input, std::vector<std::vector<T>> &output) {
    if (input.size() != 1) {
        auto msg = "The feature vector has incorrect dimensions!";
        spdlog::error(msg);
        throw std::logic_error(msg);
    }

    output = std::move(input[0]);
}

template <typename T> void Engine<T>::transformOutput(std::vector<std::vector<std::vector<T>>> &input, std::vector<T> &output) {
    if (input.size() != 1 || input[0].size() != 1) {
        auto msg = "The feature vector has incorrect dimensions!";
        spdlog::error(msg);
        throw std::logic_error(msg);
    }

    output = std::move(input[0][0]);
}

template <typename T>
int Engine<T>::constrainToMultipleOf(const float x, const int multiple, const int min_val, const int max_val) {
    int y;
    y = std::round(x / multiple) * multiple;
    if (max_val > 0 && y > max_val)
        y = std::floor(x / multiple) * multiple;
    if (y < min_val)
        y = std::ceil(x / multiple) * multiple;
    return y;
}

template <typename T>
void Engine<T>::extractAndResizeROI(const cv::Mat &input, cv::Mat &output, size_t roi_h, size_t roi_w, size_t height, size_t width) {
    cv::Mat roi = input(cv::Rect(0, 0, roi_w, roi_h));
    cv::resize(roi, output, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
}

template <typename T>
cv::cuda::GpuMat Engine<T>::resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat &input, size_t height, size_t width,
                                                                int &unpad_h, int &unpad_w, const int multiple, const cv::Scalar &bgcolor) {
    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    unpad_w = r * input.cols;
    unpad_w = constrainToMultipleOf(unpad_w, multiple, width, -1);
    unpad_h = r * input.rows;
    unpad_h = constrainToMultipleOf(unpad_h, multiple, height, -1);
    // cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    // new matrix with same type as input
    cv::cuda::GpuMat re(unpad_h, unpad_w, input.type());
    cv::cuda::resize(input, re, re.size());
    // cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);
    cv::cuda::GpuMat out(height, width, input.type(), bgcolor);
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

template <typename T> void Engine<T>::getDeviceNames(std::vector<std::string> &deviceNames) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    for (int device = 0; device < numGPUs; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        deviceNames.push_back(std::string(prop.name));
    }
}

template <typename T> std::string Engine<T>::serializeEngineOptions(const Options &options, const std::string &onnxModelPath) {
    const auto filenamePos = onnxModelPath.find_last_of('/') + 1;
    std::string engineName = onnxModelPath.substr(filenamePos, onnxModelPath.find_last_of('.') - filenamePos) + ".engine";

    // Add the GPU device name to the file to ensure that the model is only used
    // on devices with the exact same GPU
    std::vector<std::string> deviceNames;
    getDeviceNames(deviceNames);

    if (static_cast<size_t>(options.deviceIndex) >= deviceNames.size()) {
        auto msg = "Error, provided device index is out of range!";
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    auto deviceName = deviceNames[options.deviceIndex];
    // Remove spaces from the device name
    deviceName.erase(std::remove_if(deviceName.begin(), deviceName.end(), ::isspace), deviceName.end());

    engineName += "." + deviceName;

    // Serialize the specified options into the filename
    if (options.precision == Precision::FP16) {
        engineName += ".fp16";
    } else if (options.precision == Precision::FP32) {
        engineName += ".fp32";
    } else {
        engineName += ".int8";
    }

    engineName += "." + std::to_string(options.maxBatchSize);
    engineName += "." + std::to_string(options.optBatchSize);

    spdlog::info("Engine name: {}", engineName);
    return engineName;
}

template <typename T>
cv::cuda::GpuMat Engine<T>::blobFromGpuMats(const std::vector<cv::cuda::GpuMat> &batchInput) {
   
    CHECK(!batchInput.empty())
    CHECK(batchInput[0].channels() == 3 || batchInput[0].channels() == 1)
    
    cv::cuda::GpuMat gpuDst(1, batchInput[0].rows * batchInput[0].cols * batchInput.size(), batchInput[0].type());

    size_t width = batchInput[0].cols * batchInput[0].rows;
    int elemSize = (int)CV_ELEM_SIZE(CV_MAT_DEPTH(batchInput[0].type()));
    for (size_t img = 0; img < batchInput.size(); ++img) {
        if (batchInput[0].channels() == 3) {
            std::vector<cv::cuda::GpuMat> input_channels{
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_MAT_DEPTH(batchInput[0].type()), &(gpuDst.ptr()[elemSize*(0 + width * 3 * img)])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_MAT_DEPTH(batchInput[0].type()), &(gpuDst.ptr()[elemSize*(width + width * 3 * img)])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_MAT_DEPTH(batchInput[0].type()), &(gpuDst.ptr()[elemSize*(width * 2 + width * 3 * img)]))};
            cv::cuda::split(batchInput[img], input_channels); // HWC -> CHW
        } else { // 1 channel
            std::vector<cv::cuda::GpuMat> input_channels{
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_MAT_DEPTH(batchInput[0].type()), &(gpuDst.ptr()[elemSize*(0 + width * img)]))};
            cv::cuda::split(batchInput[img], input_channels); // HWC -> CHW
        }
    }
    return gpuDst;
}

template <typename T>
cv::cuda::GpuMat Engine<T>::blobFromGpuMats(const std::vector<cv::cuda::GpuMat> &batchInput, const std::array<float, 3> &subVals,
                                            const std::array<float, 3> &divVals, bool normalize, bool swapRB) {
   
    CHECK(!batchInput.empty())
    CHECK(batchInput[0].channels() == 3)
    
    cv::cuda::GpuMat gpuDst(1, batchInput[0].rows * batchInput[0].cols * batchInput.size(), batchInput[0].type());

    size_t width = batchInput[0].cols * batchInput[0].rows;
    int elemSize = (int)CV_ELEM_SIZE(CV_MAT_DEPTH(batchInput[0].type()));
    if (swapRB) {
        for (size_t img = 0; img < batchInput.size(); ++img) {
            std::vector<cv::cuda::GpuMat> input_channels{
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_MAT_DEPTH(batchInput[0].type()), &(gpuDst.ptr()[elemSize*(width * 2 + width * 3 * img)])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_MAT_DEPTH(batchInput[0].type()), &(gpuDst.ptr()[elemSize*(width + width * 3 * img)])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_MAT_DEPTH(batchInput[0].type()), &(gpuDst.ptr()[elemSize*(0 + width * 3 * img)]))};
            cv::cuda::split(batchInput[img], input_channels); // HWC -> CHW
        }
    } else {
        for (size_t img = 0; img < batchInput.size(); ++img) {
            std::vector<cv::cuda::GpuMat> input_channels{
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_MAT_DEPTH(batchInput[0].type()), &(gpuDst.ptr()[elemSize*(0 + width * 3 * img)])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_MAT_DEPTH(batchInput[0].type()), &(gpuDst.ptr()[elemSize*(width + width * 3 * img)])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_MAT_DEPTH(batchInput[0].type()), &(gpuDst.ptr()[elemSize*(width * 2 + width * 3 * img)]))};
            cv::cuda::split(batchInput[img], input_channels); // HWC -> CHW
        }
    }
    cv::cuda::GpuMat mfloat;
    if (normalize) {
        // [0.f, 1.f]
        gpuDst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
    } else {
        // [0.f, 255.f]
        gpuDst.convertTo(mfloat, CV_32FC3);
    }

    // Apply scaling and mean subtraction
    cv::cuda::subtract(mfloat, cv::Scalar(subVals[0], subVals[1], subVals[2]), mfloat, cv::noArray(), -1);
    cv::cuda::divide(mfloat, cv::Scalar(divVals[0], divVals[1], divVals[2]), mfloat, 1, -1);

    return mfloat;
}

template <typename T> void Engine<T>::clearGpuBuffers() {
    if (!m_buffers.empty()) {
        // Free GPU memory of outputs
        const auto numInputs = m_inputDims.size();
        for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
            Util::checkCudaErrorCode(cudaFree(m_buffers[outputBinding]));
        }
        m_buffers.clear();
    }
}

template <typename T>
cv::cuda::GpuMat Engine<T>::blobFromMat(const cv::Mat &batchInput, const std::array<float, 3> &subVals,
                                            const std::array<float, 3> &divVals, bool normalize) {

    // std::vector<cv::Mat> channels;
    // cv::split(batchInput, channels);
    // // Stretch one-channel images to vector
    // for (auto &tmp : channels) {
    //     tmp = tmp.reshape(1, 1);
    // }

    // cv::Mat batchInputReshaped;
    // cv::hconcat(channels, batchInputReshaped);

    // batchInputReshaped = cv::Mat::zeros(1, batchInput.size[0] * batchInput.size[1] * batchInput.size[2] * batchInput.size[3], CV_32F);

    // std::cout << "reshaped size: " << batchInputReshaped.size[0] << ", " << batchInputReshaped.size[1] << std::endl;

    // cv::cuda::GpuMat mfloat; //(batchInput.size(), batchInput.type());
    // cv::cuda::GpuMat gpuDst(1, batchInput.size[0] * batchInput.size[1] * batchInput.size[2], CV_32FC1);
    // gpuDst.upload(batchInput);
    // std::cout << "input type: " << batchInput.type() << std::endl;
    cv::Mat batchInputFlat = batchInput.reshape(1, {1, batchInput.size[0] * batchInput.size[1] * batchInput.size[2] * batchInput.size[3]});
    // batchInputFlat.convertTo(batchInputFlat, CV_32FC1);
    // cv::cuda::GpuMat gpuDst(batchInput.reshape(1, {1, batchInput.size[0] * batchInput.size[1] * batchInput.size[2] * batchInput.size[3]}));
    // gpuDst.upload(batchInputFlat);
    cv::cuda::GpuMat gpuDst(batchInputFlat);
    // std::cout << "GPU mat element size: " << gpuDst.elemSize() << std::endl;


    // size_t width = batchInput[0].cols * batchInput[0].rows;
    // for (size_t img = 0; img < batchInput.size(); img++) {
    //     std::vector<cv::cuda::GpuMat> input_channels{
    //         cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_32FC1, &(gpuDst.ptr()[0 + width * 3 * img])),
    //         cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_32FC1, &(gpuDst.ptr()[width + width * 3 * img])),
    //         cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_32FC1, &(gpuDst.ptr()[width * 2 + width * 3 * img]))};
    //     cv::cuda::split(batchInput[img], input_channels); // HWC -> CHW
    // }

    // cv::cuda::GpuMat mfloat;
    // if (normalize) {
    //     // [0.f, 1.f]
    //     gpuDst.convertTo(mfloat, CV_32FC1, 1.f / 255.f);
    // } else {
    //     // [0.f, 255.f]
    //     gpuDst.convertTo(mfloat, CV_32FC1);
    // }

    // // Apply scaling and mean subtraction
    // cv::cuda::subtract(mfloat, cv::Scalar(subVals[0], subVals[1], subVals[2]), mfloat, cv::noArray(), -1);
    // cv::cuda::divide(mfloat, cv::Scalar(divVals[0], divVals[1], divVals[2]), mfloat, 1, -1);

    return gpuDst;
}