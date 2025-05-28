#pragma once
#include <filesystem>
#include <spdlog/spdlog.h>
#include "util/Util.h"

template <typename T>
bool Engine<T>::runInference(const std::vector<std::vector<cv::cuda::GpuMat>> &inputs,
                             std::vector<std::vector<std::vector<T>>> &featureVectors) {
    // First we do some error checking
    if (inputs.empty() || inputs[0].empty()) {
        spdlog::error("Provided input vector is empty!");
        return false;
    }

    const auto numInputs = m_inputDims.size();
    if (inputs.size() != numInputs) {
        spdlog::error("Incorrect number of inputs provided!");
        return false;
    }

    // Ensure the batch size does not exceed the max
    if (inputs[0].size() > static_cast<size_t>(m_options.maxBatchSize)) {
        spdlog::error("===== Error =====");
        spdlog::error("The batch size is larger than the model expects!");
        spdlog::error("Model max batch size: {}", m_options.maxBatchSize);
        spdlog::error("Batch size provided to call to runInference: {}", inputs[0].size());
        return false;
    }

    // Ensure that if the model has a fixed batch size that is greater than 1, the
    // input has the correct length
    if (m_inputBatchSize != -1 && inputs[0].size() != static_cast<size_t>(m_inputBatchSize)) {
        spdlog::error("===== Error =====");
        spdlog::error("The batch size is different from what the model expects!");
        spdlog::error("Model batch size: {}", m_inputBatchSize);
        spdlog::error("Batch size provided to call to runInference: {}", inputs[0].size());
        return false;
    }

    const auto batchSize = static_cast<int32_t>(inputs[0].size());
    // Make sure the same batch size was provided for all inputs
    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i].size() != static_cast<size_t>(batchSize)) {
            spdlog::error("===== Error =====");
            spdlog::error("The batch size is different for each input!");
            return false;
        }
    }

    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    std::vector<cv::cuda::GpuMat> preprocessedInputs;

    // Preprocess all the inputs
    for (size_t i = 0; i < numInputs; ++i) {
        const auto &batchInput = inputs[i];
        const auto &dims = m_inputDims[i];

        auto &input = batchInput[0];
        if (input.channels() != dims.d[0] || input.rows != dims.d[1] || input.cols != dims.d[2]) {
            spdlog::error("===== Error =====");
            spdlog::error("Input does not have correct size!");
            spdlog::error("Expected: ({}, {}, {})", dims.d[0], dims.d[1], dims.d[2]);
            spdlog::error("Got: ({}, {}, {})", input.channels(), input.rows, input.cols);
            spdlog::error("Ensure you resize your input image to the correct size");
            return false;
        }

        nvinfer1::Dims4 inputDims = {batchSize, dims.d[0], dims.d[1], dims.d[2]};
        m_context->setInputShape(m_IOTensorNames[i].c_str(),
                                 inputDims); // Define the batch size

        // OpenCV reads images into memory in NHWC format, while TensorRT expects
        // images in NCHW format. The following method converts NHWC to NCHW. Even
        // though TensorRT expects NCHW at IO, during optimization, it can
        // internally use NHWC to optimize cuda kernels See:
        // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-layout
        // Copy over the input data and perform the preprocessing
        auto mfloat = blobFromGpuMats(batchInput, m_subVals, m_divVals, m_normalize);
        preprocessedInputs.push_back(mfloat);
        m_buffers[i] = mfloat.ptr<void>();
    }

    // Ensure all dynamic bindings have been defined.
    if (!m_context->allInputDimensionsSpecified()) {
        auto msg = "Error, not all required dimensions specified.";
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    // Set the address of the input and output buffers
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
        if (!status) {
            return false;
        }
    }

    // Run inference.
    bool status = m_context->enqueueV3(inferenceCudaStream);
    if (!status) {
        return false;
    }

    // Copy the outputs back to CPU
    featureVectors.clear();

    for (int batch = 0; batch < batchSize; ++batch) {
        // Batch
        std::vector<std::vector<T>> batchOutputs{};
        for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
            // We start at index m_inputDims.size() to account for the inputs in our
            // m_buffers
            std::vector<T> output;
            auto outputLength = m_outputLengths[outputBinding - numInputs];
            output.resize(outputLength);
            // Copy the output
            Util::checkCudaErrorCode(cudaMemcpyAsync(output.data(),
                                                     static_cast<char *>(m_buffers[outputBinding]) + (batch * sizeof(T) * outputLength),
                                                     outputLength * sizeof(T), cudaMemcpyDeviceToHost, inferenceCudaStream));
            batchOutputs.emplace_back(std::move(output));
        }
        featureVectors.emplace_back(std::move(batchOutputs));
    }

    // Synchronize the cuda stream
    Util::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    Util::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
    return true;
}

template <typename T>
bool Engine<T>::runInference(const std::vector<std::vector<cv::cuda::GpuMat>> &inputs, cv::Mat &output) {
    // First we do some error checking
    if (inputs.empty() || inputs[0].empty()) {
        spdlog::error("Provided input vector is empty!");
        return false;
    }

    const auto numInputs = m_inputDims.size();
    if (inputs.size() != numInputs) {
        spdlog::error("Incorrect number of inputs provided!");
        return false;
    }

    // Ensure the batch size does not exceed the max
    if (inputs[0].size() > static_cast<size_t>(m_options.maxBatchSize)) {
        spdlog::error("===== Error =====");
        spdlog::error("The batch size is larger than the model expects!");
        spdlog::error("Model max batch size: {}", m_options.maxBatchSize);
        spdlog::error("Batch size provided to call to runInference: {}", inputs[0].size());
        return false;
    }

    // Ensure that if the model has a fixed batch size that is greater than 1, the
    // input has the correct length
    if (m_inputBatchSize != -1 && inputs[0].size() != static_cast<size_t>(m_inputBatchSize)) {
        spdlog::error("===== Error =====");
        spdlog::error("The batch size is different from what the model expects!");
        spdlog::error("Model batch size: {}", m_inputBatchSize);
        spdlog::error("Batch size provided to call to runInference: {}", inputs[0].size());
        return false;
    }

    const auto batchSize = static_cast<int32_t>(inputs[0].size());
    // Make sure the same batch size was provided for all inputs
    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i].size() != static_cast<size_t>(batchSize)) {
            spdlog::error("===== Error =====");
            spdlog::error("The batch size is different for each input!");
            return false;
        }
    }

    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    std::vector<cv::cuda::GpuMat> preprocessedInputs;

    // Preprocess all the inputs
    for (size_t i = 0; i < numInputs; ++i) {
        const auto &batchInput = inputs[i];
        const auto &dims = m_inputDims[i];

        auto &input = batchInput[0];
        if (input.channels() != dims.d[0] || input.rows != dims.d[1] || input.cols != dims.d[2]) {
            spdlog::error("===== Error =====");
            spdlog::error("Input does not have correct size!");
            spdlog::error("Expected: ({}, {}, {})", dims.d[0], dims.d[1], dims.d[2]);
            spdlog::error("Got: ({}, {}, {})", input.channels(), input.rows, input.cols);
            spdlog::error("Ensure you resize your input image to the correct size");
            return false;
        }

        nvinfer1::Dims4 inputDims = {batchSize, dims.d[0], dims.d[1], dims.d[2]};
        m_context->setInputShape(m_IOTensorNames[i].c_str(),
                                 inputDims); // Define the batch size
        // OpenCV reads images into memory in NHWC format, while TensorRT expects
        // images in NCHW format. The following method converts NHWC to NCHW. Even
        // though TensorRT expects NCHW at IO, during optimization, it can
        // internally use NHWC to optimize cuda kernels See:
        // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-layout
        // Copy over the input data and perform the preprocessing
        // auto mfloat = blobFromGpuMats(batchInput, m_subVals, m_divVals, m_normalize);
        // Assume already preprocessed input
        auto mfloat = blobFromGpuMats(batchInput);
        preprocessedInputs.push_back(mfloat);
        m_buffers[i] = mfloat.ptr<void>();
    }

    // Ensure all dynamic bindings have been defined.
    int32_t const size(m_engine->getNbIOTensors());
    std::vector<char const*> names(size);
    int32_t const nbNames = m_context->inferShapes(size, names.data());
    // if (!m_context->allInputDimensionsSpecified()) {
    if (nbNames < 0) {
        throw std::runtime_error("Error, not all required dimensions specified.");
    }

    // Set the address of the input and output buffers
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
        if (!status) {
            return false;
        }
    }

    // Check that all bindings are set before enqueue
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const char* tensorName = m_engine->getIOTensorName(i);
        if (!m_context->getTensorAddress(tensorName)) {
            std::cerr << "Tensor " << tensorName << " not bound!" << std::endl;
            return false;
        }
    }

    // Run inference.
    bool status = m_context->enqueueV3(inferenceCudaStream);
    if (!status) {
        return false;
    }

    // Copy the outputs back to CPU
    // const auto outputLength = m_outputLengths[0];
    // const auto outputDim = m_outputDims[0];
    // int dims[] = {batchSize, outputDim.d[1], outputDim.d[2]};
    // output = cv::Mat(3, dims, CV_32F);
    // int32_t outputBinding = numInputs; // We start at index m_inputDims.size() to account for the inputs in our m_buffers
    // Util::checkCudaErrorCode(cudaMemcpyAsync(output.data, 
    //                                          static_cast<char *>(m_buffers[outputBinding]),
    //                                          outputLength * sizeof(T) * batchSize, cudaMemcpyDeviceToHost, inferenceCudaStream));
    // TODO: currently only supports batch size 1
    const auto outputLength = m_outputLengths[0];
    const auto outputDim = m_outputDims[0];
    if (outputDim.nbDims == 4)
        output = cv::Mat(outputDim.d[2], outputDim.d[3], CV_32FC1);
    else if (outputDim.nbDims == 3)
        output = cv::Mat(outputDim.d[1], outputDim.d[2], CV_32FC1);
    else
        throw std::runtime_error("Output tensor is not 3D or 4D");
    int32_t outputBinding = numInputs; // We start at index m_inputDims.size() to account for the inputs in our m_buffers
    Util::checkCudaErrorCode(cudaMemcpyAsync(output.data, 
                                             static_cast<char *>(m_buffers[outputBinding]),
                                             outputLength * sizeof(T), cudaMemcpyDeviceToHost, inferenceCudaStream));

    // Synchronize the cuda stream
    Util::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    Util::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));

    return true;
}


template <typename T>
bool Engine<T>::runInference(const cv::Mat &input, cv::Mat &output) {
    const auto numInputs = m_inputDims.size();
    if (numInputs > 1) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Incorrect number of inputs provided!" << std::endl;
        return false;
    }

    // Ensure the batch size does not exceed the max
    if (input.size[0] > static_cast<size_t>(m_options.maxBatchSize)) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "The batch size is larger than the model expects!" << std::endl;
        std::cout << "Model max batch size: " << m_options.maxBatchSize << std::endl;
        std::cout << "Batch size provided to call to runInference: " << input.size[0] << std::endl;
        return false;
    }

    // Ensure that if the model has a fixed batch size that is greater than 1, the
    // input has the correct length
    if (m_inputBatchSize != -1 && input.size[0] != static_cast<size_t>(m_inputBatchSize)) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "The batch size is different from what the model expects!" << std::endl;
        std::cout << "Model batch size: " << m_inputBatchSize << std::endl;
        std::cout << "Batch size provided to call to runInference: " << input.size[0] << std::endl;
        return false;
    }

    const auto batchSize = static_cast<int32_t>(input.size[0]);

    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    std::vector<cv::cuda::GpuMat> preprocessedInputs;

    // Preprocess the input
    const auto &batchInput = input;
    const auto &dims = m_inputDims[0];

    // auto &input = batchInput[0];
    if (batchInput.size[1] != dims.d[0] || batchInput.size[2] != dims.d[1] || batchInput.size[3] != dims.d[2]) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "batchInput does not have correct size!" << std::endl;
        std::cout << "Expected: (" << dims.d[0] << ", " << dims.d[1] << ", " << dims.d[2] << ")" << std::endl;
        std::cout << "Got: (" << batchInput.size[1] << ", " << batchInput.size[2] << ", " << batchInput.size[3] << ")" << std::endl;
        std::cout << "Ensure you resize your batched input to the correct size" << std::endl;
        return false;
    }

    nvinfer1::Dims4 inputDims = {batchSize, dims.d[0], dims.d[1], dims.d[2]};
    m_context->setInputShape(m_IOTensorNames[0].c_str(), inputDims); // Define the batch size

    auto tShape = m_context->getTensorShape(m_IOTensorNames[0].c_str());

    // OpenCV reads images into memory in NHWC format, while TensorRT expects
    // images in NCHW format. The following method converts NHWC to NCHW. Even
    // though TensorRT expects NCHW at IO, during optimization, it can
    // internally use NHWC to optimize cuda kernels See:
    // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-layout
    // Copy over the input data and perform the preprocessing
    // auto mfloat = blobFromGpuMats(batchInput, m_subVals, m_divVals, m_normalize);
    auto mfloat = blobFromMat(batchInput, m_subVals, m_divVals, m_normalize);
    preprocessedInputs.push_back(mfloat);
    m_buffers[0] = mfloat.ptr<void>();

    // Ensure all dynamic bindings have been defined.
    int32_t const size(m_engine->getNbIOTensors());
    std::vector<char const*> names(size);
    int32_t const nbNames = m_context->inferShapes(size, names.data());
    // if (!m_context->allInputDimensionsSpecified()) {
    if (nbNames < 0) {
        throw std::runtime_error("Error, not all required dimensions specified.");
    }

    // Set the address of the input and output buffers
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
        if (!status) {
            return false;
        }
    }

    // Run inference.
    bool status = m_context->enqueueV3(inferenceCudaStream);
    if (!status) {
        return false;
    }

    // Copy the outputs back to CPU
    const auto outputLength = m_outputLengths[0];
    output = cv::Mat(batchSize, outputLength, CV_32F);

    int32_t outputBinding = numInputs; // We start at index m_inputDims.size() to account for the inputs in our m_buffers
    Util::checkCudaErrorCode(cudaMemcpyAsync(output.data, 
                                             static_cast<char *>(m_buffers[outputBinding]),
                                             outputLength * sizeof(T) * batchSize, cudaMemcpyDeviceToHost, inferenceCudaStream));

    // // Copy the outputs back to CPU
    // featureVectors.clear();

    // for (int batch = 0; batch < batchSize; ++batch) {
    //     // Batch
    //     std::vector<std::vector<T>> batchOutputs{};
    //     for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
    //         // We start at index m_inputDims.size() to account for the inputs in our
    //         // m_buffers
    //         std::vector<T> output;
    //         auto outputLength = m_outputLengths[outputBinding - numInputs];
    //         output.resize(outputLength);
    //         // Copy the output
    //         Util::checkCudaErrorCode(cudaMemcpyAsync(output.data(),
    //                                                  static_cast<char *>(m_buffers[outputBinding]) + (batch * sizeof(T) * outputLength),
    //                                                  outputLength * sizeof(T), cudaMemcpyDeviceToHost, inferenceCudaStream));
    //         batchOutputs.emplace_back(std::move(output));
    //     }
    //     featureVectors.emplace_back(std::move(batchOutputs));
    // }

    // Synchronize the cuda stream
    Util::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    Util::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));

    // transpose output matrix
    // output = output.t();

    // print first batch of inputMatrix
    // auto tmp = batchInput({cv::Range(0,1), cv::Range::all(), cv::Range::all(), cv::Range::all()}).reshape(1, {32, 32});
    // std::cout << "batch: " << tmp.row(0) << std::endl;

    // std::cout << "number of features: " << output.size[0] << std::endl;
    // std::cout << "feature length: " << output.size[1] << std::endl;
    // std::cout << "first feature: " << output.row(0) << std::endl;
    // std::cout << "second feature: " << output.row(1) << std::endl;
    // std::cout << "last feature: " << output.row(output.size[0] - 1) << std::endl;


    return true;
}


template <typename T>
bool Engine<T>::runInferenceCUDA(const std::vector<int> &inputShape, T *input, cv::Mat &output) {
    const auto numInputs = m_inputDims.size();
    if (numInputs > 1) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Incorrect number of inputs provided!" << std::endl;
        return false;
    }

    const int batchSize = inputShape[0];
    // Ensure the batch size does not exceed the max
    if (batchSize > static_cast<size_t>(m_options.maxBatchSize)) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "The batch size is larger than the model expects!" << std::endl;
        std::cout << "Model max batch size: " << m_options.maxBatchSize << std::endl;
        std::cout << "Batch size provided to call to runInference: " << batchSize << std::endl;
        return false;
    }

    // Ensure that if the model has a fixed batch size that is greater than 1, the
    // input has the correct length
    if (m_inputBatchSize != -1 && batchSize != static_cast<size_t>(m_inputBatchSize)) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "The batch size is different from what the model expects!" << std::endl;
        std::cout << "Model batch size: " << m_inputBatchSize << std::endl;
        std::cout << "Batch size provided to call to runInference: " << batchSize << std::endl;
        return false;
    }

    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    // Preprocess the input
    const auto &dims = m_inputDims[0];

    // auto &input = batchInput[0];
    if (inputShape[1] != dims.d[0] || inputShape[2] != dims.d[1] || inputShape[3] != dims.d[2]) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "batchInput does not have correct size!" << std::endl;
        std::cout << "Expected: (" << dims.d[0] << ", " << dims.d[1] << ", " << dims.d[2] << ")" << std::endl;
        std::cout << "Got: (" << inputShape[1] << ", " << inputShape[2] << ", " << inputShape[3] << ")" << std::endl;
        std::cout << "Ensure you resize your batched input to the correct size" << std::endl;
        return false;
    }

    nvinfer1::Dims4 inputDims = {batchSize, dims.d[0], dims.d[1], dims.d[2]};
    // std::cout << "m_IOTensorNames: " << m_IOTensorNames[0] << std::endl;
    m_context->setInputShape(m_IOTensorNames[0].c_str(),
                             inputDims); // Define the batch size

    auto tShape = m_context->getTensorShape(m_IOTensorNames[0].c_str());
    // print shape
    // std::cout << "Shape: ";
    // for (int i = 0; i < tShape.nbDims; ++i)
    //     std::cout << tShape.d[i] << " ";
    // std::cout << std::endl;

    // cast to void ptr
    m_buffers[0] = static_cast<void *>(input);

    // Ensure all dynamic bindings have been defined.
    int32_t const size(m_engine->getNbIOTensors());
    std::vector<char const*> names(size);
    int32_t const nbNames = m_context->inferShapes(size, names.data());
    if (nbNames < 0) {
        throw std::runtime_error("Error, not all required dimensions specified.");
    }

    // Set the address of the input and output buffers
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
        if (!status) {
            return false;
        }
    }

    // Run inference.
    bool status = m_context->enqueueV3(inferenceCudaStream);
    if (!status) {
        return false;
    }

    // Copy the outputs back to CPU
    const auto outputLength = m_outputLengths[0];
    output = cv::Mat(batchSize, outputLength, CV_32F);

    int32_t outputBinding = numInputs; // We start at index m_inputDims.size() to account for the inputs in our m_buffers
    Util::checkCudaErrorCode(cudaMemcpyAsync(output.data, 
                                             static_cast<char *>(m_buffers[outputBinding]),
                                             outputLength * sizeof(T) * batchSize, cudaMemcpyDeviceToHost, inferenceCudaStream));

    // Synchronize the cuda stream
    Util::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    Util::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));

    return true;
}