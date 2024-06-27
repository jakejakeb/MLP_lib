#pragma once

#include <../Matrix_lib/include/matrix.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>

namespace MLP
{
	inline float Sigmoid(float x)
	{
		return 1.0f / (1 + exp(-x));
	}
	inline float DSigmoid(float x)
	{
		return (x * (1 - x));
	}
    inline size_t getFileSize(std::string filename)
    {
        std::ifstream file(filename);
        
        if (!file.is_open()) 
            std::cerr << "Error opening file\n";

        size_t lineCount = 0;
        std::string line;
        
        while (std::getline(file, line))
        {
            lineCount++;
        }

        file.close();
        return lineCount;
    }
    inline std::vector<float> oneHot(const std::vector<float> input) 
    {
        float maxVal = input[0];
        size_t maxIndex = 0;
        for (size_t i = 1; i < input.size(); ++i) {
            if (input[i] > maxVal) {
                maxVal = input[i];
                maxIndex = i;
            }
        }

        std::vector<float> result(input.size(), 0.0f);
        result[maxIndex] = 1.0f;

        return result;
    }
    inline std::vector<float> round(const std::vector<float> input) 
    {
        std::vector<float> result;
        for (float f : input)
            result.push_back((f > 0.5f) ? 1 : 0);
        return result;
    }
	class NeuralNet
	{
	public:
        std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> _trainData;
		std::vector<size_t> _topology;
		std::vector<Mat2D<float>> _weightMat;
		std::vector<Mat2D<float>> _valueMat;
		std::vector<Mat2D<float>> _biasMat;
		double _learningRate = 0.1;

		NeuralNet(std::vector<size_t> topology, float learningRate = 0.1f)
			: _topology(topology), _weightMat({}), _valueMat({}), _biasMat({}), _learningRate(learningRate)
		{
			for (size_t i = 0; i < topology.size() - 1; i++)
			{
				Mat2D<float> weightMat(topology[i + 1], topology[i]);
				weightMat = weightMat.applyFunction([](const float& val) {
					return (float)rand() / RAND_MAX;
					});
				_weightMat.push_back(weightMat);

				Mat2D<float> biasMat(topology[i + 1], 1);
				biasMat = biasMat.applyFunction([](const float& val) {
					return (float)rand() / RAND_MAX;
					});
				_biasMat.push_back(biasMat);
			}
			_valueMat.resize(topology.size());
		}

		bool feedForword(std::vector<float> input)
		{
			if (input.size() != _topology[0])
				return false;

			Mat2D<float> values(input.size(), 1);
			for (size_t i = 0; i < input.size(); i++)
				values._vals[i] = input[i];

			for (size_t i = 0; i < _weightMat.size(); i++)
			{
				_valueMat[i] = values;
				values = values.mult(_weightMat[i]);
				values = values.add(_biasMat[i]);
				values = values.applyFunction(Sigmoid);
			}
			_valueMat[_weightMat.size()] = values;
			return true;
		}

		bool backPropagate(std::vector<float> targetOutput)
		{
			if (targetOutput.size() != _topology.back())
				return false;

			Mat2D<float> errors(targetOutput.size(), 1);
			errors._vals = targetOutput;
			auto err = _valueMat.back().negetive();
			errors = errors.add(err);


			for (int i = _weightMat.size() - 1; i >= 0; i--)
			{
				auto weightmattranspsoe = _weightMat[i].transpose();
				Mat2D<float> prevErrors = errors.mult(weightmattranspsoe);

				Mat2D<float> dOutputs = _valueMat[i + 1].applyFunction(DSigmoid);
				Mat2D<float> gradients = errors.multElem(dOutputs);
				gradients = gradients.multScaler(_learningRate);
				Mat2D<float> weightGradients = _valueMat[i].transpose().mult(gradients);

				_biasMat[i] = _biasMat[i].add(gradients);
				_weightMat[i] = _weightMat[i].add(weightGradients);
				errors = prevErrors;
			}

			return true;
		}

		std::vector<float> getPredictions()
		{
			return _valueMat.back()._vals;
		}

		double meanSquaredError(const std::vector<float>& predictions, const std::vector<float>& targets) 
		{
			double error = 0.0;
			if (predictions.size() != targets.size()) 
			{
				return -1.0; 
			}
			for (size_t i = 0; i < predictions.size(); ++i) 
			{
				error += (predictions[i] - targets[i]) * (predictions[i] - targets[i]);
			}
			return error / predictions.size();
		}

        void importWB(const std::string &filename) 
        {
            std::vector<std::vector<float>> W, B;
            std::ifstream file(filename);
            if (!file.is_open())
                std::cerr << "Error opening file" << std::endl;

            std::pair<std::vector<Mat2D<float>>, std::vector<Mat2D<float>>> _data;
            std::string line;
            while (getline(file, line)) 
            {
                if (line.empty()) continue;
                
                std::istringstream iss(line);
                char group;
                int x, y;
                iss >> group >> y >> x;
                
                std::vector<float> numbers(x * y);

                for (int i = 0; i < x * y; ++i)
                    file >> numbers[i];

                if (group == 'W') 
                {
                    W.push_back(numbers);
                    _data.second.push_back(Mat2D<float>(x, y));
                } 
                else if (group == 'B') 
                {
                    B.push_back(numbers);
                    _data.first.push_back(Mat2D<float>(x, y));
                }

                file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            }
            
            for (size_t i = 0; i < W.size(); i++)
                _data.second[i]._vals = W[i];
            for (size_t i = 0; i < B.size(); i++)
                _data.first[i]._vals = B[i];
            file.close();
            NeuralNet::_weightMat = _data.second;
            NeuralNet::_biasMat = _data.first;
        }

        void importTrainData(const std::string& filename)
        {
            std::ifstream file(filename);
            if (!file.is_open()) 
            {
                std::cerr << "Error opening file: " << filename << std::endl;
                return;
            }
            std::string line;

            while (std::getline(file, line)) 
            {
                std::stringstream line_stream(line);
                std::vector<float> inputLineData;
                std::vector<float> outputLineData;
                float value;
                std::string token;

                std::getline(line_stream, token, ':');
                std::stringstream input_stream(token);
                while (input_stream >> value)
                    inputLineData.push_back(value);

                if (std::getline(line_stream, token)) 
                {
                    std::stringstream output_stream(token);
                    while (output_stream >> value)
                        outputLineData.push_back(value);
                }

                if (!inputLineData.empty())
                    _trainData.first.push_back(inputLineData);
                if (!outputLineData.empty())
                    _trainData.second.push_back(outputLineData); 
            }
            file.close();
        }
	
        void exportWB(const std::string& filename)
        {
            std::pair<std::vector<Mat2D<float>>, std::vector<Mat2D<float>>> matrices = {_biasMat, _weightMat};
            std::ofstream outFile(filename);

            if (!outFile.is_open()) 
            {
                std::cerr << "Failed to open the file." << std::endl;
                return;
            }

            size_t size = matrices.first.size();
            if (size != matrices.second.size()) 
            {
                std::cerr << "Vectors in the pair have different sizes." << std::endl;
                return;
            }

            for (size_t i = 0; i < size; ++i) 
            {
                Mat2D<float>& mat2 = matrices.first[i];
                Mat2D<float>& mat1 = matrices.second[i];

                outFile << "\nW " << mat1._rows << " " << mat1._cols << "\n";
                for (size_t y = 0; y < mat1._rows; ++y) 
                {
                    for (size_t x = 0; x < mat1._cols; ++x)
                        outFile  << mat1.at(x, y) << "\n";
                }

                outFile << "\nB " << mat2._rows << " " << mat2._cols << "\n";
                for (size_t y = 0; y < mat2._rows; ++y) 
                {
                    for (size_t x = 0; x < mat2._cols; ++x)
                        outFile << mat2.at(x, y) << "\n";
                }
            }

            outFile.close();
            std::cout << "\nData written to " << filename << std::endl;
        }
    };
}

#ifdef QUICKSTART

    class Controller 
    {
    public:
        std::string WB_path = "../data/w_b.txt";
        std::string traindata_importPath = "../data/inputs.txt";
        netlib::NeuralNet nn;
        Controller(std::vector<size_t> topology, float learningRate = 0.1f)
        : nn(topology, learningRate)
        {
            nn.importTrainData(traindata_importPath);
            std::thread t(&Controller::start, this);
            t.detach(); 
        }


        void testNetwork(std::vector<float> testData, netlib::NeuralNet& nn) {
            try {
                nn.feedForword(testData);
                std::vector<float> preds = nn.getPredictions();
                std::cout << "\nPredictions: ";
                preds = netlib::round(preds);
                for (float f : preds) std::cout << f << " ";
                std::cout << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Exception occurred: " << e.what() << std::endl;
            }
        }

        void trainNetwork(netlib::NeuralNet &nn, const std::vector<std::vector<float>> &fileInputs,
                  const std::vector<std::vector<float>> &targetOutputs, size_t iterations, 
                  const std::string &inputFile) 
        {
            auto start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < iterations; i++) 
            {
                size_t index = rand() % netlib::getFileSize(inputFile);
                nn.feedForword(fileInputs[index]);
                nn.backPropagate(targetOutputs[index]);

                if (i % 100000 == 0 || i == 0) 
                {
                    auto current = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed = current - start;

                    double time_per_iteration = elapsed.count() / (i + 1);
                    double remaining_iterations = iterations - (i + 1);
                    double estimated_remaining_time = remaining_iterations * time_per_iteration;

                    std::cout << std::fixed << std::setprecision(2);
                    std::cout << "Iteration: " << i << ", Estimated time remaining: "
                            << ((estimated_remaining_time / 60 > 60) ? (estimated_remaining_time / 60 / 60)
                            : (estimated_remaining_time / 60)) << ((estimated_remaining_time / 60 > 60)
                            ? " hours " : " mins ") << std::endl;
                }
            }
            std::cout << "training complete\n";

            double averageLoss = 0;
            for (size_t i = 0; i < fileInputs.size(); i++) {
                nn.feedForword(fileInputs[i]);
                std::vector<float> preds = nn.getPredictions();
                averageLoss += nn.meanSquaredError(preds, targetOutputs[i]);
            }
            std::cout << "average error: " << averageLoss / fileInputs.size() << "\n";
        }

        void start()
        {
            std::string command;
            bool running = true;
            while (running)
            {
                std::getline(std::cin, command);

                std::istringstream iss(command);

                if (command == "train") {
                    size_t iterations;
                    std::cout << "iterations: ";
                    std::cin >> iterations;
                    trainNetwork(nn, nn._trainData.first, nn._trainData.second, iterations, traindata_importPath);
                }
                else if (command.rfind("test", 0) == 0) 
                {
                    std::istringstream iss(command.substr(5));
                    std::vector<float> _data;
                    float number;
                    while (iss >> number)
                    {
                        _data.push_back(number);
                    }
                    testNetwork(_data, nn);
                } else if (command == "export") {
                    std::cout << "W & B exported\n";
                    nn.exportWB(WB_path);
                } else if (command == "import") {
                    std::cout << "W & B imported\n";
                    nn.importWB(WB_path);  
                } else if (command == "close") {
                    running = false;
                } 

            }
        }
    };

#endif
