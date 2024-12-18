#pragma once

#include "NetworkLayer.h"

#include <memory>


drow calcLoss(const dmatrix& networkOutputs, const dmatrix& targetOutputs); // given what the network outputs, and the correct outputs, calculate the loss total so the network can paramDoOptimize itself later

double calcAvgTotalLoss(const dmatrix& networkOutputs, const dmatrix& targetOutputs); // same formula and math as before, but averages each batch loss into one double value (total loss)

class NeuralNetwork
{
private:
	dmatrix inputs; // initial batch(s) of inputs for neural network
	std::vector<dmatrix> inputBatches;
	int batchSize; // when given a large amount of input batches, for training purposes it is usually better to split up into batches to stop overfitting

	std::vector<std::shared_ptr<NetworkLayer>> hiddenLayers; // size is amt of hidden layers and the val is the number of neurons for the corresponding layer

	dmatrix targetOutputs; // outputs given by user to train the network
	std::vector<dmatrix> targetBatches;

	//int trainingType; // tells network whether to run backprogagation after calculating loss
	dmatrix outputs; // final outputs for neural network, after running through each hidden layer, and applying softMax activation to the output layer
	drow loss; // the values (one for each batch) of how wrong the network was at predicting the correct output
public:
	NeuralNetwork(const dmatrix& paramInputs, const std::vector<int>& paramHiddenLayers, const dmatrix& paramTargetOutputs = {}, const int paramBatchSize = 0); 
	// ctor takes in initial input layer, hidden layers, if needed the correct outputs for training, and batch size if specified for how much data to run through/train at a time

	void getBatches(); // called during ctor, generates batches of inputs based on user inputted batch size
	void runNetworkBackprop(int runAmt = 10, const double learnRate = 1); // runs the network, starting with inputs, through all hidden layers, and finally calculating the outputs and loss.
	void backpropagate(const std::pair<dmatrix, dmatrix>& inputTargetBatch, const double learnRate); // optimizes weights and biases in network

	dmatrix getOutputs() { return outputs; }

	drow getLoss() { return loss; }

};

