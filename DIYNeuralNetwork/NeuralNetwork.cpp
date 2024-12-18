#include "NeuralNetwork.h"

#include <iostream>
#include <math.h>
#include <numeric>

#include <algorithm>
#include <time.h>

drow calcLoss(const dmatrix& networkOutputs, const dmatrix& targetOutputs) 
{
	drow lossVals;
	for (int i = 0; i < networkOutputs.size(); ++i) {
		//double batchLoss = 0;
		for (int j = 0; j < networkOutputs.at(0).size(); ++j) {

			if (targetOutputs[i][j] == 1) { // for each vector/batch of neurons, check where the target output is true (= 1) and calculate loss (each targetOutput vec has one value that = 1)
				lossVals.push_back(-log(networkOutputs[i][j] + 1e-8)); // 1e-8 is so we don't divide by 0
			}   // if the network outputted .06 where target output is 1, loss is = -log(.6) (0.22185)

			// ********************************* actual equation, simplified because of one-hot encoding (i.e. {0, 1, 0} target values), so equation above is used ***************************************
			//batchLoss -= (targetOutputs[i][j] * log(networkOutputs[i][j])); // calculates loss using the negative log of the output given by the network, multiplyed by the target value for that neuron
		}
		//lossVals.push_back(batchLoss);
	}

	//calc loss of each individual output
	//dmatrix outErrors;
	//for (int i = 0; i < networkOutputs.size(); ++i) {
	//	outErrors.push_back({});
	//	for (int j = 0; j < networkOutputs.at(0).size(); ++j) {
	//		outErrors.at(i).push_back( -((targetOutputs[i][j] * 1 / networkOutputs[i][j])) ); // calculates loss of each output by using deriv of loss function with respect to output
	//	}
	//}
	return lossVals;
}

double calcAvgTotalLoss(const dmatrix& networkOutputs, const dmatrix& targetOutputs)
{
	double lossVal = 0;
	for (int i = 0; i < networkOutputs.size(); ++i) {
		for (int j = 0; j < networkOutputs.at(0).size(); ++j) {
			if (targetOutputs[i][j] == 1) { // for each vector/batch of neurons, check where the target output is true (= 1) and calculate loss (each targetOutput vec has one value that = 1)
				lossVal -= (log(networkOutputs[i][j] + 1e-8)); // 1e-8 is so we don't divide by 0
			}   // if the network outputted .06 where target output is 1, loss is = -log(.6) (0.22185)
		}
	}
	lossVal /= networkOutputs.size();
	return lossVal;
}


NeuralNetwork::NeuralNetwork(const dmatrix & paramInputs, const std::vector<int>& paramHiddenLayers, const dmatrix& paramTargetOutputs, const int paramBatchSize) // hiddenLayers vector size is num of hiddenLayers, the values in the vector are the #output neurons
	: inputs{ paramInputs }, targetOutputs{ paramTargetOutputs }, batchSize{ paramBatchSize }
{ 

	if (paramHiddenLayers.size() == 1) { // if there's no hidden layers, make the only layer a FinalLayer obj, then return
		hiddenLayers.push_back(std::shared_ptr<FinalLayer>(new FinalLayer(inputs.at(0).size(), paramHiddenLayers.at(0))));
		return;
	}

	for (int i = 0; i < paramHiddenLayers.size(); ++i) {
		if (i == 0 && paramHiddenLayers.size()) { //first hidden layer #input neurons = to the # of initial input neurons
			hiddenLayers.push_back(std::shared_ptr<NetworkLayer>(new NetworkLayer(inputs.at(0).size(), paramHiddenLayers.at(i))));
		}
		else if (i == (paramHiddenLayers.size() - 1)) { // at the last layer, which has the final outputs, add a FinalLayer obj instead, which uses softMax activation for outputs
			hiddenLayers.push_back(std::shared_ptr<FinalLayer>(new FinalLayer(paramHiddenLayers.at(i - 1), paramHiddenLayers.at(i))));
		}
		else { //all other hidden layers #input neurons = to #of previous layer's #output neurons
			hiddenLayers.push_back(std::shared_ptr<NetworkLayer>(new NetworkLayer(paramHiddenLayers.at(i - 1), paramHiddenLayers.at(i))));
		}
	}
}

void NeuralNetwork::getBatches()
{
	inputBatches.clear(); // this runs each time, amt = runAmt param, so we need to clear the batches each time since they are randomized
	targetBatches.clear();
	std::vector<int> randIndexes;
	for (int i = 0; i < inputs.size(); ++i) {
		randIndexes.push_back(i);
	}
	std::random_shuffle(randIndexes.begin(), randIndexes.end()); // randomize indexes
	dmatrix randInputs;
	dmatrix randTarOutputs;
	for (auto& indx : randIndexes) { // randomize the inputs/targetOutputs in the same order corresponding to each random indx value
		randInputs.push_back(inputs.at(indx));
		randTarOutputs.push_back(targetOutputs.at(indx));
	}
	//std::cout << "random inputs: " << randInputs;
	//std::cout << "random target outputs: " << randTarOutputs;
	if (batchSize != 0) { // if 0 don't worry about creating batches
		int numBatches;
		int modInputs = 0; // for when batches aren't evenly divided
		bool modLess = false; // false if final batch should add inputs = to modInputs, true if it should subtract
		if ((randInputs.size() % batchSize) > batchSize / 2) { // e.g. 47 / 16 is 2, but it would obviously better to have 3 batches, this is code to account for that
			numBatches = (randInputs.size() / batchSize) + 1;
			modInputs = batchSize - (randInputs.size() % batchSize); // now numBatches is three, modInputs is 1, and modLess will be set to subtract that 1 from final batch
			modLess = true;
		}
		else
		{
			numBatches = randInputs.size() / batchSize;
			modInputs = randInputs.size() % batchSize; // If there's a remainder, we just add it 
		}
		for (int i = 0; i < numBatches; ++i) {
			inputBatches.push_back({}); // push back an empty dmatrix

			if (i == (numBatches - 1) && modInputs != 0) { // for last batch, accounting for modInputs
				if (modLess) {
					for (int m = 0; m < (batchSize - modInputs); ++m) {
						inputBatches.at(i).push_back(randInputs.at(m + (i * batchSize)));
					}
				}
				else
				{
					for (int m = 0; m < (batchSize + modInputs); ++m) {
						inputBatches.at(i).push_back(randInputs.at(m + (i * batchSize)));
					}
				}
			}
			else // for all other batches
			{
				for (int j = 0; j < batchSize; ++j) {
					inputBatches.at(i).push_back(randInputs.at(j + (i * batchSize)));
				}
			}
		}
	}
	else {
		inputBatches.push_back(randInputs); // if not using batches, put all inputs into one inputBatches dmatrix
	}
	for (int i = 0; i < inputBatches.size(); ++i) {
		targetBatches.push_back({});
		for (int j = 0; j < inputBatches.at(i).size(); ++j) {
			targetBatches.at(i).push_back(randTarOutputs.at(j + (i * batchSize)));
		}
	}
}

void NeuralNetwork::runNetworkBackprop(int runAmt, const double learnRate)
{
	while (runAmt > 0) {
		getBatches(); // Get input & target batches based off of user inputted batch size (or if not inputted, make one batch with all inputs)
		//for (int i = 0; i < inputBatches.size(); ++i) {
		//	std::cout << "input batch: " << inputBatches.at(i) << "target batch: " << targetBatches.at(i);
		//}
		for (int b = 0; b < inputBatches.size(); ++b) { // run network (and if needed backprop) for every batch of inputs
			auto inputTargetBatch = std::make_pair(inputBatches.at(b), targetBatches.at(b));
			for (int i = 0; i < hiddenLayers.size(); ++i) {
				if (i == 0) { // if it's the first layer, run the network layer with the params of the initial inputs (as long as there's not only one layer
					hiddenLayers.at(i)->forwardPass(inputTargetBatch.first);
				}
				else //otherwise, feed in the previous layer's outputs as params
				{
					hiddenLayers.at(i)->forwardPass(hiddenLayers.at(i - 1)->getOutputs());
				}
			}
			//std::cout << hiddenLayers.at(hiddenLayers.size() - 2)->getOutputs();
			outputs = hiddenLayers.back()->getOutputs(); // the final outputs are the last layer's outputs (which is a FinalLayer obj)
			//std::cout << outputs;
			if (!targetOutputs.empty()) { // if the network was given target outputs to train itself, and user specified network should be training
				backpropagate(inputTargetBatch, learnRate); // begin optimization
			}
		}
		--runAmt;
	}
	hiddenLayers.at(0)->forwardPass(inputs);
	outputs = hiddenLayers.at(0)->getOutputs();
	std::cout << "NETWORK INPUTS" << inputs;
	std::cout << "NETWORK TARGET OUTPUTS" << targetOutputs;
	std::cout << "\t************   NETWORK OUTPUTS   ************\n" << outputs;
	std::cout << "FINAL ORIG NETWORK TOTAL ERROR: " << calcLoss(outputs, targetOutputs) << "AVERAGE: " << calcAvgTotalLoss(outputs, targetOutputs) << "\n\n";
}

void NeuralNetwork::backpropagate(const std::pair<dmatrix, dmatrix>& inputTargetBatch, const double learnRate)
{
	/******************************************** ABBREVIATION GUIDE **********************************
			* y = target outputs
			* x = network outputs/"guesses"
			* z = network outputs BEFORE activation
			* h = previous layers outputs  
			* w = weights
			* b = biases
	*/

	//loss = calcLoss(outputs, targetOutputs); // calculate the loss and pass it the the data member in the class
	//std::cout << loss;

	for (int i = (hiddenLayers.size() - 1); i >= 0; --i) { // start from final layer
		std::shared_ptr<NetworkLayer> currHL = hiddenLayers.at(i);
		if (i != 0) {
			std::shared_ptr<NetworkLayer> prevHL = hiddenLayers.at(i-1);
			if (hiddenLayers.at(i) == hiddenLayers.back()) { // we have to first backprop with final layer and with deriv of softMax
				// deriv of Err with respect to z (pre-softMax outputs) = x (network "guess") - y (network target)
				dmatrix dZ = matrixSub(outputs, inputTargetBatch.second);
				// deriv of Err with respect to Weights = dZ * h (previous layers outputs)
				dmatrix dW = dotProduct(tposeMatrix(dZ), prevHL->getOutputs());
				// deriv of Err with respect to Biases is just the avg of the seperate outputs dZ
				drow dB;
				double dZtotal = 0;
				for (int i = 0; i < dZ.at(0).size(); ++i) {
					for (int j = 0; j < dZ.size(); ++j) {
						dZtotal += dZ[j][i];
					}
					dB.push_back(dZtotal / dZ.size());
					dZtotal = 0;
				}
				dmatrix dBMat = toMatrix(dB);
				currHL->optimizeWeights(matrixSub(currHL->getWeights(), matrixMult(tposeMatrix(dW), matrixSet(tposeMatrix(dW), learnRate))));
				currHL->optimizeBiases(fromMatrix(matrixSub(toMatrix(currHL->getBiases()), matrixMult(dBMat, matrixSet(dBMat, learnRate)))));
			}
		}
		else
		{
			if (hiddenLayers.at(i) == hiddenLayers.back()) { // if this is true, then there are no hidden layers, just output layer and input layer
				// deriv of Err with respect to z (pre-softMax outputs) = x (network "guess") - y (network target)
				dmatrix dZ = matrixSub(outputs, inputTargetBatch.second);
				// deriv of Err with respect to Weights = dZ * h (previous layers outputs)
				dmatrix dW = dotProduct(tposeMatrix(dZ), inputTargetBatch.first);
				// deriv of Err with respect to Biases is just the avg of the seperate outputs dZ
				drow dB;
				double dZtotal = 0;
				for (int i = 0; i < dZ.at(0).size(); ++i) {
					for (int j = 0; j < dZ.size(); ++j) {
						dZtotal += dZ[j][i];
					}
					dB.push_back(dZtotal / dZ.size());
					dZtotal = 0;
				}
				dmatrix dBMat = toMatrix(dB);
				currHL->optimizeWeights(matrixSub(currHL->getWeights(), matrixMult(tposeMatrix(dW), matrixSet(tposeMatrix(dW), learnRate))));
				currHL->optimizeBiases(fromMatrix(matrixSub(toMatrix(currHL->getBiases()), matrixMult(dBMat, matrixSet(dBMat, learnRate)))));
			}
			else
			{
			}
		}
		
	}
}

