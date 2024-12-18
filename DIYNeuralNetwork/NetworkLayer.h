#pragma once

#include <vector>
#include <iostream>


using dmatrix = std::vector<std::vector<double>>; //matrix of type double
using drow = std::vector<double>; //singular vector row of type double

dmatrix tposeMatrix(const dmatrix& m); // columns = rows, rows = columns

dmatrix dotProduct(const dmatrix& m1, const dmatrix& m2); // The output is the dot product matrix of the two matrices

dmatrix matrixMult(const dmatrix& m1, const dmatrix& m2);

dmatrix matrixAdd(const dmatrix& m1, const dmatrix& m2);

dmatrix matrixSub(const dmatrix& m1, const dmatrix& m2);

dmatrix matrixSet(const dmatrix& m, const double& d); // returns a matrix of the same size with double d for all values

dmatrix matrixVecAddition(const dmatrix& m1, const drow& row); // for adding the biases to the values given by the dot product of the neurons and weights

dmatrix toMatrix(const drow& r); // converts a drow into a dmatrix of size 1

drow fromMatrix(const dmatrix& m); // converts a dmatrix into a drow (any dmatrix's above size 1 will have values lost)

dmatrix operator+=(dmatrix& m1, const dmatrix& m2);

std::ostream& operator<<(std::ostream& out, const drow& r); // prints a 1D vector to the kernel

std::ostream& operator<<(std::ostream& out, const dmatrix& m); // prints a 2D matrix to the kernel

double random(const double& min, const double& max); // returns a random value between the given min and max #s

dmatrix activationReLU(const dmatrix& m); // The ReLU activation function passes the max of 0 & each value in the matrix

dmatrix activationSoftMax(const dmatrix& m); // exponentiates and normalizes each batch of output neurons so that the total of neurons in each batch is 1

class NetworkLayer
{
protected:
	dmatrix weights;
	drow biases;

	dmatrix preActivation; // used for backprop
	dmatrix outputs;
public:
	NetworkLayer(const int numOfInputs, const int numOfNeurons);  // ctor takes in # of input neurons given and # of output neurons to be outputted (# of batches given can be as high or low as needed)

	// first multiply by weights, then add results with biases, then apply ReLU activation function
	virtual void forwardPass(const dmatrix& inputs) { preActivation = matrixVecAddition(dotProduct(inputs, weights), biases); outputs = activationReLU(preActivation); }

	dmatrix getWeights() { return weights; }
	drow getBiases() { return biases; }

	dmatrix getPreActivation() { return preActivation; }
	dmatrix getOutputs() { return outputs; }
	void optimizeWeights(const dmatrix& newWeights) { weights = newWeights; }
	void optimizeBiases(const drow& newBiases) { biases = newBiases; }
};

class FinalLayer : public NetworkLayer
{
private:
	dmatrix preSoftMax;
public:
	FinalLayer(const int numOfInputs, const int numOfNeurons) : NetworkLayer(numOfInputs, numOfNeurons) {} // ctor just uses base classes' ctor

	// first multiply by weights, then add results with biases, then apply softMax activation function
	void forwardPass(const dmatrix& inputs) override { preActivation = matrixVecAddition(dotProduct(inputs, weights), biases); outputs = activationSoftMax(preActivation); }
};
