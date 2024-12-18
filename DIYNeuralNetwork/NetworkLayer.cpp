#include "NetworkLayer.h"

#include <iostream>
#include <string>
#include <numeric>
#include <random>
#include <ctime>
#include <iomanip>

dmatrix tposeMatrix(const dmatrix& m) {
	dmatrix tposed;
	for (int i = 0; i < m.at(0).size(); ++i) {
		tposed.push_back({});
		for (int j = 0; j < m.size(); ++j) {
			tposed.at(i).push_back(m.at(j).at(i));
		}
	}
	return tposed; // #columns become #rows, and vice versa
}

dmatrix dotProduct(const dmatrix& m1, const dmatrix& m2) { 
	dmatrix dotP;
	try
	{
		dmatrix m3 = tposeMatrix(m2); // transpose weights so matrices line up correctly to calculate dot product;
		if (m1.at(0).size() != m3.at(0).size()) { // either matrices are not able to be multiplied, or transpose was unnecessary
			if (m1.at(0).size() != m2.at(0).size()) { // checked for original matrices compatibility, still not able.
				throw std::string("Matrices are incompatible for matrix multiplication!"); //throw error message
			}
			else // use original m2 instead of transposed m3 *maybe user already transposed matrix* *maybe user inputted matrices in param such that transpose was unnecessary*
			{
				for (int i = 0; i < m1.size(); ++i) { // for each vector in matrix m1
					dotP.push_back({}); // pushback a empty vector in dotP
					for (int j = 0; j < m2.size(); ++j) { // for each vec in transposed m2
						dotP.at(i).push_back(std::inner_product(m1.at(i).begin(), m1.at(i).end(), m2.at(j).begin(), 0.0)); // calculate the dot product for dotP for each vector in m1 & m2
					}
				}
				return dotP;
			}
		}
		else
		{

			for (int i = 0; i < m1.size(); ++i) { // for each vector in matrix m1
				dotP.push_back({}); // pushback a empty vector in dotP
				for (int j = 0; j < m3.size(); ++j) { // for each vec in transposed m2
					dotP.at(i).push_back(std::inner_product(m1.at(i).begin(), m1.at(i).end(), m3.at(j).begin(), 0.0)); // calculate the dot product for dotP for each vector in m1 & m2
				}
			}
			return dotP;
		}
	}
	catch (const std::string& error)
	{
		std::cerr << "ERROR || " << error;
	}
	catch (std::exception& e)
	{
		std::cerr << "ERROR || " << e.what();
	}
	
}

dmatrix matrixMult(const dmatrix& m1, const dmatrix& m2) {
	dmatrix Mult;
	for (int i = 0; i < m1.size(); ++i) { // for each vector in matrix m1
		Mult.push_back({}); // pushback a empty vector in Mult
		for (int j = 0; j < m1.at(0).size(); ++j) { // for each value at the current vector in m1
			Mult.at(i).push_back(m1[i][j] * m2[i][j]); // Multiply for each vector in m1 & m2
		}
	}
	return Mult;
}

dmatrix matrixAdd(const dmatrix& m1, const dmatrix& m2) {
	dmatrix Add;
	for (int i = 0; i < m1.size(); ++i) { // for each vector in matrix m1
		Add.push_back({}); // pushback a empty vector in Add
		for (int j = 0; j < m1.at(0).size(); ++j) { // for each value at the current vector in m1
			Add.at(i).push_back(m1[i][j] + m2[i][j]); // Add for each vector in m1 & m2
		}
	}
	return Add;
}

dmatrix matrixSub(const dmatrix& m1, const dmatrix& m2) {
	dmatrix Sub;
	for (int i = 0; i < m1.size(); ++i) { // for each vector in matrix m1
		Sub.push_back({}); // pushback a empty vector in Add
		for (int j = 0; j < m1.at(0).size(); ++j) { // for each value at the current vector in m1
			Sub.at(i).push_back(m1[i][j] - m2[i][j]); // Add for each vector in m1 & m2
		}
	}
	return Sub;
}

dmatrix matrixSet(const dmatrix& m, const double& d) {
	dmatrix Set;
	for (int i = 0; i < m.size(); ++i) { // for each vector in matrix m1
		Set.push_back({}); // pushback a empty vector in Add
		for (int j = 0; j < m.at(0).size(); ++j) { // for each value at the current vector in m1
			Set.at(i).push_back(d); // Add for each vector in m1 & m2
		}
	}
	return Set;
}

dmatrix matrixVecAddition(const dmatrix& m, const drow& row) { 
	dmatrix matrixA;
	for (int i = 0; i < m.size(); ++i) { // for each vector in matrix m
		matrixA.push_back({}); // pushback a empty vector in matrixA
		for (int j = 0; j < m.at(i).size(); ++j) {
			matrixA.at(i).push_back(m.at(i).at(j) + row.at(j)); // for each value in the given matrix, add the row's value at j to it
		}
	}
	return matrixA;
}

dmatrix toMatrix(const drow & r)
{
	dmatrix toMat;
	toMat.push_back({});
	for (auto& val : r) {
		toMat.at(0).push_back(val);
	}

	return toMat;
}

drow fromMatrix(const dmatrix & m)
{
	drow fromMat;
	for (auto& val : m.at(0)) { // only uses the first vec in the dmatrix, so other vec's values will be lost
		fromMat.push_back(val);
	}

	return fromMat;
}

dmatrix operator+=(dmatrix& m1, const dmatrix& m2)
{
	for (int i = 0; i < m1.size(); ++i) { // for each vector in matrix m1
		for (int j = 0; j < m1.at(0).size(); ++j) { // for each value at the current vector in m1
			m1[i][j] = m1[i][j] + m2[i][j]; // Add for each vector in m1 & m2
		}
	}
	return m1;
}

std::ostream& operator<<(std::ostream& out, const drow& r) 
{
	out << " [";
	for (auto& val : r) {
		out << std::setw(15) << val << " ";
	}
	out << "]\n";

	return out;
}

std::ostream& operator<<(std::ostream& out, const dmatrix& m) 
{
	out << "\n[\n";
	for (auto& vec : m) {
		out << vec;
	}
	out << "]\n";

	return out;
}

double random(const double& min, const double& max) { 

	std::mt19937_64 rng{};
	rng.seed(std::random_device{}());
	return std::uniform_real_distribution<>{min, max}(rng);
}

dmatrix activationReLU(const dmatrix& m) 
{
	dmatrix finalOutputs;
	for (int i = 0; i < m.size(); ++i) {
		finalOutputs.push_back({});
		for (int j = 0; j < m.at(i).size(); ++j) {
			finalOutputs.at(i).push_back(std::max<double>(0, m[i][j])); // for each singular value in the matrix, place the max of 0 and the value in the new matrix
		}
	}
	return finalOutputs;
}

dmatrix activationSoftMax(const dmatrix& m) 
{
	dmatrix expOutputs; // this matrix contains the values of the each output neuron with e^x, where e is euler's number, and x is the value of the neuron subtracted by the largest value neuron in the batch
	// Ex. { 1, 2, 0 } -> subtracted by largest { -1, 0, -2 } -> exp with euler's number { 0.368, 1, 0.135 } -> normalized with totals to produce final outputs { .245, .665, .090 }, the sum of these = 1!
	drow normalizeTotals(m.size()); // this matrix contains the totals of each output neuron per each vector, and is used to normalize the values
	for (int i = 0; i < m.size(); ++i) {
		double largest = -1000000;
		expOutputs.push_back({});
		for (int x = 0; x < m.at(i).size(); ++x) {
			if (m[i][x] >= largest) {
				largest = m[i][x]; // set largest to whatever the output batch's highest value neuron is
			}
		}
		for (int j = 0; j < m.at(i).size(); ++j) {
			expOutputs.at(i).push_back(exp(m[i][j] - largest)); // for each value, take euler's number to the power of the value - the largest value in the current vector/batch of neurons
			normalizeTotals.at(i) += expOutputs[i][j]; // calculate the total of all the values for each batch of neurons
		}
	}
	dmatrix finalOutputs;
	for (int i = 0; i < expOutputs.size(); ++i) {
		finalOutputs.push_back({});
		for (int j = 0; j < expOutputs.at(i).size(); ++j) {
			finalOutputs.at(i).push_back(expOutputs[i][j] / normalizeTotals.at(i)); // all values in each batch to add up to 1, normalizing them to % values
		}
	}
	return finalOutputs;
}

NetworkLayer::NetworkLayer(const int numOfInputs, const int numOfNeurons)
	: weights{ dmatrix(numOfInputs, drow(numOfNeurons)) }, // create matrix with #input rows & #output columns
	biases{ drow(numOfNeurons, 0) } // num biases are = to amount of outputs (& initialized to 0)
{
	for (int i = 0; i < weights.size(); ++i) {
		for (int j = 0; j < weights.at(i).size(); ++j) {
			weights.at(i).at(j) = random(-1, 1); // each weight should be initialized to a random # between -1 & 1
		}
	}
}
