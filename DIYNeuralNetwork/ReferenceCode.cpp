
/*
template <typename T1, typename T2>
std::vector<std::tuple<T1, T2>> zip(typename std::vector<T1>::iterator weightsBegin, typename std::vector<T2>::iterator biasesBegin, typename std::vector<T2>::iterator biasesEnd) {
	const auto size = std::distance(static_cast<decltype(biasesEnd)>(biasesBegin), biasesEnd); //size = biases.size(), but decltype is used to make sure all elements in biases are type consistent
	std::vector<std::tuple<T1, T2>> zipped;
	zipped.reserve(size); //reserve the size of the biases vector in the zipped tuple vector
	while (biasesBegin != biasesEnd) {
		zipped.emplace_back(*weightsBegin++, *biasesBegin++); //weightsBegin is actually (most likely) another vector of double weights, so T1 = std::vector<double>, and T2 = double
	}
	return zipped;
}


std::vector<std::vector<double>> inputs = { {2.3,9.1,0.5 } };
	std::vector<std::vector<double>> weights = { {0.1,-1.2,5.4}, {3.2,10.7,-1.6}, {6.8,2.0,-5.3}, {6.8,2.0,-5.3} };
	std::vector<double> biases = { 2.1,-7,1.9,8.6 };

	auto zippedWeightsBiases = zip<decltype(weights)::value_type, double>(weights.begin(), biases.begin(), biases.end());

	std::vector<std::vector<double>> outputs;

	for (auto &inputBatch : inputs) {
		std::vector<double> outputBatchPush;
		for (auto &weightsBias : zippedWeightsBiases) {
			auto &weightVec = std::get<0>(weightsBias);
			double nOutput = 0;
			nOutput += (std::inner_product(std::begin(weightVec), std::end(weightVec), std::begin(inputBatch), 0.0) + std::get<1>(weightsBias));
			outputBatchPush.emplace_back(nOutput);
		}
		outputs.emplace_back(outputBatchPush);
	}

	for (auto &outputBatch : outputs) {
		for (auto val : outputBatch) {
			std::cout << val << std::endl;
		}
	}
	std::cin.get();*/












	////calc derivative of each output's error with respect to each output.
				//dmatrix derivOutErrors;
				//for (int i = 0; i < outputs.size(); ++i) {
				//	derivOutErrors.push_back({});
				//	for (int j = 0; j < outputs.at(0).size(); ++j) {
				//		derivOutErrors.at(i).push_back( -(tarOutputs[i][j] * (1 / outputs[i][j])) + (1 - tarOutputs[i][j]) * (1 / (1 - outputs[i][j])) ); // calculates loss of each output by using deriv of loss function with respect to output
				//	}
				//}
				//std::cout << "\n" << derivOutErrors;

				//activInputs = { {1.8658, 2.2292, 2.8204} };
				//std::cout << activInputs;
				////calc all pre-softmax vals with euler's num (used for calculating the deriv
				//dmatrix inExpd;
				//for (int i = 0; i < activInputs.size(); ++i) {
				//	inExpd.push_back({});
				//	for (int j = 0; j < activInputs.at(0).size(); ++j) {
				//		inExpd.at(i).push_back(exp(activInputs[i][j]));
				//	}
				//}
				//std::cout << "presoftmax euler's" << inExpd;
				// calc deriv of each output neuron with respect to its value before softmax (deriv of softmax)
				//dmatrix derivOutIn;
				//for (int i = 0; i < outputs.size(); ++i) {
				//	derivOutIn.push_back({});
				//	for (auto ptr = inExpd.at(i).begin(); ptr < inExpd.at(i).end(); ++ptr) {
				//		derivOutIn.at(i).push_back(((*ptr) * (std::accumulate(inExpd.at(i).begin(), inExpd.at(i).end(), 0.0) - (*ptr))) / pow(std::accumulate(inExpd.at(i).begin(), inExpd.at(i).end(), 0.0), 2)); // deriv of softmax for each output neuron
				//	}
				//}

				/*std::cout << "\n" << derivOutIn;
				std::cout << (inExpd.at(0).at(2) * (inExpd.at(0).at(0) + inExpd.at(0).at(1))) / pow((inExpd.at(0).at(2) + inExpd.at(0).at(0) + inExpd.at(0).at(1)),2);*/

				// cacl deriv of Error with respect to softMax inputs
				/*dmatrix errorSoftIn;
				for (int i = 0; i < outputs.size(); ++i) {
					errorSoftIn.push_back({});
					for (int j = 0; j < outputs.at(0).size(); ++j) {
						errorSoftIn.at(i).push_back(outputs[i][j] - tarOutputs[i][j]);
					}
				}*/
				//std::cout << errorSoftIn;
				// calc deriv of pre softMax with respek to weights of hiddenlayer
				// deriv of x^/W = inputs
				//std::cout << "\n" << inputs;


//// Finally, the deriv of Error/Weights is = to deriv Error/ouput * deriv output/softmax input * deriv softmax input/Weights
			////dmatrix changeWeights = matrixMult(derivOutErrors, derivOutIn);
			//dmatrix tposeHL = tposeMatrix(inputs);

			//dmatrix changeWeights = dotProduct(tposeHL, errorSoftIn);
			////std::cout << changeWeights;





 // ****************************** THIS CODE IS THE SAME AS THE DOTPRODUCT OF THE X^ * LAST HIDDEN LAYER INPUTS (DERIV) *******************

//dmatrix dAvg;
//double batchTotal = 0;
//dAvg.push_back({});
//for (int i = 0; i < dpreSoftAvg.at(0).size(); ++i) {
//	for (int j = 0; j < dpreSoftAvg.size(); ++j) {
//		batchTotal += dpreSoftAvg[j][i];
//	}
//	dAvg.at(0).push_back(batchTotal / dpreSoftAvg.size());
//	batchTotal = 0;
//}
//for (int i = 0; i < dpreSoftAvg.size() - 1; ++i) {
//	dAvg.push_back(dAvg.at(0));
//}
//
////std::cout << dAvg;
//std::cout << inputs;
//
//std::vector<dmatrix> dWAvgs;
//for (int i = 0; i < dpreSoftAvg.size(); ++i) {
//	//for (int j = 0; j < inputs.size(); ++j) {
//	dWAvgs.push_back(dotProduct(tposeMatrix(toMatrix(dpreSoftAvg.at(i))), toMatrix(inputs.at(i))));
//	//}
//}
//dmatrix dWAvg = dWAvgs.at(0);
//for (int i = 1; i < dWAvgs.size(); ++i) {
//	dWAvg += dWAvgs.at(i);
//}
////dmatrix dWAvg = dotProduct(tposeMatrix(dAvg), inputs);
//std::cout << dWAvg;