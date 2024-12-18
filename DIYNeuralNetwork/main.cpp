#include <vector>
#include <iostream>
#include <string>
#include <iterator>
#include <numeric>
#include <algorithm>
#include <time.h>

#include "NeuralNetwork.h"


/************************************* PURPOSE/GOAL: ***********************************
						* Given two numbers (wins/losses of a NBA team) determine:
							* Whether or not they will make the playoffs. (or probability of whether they will make the playoffs)

						* Other options:
							*Give more input neurons (# of allstars on team, Coach win percentage, etc...)
							*More/different output neurons (% to win finals, to make a trade, etc...)


*/


int main() {
	dmatrix inputTest = { {1,0}, {0,0}, {0,1}, {0,0}, {1,1}, {0,0}, {0,0}, {0,1}, {0,0}, {1,1}, {1,0}, {1,1} };
	std::vector<int> hiddenLayers = { 2 };
	dmatrix tarOutputTest = { {1,0}, {0,1}, {1,0}, {0,1}, {1,0}, {0,1}, {0,1}, {1,0}, {0,1}, {1,0}, {1,0}, {1,0} };

	NeuralNetwork batchtest1(inputTest, hiddenLayers, tarOutputTest, 1);
	//NeuralNetwork batchtest2(inputTest, hiddenLayers, tarOutputTest, 2);
	NeuralNetwork batchtest3(inputTest, hiddenLayers, tarOutputTest, 3);
	//NeuralNetwork batchtest4(inputTest, hiddenLayers, tarOutputTest, 4);
	//NeuralNetwork batchtest5(inputTest, hiddenLayers, tarOutputTest, 5);
	NeuralNetwork batchtest5(inputTest, hiddenLayers, tarOutputTest, 5);
	NeuralNetwork batchtest7(inputTest, hiddenLayers, tarOutputTest, 7);
	NeuralNetwork batchtestAll(inputTest, hiddenLayers, tarOutputTest);

	std::cout << "1 PER BATCH\n";
	batchtest1.runNetworkBackprop(5,1);
	//batchtest2.runNetworkBackprop();
	std::cout << "3 PER BATCH\n";
	batchtest3.runNetworkBackprop(5, 1);
	//batchtest4.runNetworkBackprop();
	//batchtest5.runNetworkBackprop();
	std::cout << "5 PER BATCH\n";
	batchtest5.runNetworkBackprop(5, 1);
	std::cout << "7 PER BATCH\n";
	batchtest7.runNetworkBackprop(5, 1);
	std::cout << "ALL IN ONE BATCH\n";
	batchtestAll.runNetworkBackprop(5, 1);

	/*dmatrix inputs1 = { {1,0,0,0}, {0,1,1,1}, {0,1,1,0}, {1,0,0,1} };
	dmatrix inputs2 = { {1,1,1,0}, {0,0,1,0}, {0,0,0,0}, {0,0,1,1} };
	dmatrix inputs3 = { {0,0,0,1}, {0,1,0,1}, {1,1,0,1}, {1,1,1,1} };
	std::vector<dmatrix> inputBatches;
	inputBatches.push_back(inputs1);
	inputBatches.push_back(inputs2);
	inputBatches.push_back(inputs3);

	dmatrix inputs = { {1,0,0,0}, {0,1,1,1}, {0,1,1,0}, {1,0,0,1}, {1,1,1,0}, {0,0,1,0}, {0,0,0,0}, {0,0,1,1}, {0,0,0,1}, {0,1,0,1}, {1,1,0,1}, {1,1,1,1} };
	dmatrix tarOutputs = { {0,1}, {1,0},     {0,1},     {0,1},     {1,0},     {0,1},     {0,1},     {0,1},     {0,1},     {0,1},     {1,0},     {1,0} };
	std::vector<dmatrix> tarOutputBatches;
	tarOutputBatches.push_back({ { 0,1 }, { 1,0 }, { 0,1 }, { 0,1 } });
	tarOutputBatches.push_back({ { 1,0 }, { 0,1 }, { 0,1 }, { 0,1 } });
	tarOutputBatches.push_back({ { 0,1 }, { 0,1 }, { 1,0 }, { 1,0 } });

	dmatrix untestedInputs = { {1,1,0,0}, {0,1,0,0}, {1,0,1,0}, {1,0,1,1} };
	dmatrix untestedTarOutputs = { {0,1}, {0,1},     {0,1},     {1,0} };*/

	dmatrix inputs = { {1,0,0}, {0,1,1}, {1,1,1}, {0,1,0} };
	dmatrix tarOutputs = { {0,1}, {1,0}, {1,0},   {1,0} };
	std::cout << inputs;
	FinalLayer WORKBRAIN(3, 2);
	dmatrix firstWeights = WORKBRAIN.getWeights();
	drow firstBiases = WORKBRAIN.getBiases();
	FinalLayer onlyWeights(3, 2);
	FinalLayer onlyBiases(3, 2);
	//onlyWeights.optimizeWeights(WORKBRAIN.getWeights()); 
	//onlyBiases.optimizeWeights(WORKBRAIN.getWeights()); 

	/*FinalLayer WORKBRAINSNGL(3, 2);
	WORKBRAINSNGL.optimizeWeights(WORKBRAIN.getWeights());*/
	std::string run("");
	while (run != "-1")
	{
		int runAmt = 10;
		while (runAmt > 0) {
			try
			{
				/*srand(time(NULL));
				int snglRand = rand() % inputBatches.size();*/
				WORKBRAIN.forwardPass(inputs);
				onlyWeights.forwardPass(inputs);
				onlyBiases.forwardPass(inputs);

				//WORKBRAINSNGL.forwardPass(inputBatches.at(snglRand));
				//std::cout << "************\n" << activInputs;
				dmatrix outputs = WORKBRAIN.getOutputs(); //after applying softMax to activInputs (actually wrong calc, only used for testing purposes)
				//dmatrix outputsSngl = WORKBRAINSNGL.getOutputs(); //after applying softMax to activInputs (actually wrong calc, only used for testing purposes)
				//dmatrix snglTarOutputs = tarOutputBatches.at(snglRand);
				std::cout << "NETWORK INPUTS" << inputs << "\n";
				std::cout << "ORIG NETWORK TARGET OUTPUTS\n" << tarOutputs;
				std::cout << "\t***************\nORIG NETWORK OUTPUTS\n" << outputs;
				drow totalErr = calcLoss(outputs, tarOutputs); // calc total error of network outputs using cross entropy
				std::cout << "TOTAL ERROR: " << totalErr << "AVERAGE: " << calcAvgTotalLoss(outputs, tarOutputs) << "\n\n";
				//std::cout << "SINGLE NETWORK TARGET OUTPUTS\n" << snglTarOutputs;
				//std::cout << "\t***************\nSINGLE BACKPROP NETWORK OUTPUTS\n" << WORKBRAINSNGL.getOutputs();
				//drow singleErr = calcLoss(WORKBRAINSNGL.getOutputs(), snglTarOutputs); // calc total error of network outputs using cross entropy
				//std::cout << "TOTAL ERROR: " << singleErr << "\n";
				std::cout << "\t***************\nONLYWEIGHTS NETWORK OUTPUTS\n" << onlyWeights.getOutputs();
				std::cout << "TOTAL ERROR: " << calcLoss(onlyWeights.getOutputs(), tarOutputs) << "AVERAGE: " << calcAvgTotalLoss(onlyWeights.getOutputs(), tarOutputs) << "\n\n";

				std::cout << "\t***************\ONLYBIASES NETWORK OUTPUTS\n" << onlyBiases.getOutputs();
				std::cout << "TOTAL ERROR: " << calcLoss(onlyBiases.getOutputs(), tarOutputs) << "AVERAGE: " << calcAvgTotalLoss(onlyBiases.getOutputs(), tarOutputs) << "\n\n";
				// deriv of Err with respect to x^ (pre-softMax outputs) = x sub k (network "guess") - y sub k (network target)
				dmatrix dpreSoft = matrixSub(outputs, tarOutputs);
				dmatrix dpreSoftW = matrixSub(onlyWeights.getOutputs(), tarOutputs);
				dmatrix dpreSoftB = matrixSub(onlyBiases.getOutputs(), tarOutputs);

				/*dmatrix dpreSoftSngl = matrixSub(outputsSngl, snglTarOutputs);*/
				//dmatrix dpreSoft = { {-1,1}, {-1,1}, {1,-1}, {1,-1} };
				//std::cout << dpreSoft;
				/*std::cout << dpreSoftSngl;*/

				//std::cout << inputs;

				// deriv of Err with respect to Weights = dpreSoft * inputs
				dmatrix dW = dotProduct(tposeMatrix(dpreSoft), inputs);
				dmatrix dOW = dotProduct(tposeMatrix(dpreSoftW), inputs);

				//std::cout << dW;

				// deriv of Err with respect to Biases is just the avg of the seperate batches deriv of Err with respect to x^
				drow dB;
				double batchTotal = 0;
				for (int i = 0; i < dpreSoft.at(0).size(); ++i) {
					for (int j = 0; j < dpreSoft.size(); ++j) {
						batchTotal += dpreSoft[j][i];
					}
					dB.push_back(batchTotal / dpreSoft.size());
					batchTotal = 0;
				}
				std::cout << dB;

				drow dOB;
				double OBbatchTotal = 0;
				for (int i = 0; i < dpreSoftB.at(0).size(); ++i) {
					for (int j = 0; j < dpreSoftB.size(); ++j) {
						OBbatchTotal += dpreSoftB[j][i];
					}
					dOB.push_back(OBbatchTotal / dpreSoftB.size());
					OBbatchTotal = 0;
				}
				std::cout << dOB;

				//dmatrix dWSingle = dotProduct(tposeMatrix(dpreSoftSngl), inputBatches.at(snglRand));


				double learnRate = 1.0;
				std::cout << WORKBRAIN.getWeights();
				std::cout << WORKBRAIN.getBiases();
				dmatrix dBMat = toMatrix(dB);
				WORKBRAIN.optimizeWeights(matrixSub(WORKBRAIN.getWeights(), matrixMult(tposeMatrix(dW), matrixSet(tposeMatrix(dW), 1.0))));
				WORKBRAIN.optimizeBiases(fromMatrix(matrixSub(toMatrix(WORKBRAIN.getBiases()), matrixMult(dBMat, matrixSet(dBMat, 1.0)))));

				std::cout << onlyWeights.getWeights();
				onlyWeights.optimizeWeights(matrixSub(onlyWeights.getWeights(), matrixMult(tposeMatrix(dOW), matrixSet(tposeMatrix(dOW), 1.0))));
				
				std::cout << onlyBiases.getBiases();
				dmatrix dOBMat = toMatrix(dOB);
				onlyBiases.optimizeBiases(fromMatrix(matrixSub(toMatrix(onlyBiases.getBiases()), matrixMult(dOBMat, matrixSet(dOBMat, 1.0)))));
				/*std::cout << WORKBRAINSNGL.getWeights();
				WORKBRAINSNGL.optimizeWeights(matrixSub(WORKBRAINSNGL.getWeights(), matrixMult(tposeMatrix(dWSingle), matrixSet(tposeMatrix(dWSingle), 1.0))));*/

				--runAmt;
			}
			catch (const std::string& e)
			{
				std::cerr << e;
				std::cin.get();
			}
			
		}
		WORKBRAIN.forwardPass(inputs);
		//WORKBRAINSNGL.forwardPass(inputs);
		std::cout << "NETWORK INPUTS" << inputs;
		std::cout << "NETWORK TARGET OUTPUTS" << tarOutputs;
		std::cout << "\t***************\nFINAL ORIG NETWORK OUTPUTS\n" << WORKBRAIN.getOutputs();
		drow totalErr = calcLoss(WORKBRAIN.getOutputs(), tarOutputs); // calc total error of network outputs using cross entropy
		double avgErr = calcAvgTotalLoss(WORKBRAIN.getOutputs(), tarOutputs);
		std::cout << "FINAL ORIG NETWORK TOTAL ERROR: " << totalErr << "AVERAGE: " << avgErr << "\n\n";
		std::cout << "BEGINNING WEIGHTS: " << firstWeights << "CURRENT WEIGHTS: " << WORKBRAIN.getWeights();
		std::cout << "BEGINNING BIASES: " << firstBiases << "CURRENT BIASES: " << WORKBRAIN.getBiases();

		std::cout << "\t***************\nFINAL ONLYWEIGHTS NETWORK OUTPUTS\n" << onlyWeights.getOutputs();
		std::cout << "FINAL ONLYWEIGHTS NETWORK TOTAL ERROR: " << calcLoss(onlyWeights.getOutputs(), tarOutputs) << "AVERAGE: " << calcAvgTotalLoss(onlyWeights.getOutputs(), tarOutputs) << "\n\n";
		std::cout << "BEGINNING WEIGHTS: " << firstWeights << "CURRENT WEIGHTS: " << onlyWeights.getWeights();

		std::cout << "\t***************\nFINAL ONLYBIASES NETWORK OUTPUTS\n" << onlyBiases.getOutputs();
		std::cout << "FINAL ONLYBIASES NETWORK TOTAL ERROR: " << calcLoss(onlyBiases.getOutputs(), tarOutputs) << "AVERAGE: " << calcAvgTotalLoss(onlyBiases.getOutputs(), tarOutputs) << "\n\n";
		std::cout << "BEGINNING BIASES: " << firstBiases << "CURRENT BIASES: " << onlyBiases.getBiases();

		//std::cout << "\t***************\nFINAL SINGLE BACKPROP NETWORK OUTPUTS\n" << WORKBRAINSNGL.getOutputs();
		//drow singleErr = calcLoss(WORKBRAINSNGL.getOutputs(), tarOutputs); // calc total error of network outputs using cross entropy
		//double singleAvgErr = calcAvgTotalLoss(WORKBRAINSNGL.getOutputs(), tarOutputs);
		//std::cout << "FINAL SINGLE NETWORK TOTAL ERROR: " << singleErr << "AVERAGE: " << singleAvgErr << "\n";



		//WORKBRAIN.forwardPass(untestedInputs);
		//WORKBRAINSNGL.forwardPass(untestedInputs);
		//std::cout << "UNTESTED NETWORK INPUTS" << untestedInputs;
		//std::cout << "NETWORK TARGET OUTPUTS" << untestedTarOutputs;
		//std::cout << "\t***************\nUNTESTED ORIG NETWORK OUTPUTS\n" << WORKBRAIN.getOutputs();
		//std::cout << "UNTESTED ORIG NETWORK TOTAL ERROR: " << calcLoss(WORKBRAIN.getOutputs(), untestedTarOutputs) << "AVERAGE: " << calcAvgTotalLoss(WORKBRAIN.getOutputs(), untestedTarOutputs) << "\n\n";
		//std::cout << "\t***************\nUNTESTED SINGLE BACKPROP NETWORK OUTPUTS\n" << WORKBRAINSNGL.getOutputs();
		//std::cout << "UNTESTED SINGLE NETWORK TOTAL ERROR: " << calcLoss(WORKBRAINSNGL.getOutputs(), untestedTarOutputs) << "AVERAGE: " << calcAvgTotalLoss(WORKBRAINSNGL.getOutputs(), untestedTarOutputs) << "\n";

		std::cout << "Would you like to run again??\n";
		std::cin >> run;
	}
		
	

	/*NeuralNetwork Test(inputs, hiddenLayers, targetOutputs);
	NeuralNetwork Test2(inputs, hiddenLayers, targetOutputs);
	NeuralNetwork Test3(inputs, hiddenLayers, targetOutputs);
	Test.runNetwork(10, 4);*/
	//std::cout << Test.getOutputs();
	//std::cout << Test.getLoss();
	//Test2.runNetwork();
	//Test3.runNetwork();

	//std::cout << Test.getOutputs() << "\n";
	//std::cout << Test.getLoss() << "\n\n";

	//std::cout << Test2.getOutputs() << "\n\n";
	//std::cout << Test2.getLoss() << "\n\n";

	//std::cout << Test3.getOutputs() << "\n\n";
	//std::cout << Test3.getLoss() << "\n\n";


	std::cin.get();

	/*dmatrix inputs = { {1, 1, 1} };
	dmatrix targetOutputs = { {1, 0, 0} };
	std::vector<int> hiddenLayers = { 5, 2 };
	NeuralNetwork Test(inputs, hiddenLayers, targetOutputs);

	Test.runNetwork();
	std::cout << Test.getOutputs() << "\n\n" << Test.getLoss();
	
	std::cin.get();*/


	return 0;
}