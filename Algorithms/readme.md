This folder contains all of the different algorithms and classifiers we have implemented. 
1. Decision tree (DT) - This folder contains the basic DT classifiers and code that runs the feature selection
algoritm, pruning and other experiments.
2. KNN - This folder contains the KNN classifer, as well as code for running all of the KNN experminets (feature 
selection, parameter tuning etc.).
3. time_series - This folder contains all of the time series algorithms.
	a. DataTransformation.py - This file contains the implementation of the time series transformations we offer in order to transform
	   your time series to stationary. Each transformation has an inverse transformation that can be applied to it. This library also supports
	   applying several transformations one after the other. 
	b. AlgoRunner.py - Infrastructure for running various time series forecasting algorithms. This runner supports all SARIMA famile models
	   (MA/AR/ARMA/ARIMA/SARIMA), and also implements the rolling forecast algorithm (for model retraining). 
    c. TimeSeriesAnalysis.py - This includes the main function that runs all of the experiments and demonstrations for the time sereis analysis
	   chapter. In addition here is where we preprocess all of the data needed for the time series algorithms.
	   
All code has been written by Eyal Lotan, Dor Sura and Tomer Haber.

	   
	   