# Discrete-hawakes-process-on-time-series
The code for the mode of DHP
## Files explanation
### model_DHP.py
The discrete Hawkes process model file. It defines the fiting function and prediting function.
### baseline_mu.py
Return the baseline value
### infectivity_A.py
Return the infectivity matrix A
### decay.py
Define the decay function phi
### other_layers.py
Define the loss fuction
### dataset.py and data_operation.py
Return the input dataset
## Usage
Using the train.py to make a new model for different training set. After obtaining a trained model, we can run prediton.py to prediction the futrue data. 
There is a trained model meta_full_IP_c3.pt for IP data of cluster 3. You can run the prediction.py and load this model to predict.
