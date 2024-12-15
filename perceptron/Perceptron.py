import logging

from typing import List

logger = logging.getLogger(__name__)

class Perceptron:
	'''
	Class to Create a new Perceptron
	'''
	def __init__(self, bias:float, synaptic_weights:List)-> None:
		'''
		# Params:	bias -	arbitrarily chosen value that affects the overall output
		#					regardless of the inputs
		#
		#			synaptic_weights -	list of initial synaptic weights for this Perceptron
		'''
		assert bias is not None
		assert synaptic_weights is not None
		
		self.bias = bias
		self.synaptic_weights = synaptic_weights
	
	def activation_function(self, z:int):
		'''
		# Activation function
		#	Quantizes the induced local field
		#
		# Params:	z - the value of the indiced local field
		#
		# Returns:	an integer that corresponds to one of the two possible output values (usually 0 or 1)
		'''	
		if z >=0:
			return 1
		return 0
	
	def weighted_sum_inputs(self, inputs:list):
		'''
		# Compute and return the weighted sum of all inputs (not including bias)
		#
		# Params:	inputs - a single input vector (which may contain multiple individual inputs)
		#
		# Returns:	a float value equal to the sum of each input multiplied by its
		#			corresponding synaptic weight
		'''
		weighted_sum=0
		
		for i,weight in zip(inputs, self.synaptic_weights):
			weighted_sum += i * weight
		return weighted_sum

	def induced_local_field(self, inputs:list)->int:
		'''
		# Compute the induced local field (the weighted sum of the inputs + the bias)
		#
		# Params:	inputs - a single input vector (which may contain multiple individual inputs)
		#	
		# Returns:	the sum of the weighted inputs adjusted by the bias
		'''
		return self.weighted_sum_inputs(inputs) + self.bias
	
	def predict(self, input_vector)->int:
		'''
		# Predict the output for the specified input vector
		#
		# Params:	input_vector - a vector or row containing a collection of individual inputs
		#
		# Returns:	an integer value representing the final output, which must be one of the two
		#			possible output values (usually 0 or 1)
		'''
		induced_value = self.induced_local_field(input_vector)
		return self.activation_function(induced_value)
	
	def train(self, training_set, learning_rate_parameter, number_of_epochs):
		'''
		# Train this Perceptron
		#
		# Params:	training_set - a collection of input vectors that represents a subset of the entire dataset
		#			learning_rate_parameter - 	the amount by which to adjust the synaptic weights following an
		#										incorrect prediction
		#			number_of_epochs -	the number of times the entire training set is processed by the perceptron
		#
		# Returns:	no return value
		'''
		for epoch in range(number_of_epochs):
			for row in training_set:
				inputs = row[:-1] # every input value other than the last value 
				desired_output = row[-1] # only the last value from the input vectors 
				predict = self.predict(inputs)
				error_signal = desired_output - predict #NOTE: The predicted output of sum of vectors  minus the desired output which is the last training set value
				logger.info("Error signal value: ",error_signal)
				
				# Update the weights with the error to fix the desired output
				# weights (n +1) = weights(n) + learning rate parameter * [desired-output(n) - actual-output(n)] * inputs(n)
				for index in range(len(self.synaptic_weights)):
					self.synaptic_weights[index] += learning_rate_parameter * error_signal * inputs[index]

				self.bias += learning_rate_parameter * error_signal 

	def test(self, test_set):
		'''
		Harnes the test functions	
		# Test this Perceptron
		# Params:	test_set - the set of input vectors to be used to test the perceptron after it has been trained
		#
		# Returns:	a collection or list containing the actual output (i.e., prediction) for each input vector
		'''
		correct = 0
		for row in test_set:
			inputs, desired = (row[:-1]), row[-1]
			prediction = self.predict(inputs)
			if prediction == desired:
				correct +=1
			else:
				logger.info("The prediction value %d is not equal to the expected value %d for the training set row ",prediction,desired)
		return correct / len(test_set) * 100 # Accurracy rate 
