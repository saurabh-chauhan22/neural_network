from csv import reader					
from random import shuffle				 
from random import randrange			
from Perceptron import Perceptron		

def load_csv(filename):
	dataset = list()

	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue

			dataset.append(row)

	return dataset


def convert_inputs_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())


def convert_desired_outputs_to_int(dataset, column):
	class_values = [row[column] for row in dataset]

	unique = set(class_values)

	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i

	for row in dataset:
		row[column] = lookup[row[column]]
	
	return lookup


def load_dataset(filename):
	dataset = load_csv(filename)

	for column in range(len(dataset[0])-1):
		convert_inputs_to_float(dataset, column)

	convert_desired_outputs_to_int(dataset, len(dataset[0]) - 1)


def create_training_set(dataset):
    '''
    Create the training set
		-Training set will consist of the specified percent fraction of the dataset
		-How many inputs you decide to use for the training set, and how you choose
		 those values, is entirely up to you
	
	Params:	dataset - the entire dataset
	
	Returns:	a matrix, or list of rows, containing only a subset of the input
				vectors from the entire dataset
    '''
    train_size = int(len(dataset) * 0.8)
    train_set = []
    dataset_copy = list(dataset)
    shuffle(dataset_copy) # for better 
    while len(train_set) < train_size:
        index = randrange(len(dataset_copy))
        train_set.append(dataset_copy.pop(index))
    return train_set

dataset = load_csv('sonar_all-data.csv')
for column in range(len(dataset[0])-1):
	converted_values_to_float = convert_inputs_to_float(dataset,column)

convert_values_outputs_to_int = convert_desired_outputs_to_int(dataset,len(dataset[0]) - 1)

training_set = create_training_set(dataset)

bias = 0.0
perceptron = Perceptron(bias, [0.0]*(len(dataset[0]) - 1))

perceptron.train(training_set,0.001,1000)

accuracy = perceptron.test(training_set)

print(f"Perceptron Accuracy: {accuracy:.2f}%")





