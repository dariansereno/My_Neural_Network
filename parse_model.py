import json
import deeplearningkit.loss as Loss
import deeplearningkit.layer as Layer
import deeplearningkit.optimizer as Optimizer
from deeplearningkit.model import Model
import deeplearningkit.activation as Activation
import deeplearningkit.initializer as Initializer
import pandas as pd
from preprocess import get_data

def extract_csv(filepath: str):
  return pd.read_csv(filepath, header=None)

def get_model_json(filename):
	with open(filename, 'r') as file:
		json_data = json.load(file)
	return json_data

class ModelConfigurationError(Exception):
	pass


def parse_model_json(filename: str) -> dict:
	try:
		json_model: dict = get_model_json(filename)
		
		field_validators = {
			'loss_function': validate_loss_function,
			'layers': validate_layers,
			'epochs': validate_epochs,
			'batch_size': validate_batch_size,
			'optimizer': validate_optimizer,
			'data': validate_csv_data,
		}
		
		for field, validator in field_validators.items():
			if field not in json_model:
				raise ModelConfigurationError(f"No {field} provided.")
			validator(json_model[field])
		
		loss = Loss.loss(json_model['loss_function'])
		layers:list[Layer.Layer] = []
		activations: list[Activation.Activation] = []
		for index, l in enumerate(json_model['layers']):
			if index ==  0:
				n_inputs = 0
			else:
				try:
					n_inputs = int(json_model['layers'][index -  1]['units'])
				except Exception as e:
					raise ModelConfigurationError("Units is not a number")
			layers.append(Layer.layer(l['type'], n_inputs, l['units'], Initializer.initializer(l['weights_initializer']['type'], l['weights_initializer']['params'])))
			activations.append(Activation.activation(l['activation']))
		optimizer = Optimizer.optimizer(json_model['optimizer']['type'], json_model['optimizer']['params'])
		epochs = int(json_model['epochs'])
		batch_size = int(json_model['batch_size'])
		data = extract_csv(json_model['data'])

		return {"loss": loss, "layers": layers, "activations": activations, "optimizer": optimizer, "epochs": epochs, "batch_size": batch_size, "data": data}
	
	except ModelConfigurationError as e:
		print(e)
		exit(1)
#	{"type": "Dense", "units": 24, "activation": "ReLU", "weights_initializer": "heUniform"},

def compile_and_fit_parsed_model(model_data: dict, preprocess_func, display=True, plot=True)-> Model:
	model = Model()

	(train_X, test_X), (train_Y, test_Y) = preprocess_func(model_data["data"])
	model_data["layers"][0].updateInputs(train_X.shape[1])
	for layer, activation in zip(model_data["layers"], model_data["activations"]):
		model.add(layer, activation)
	model.compile(model_data["optimizer"], model_data["loss"])
	model.fit(train_X, train_Y, model_data["batch_size"], epochs=model_data["epochs"], display=display, plot=plot)
	return model

def compile_fit_evaluate_parsed_model(model_data: dict, preprocess_func, display=True, plot=True)-> Model:
	model = Model()

	(train_X, test_X), (train_Y, test_Y) = preprocess_func(model_data["data"])
	model_data["layers"][0].updateInputs(train_X.shape[1])
	for layer, activation in zip(model_data["layers"], model_data["activations"]):
		model.add(layer, activation)
	model.compile(model_data["optimizer"], model_data["loss"])
	model.fit(train_X, train_Y, model_data["batch_size"], epochs=model_data["epochs"], display=display, plot=plot)
	model.evaluate(test_X, test_Y)
	return model

def validate_loss_function(value):
	loss = Loss.loss(value)
	if loss is None:
		raise ModelConfigurationError('Loss function does not exist.')

def validate_layers(layers):
	for index, l in enumerate(layers):
		if 'type' not in l:
			raise ModelConfigurationError('No type provided inside layer field')
		if 'units' not in l:
			raise ModelConfigurationError('No units provided inside layer field')
		if 'activation' not in l:
			raise ModelConfigurationError('No activation provided inside layer field')
		if 'weights_initializer' not in l:
			raise ModelConfigurationError('No weights initializer provided inside layer field')
		if 'type' not in l['weights_initializer']:
			raise ModelConfigurationError('No type provided inside weights initializer field')
		if 'params' not in l['weights_initializer']:
			raise ModelConfigurationError('No params provided inside weights initializer field')
		if index ==  0:
			n_inputs = 0
		else:
			n_inputs = layers[index -  1]['units']
		
		layer = Layer.layer(l['type'], n_inputs, l['units'], Initializer.initializer(l['weights_initializer']['type'], l['weights_initializer']['params']))
		if layer is None:
			raise ModelConfigurationError('Layer does not exist.')
		activation = Activation.activation(l['activation'])
		if activation is None:
			raise ModelConfigurationError('Activation does not exist.')

def validate_epochs(value):
	try:
		epochs = int(value)
	except ValueError:
		raise ModelConfigurationError('Epochs is not a number')

def validate_batch_size(value):
	try:
		batch_size = int(value)
	except ValueError:
		raise ModelConfigurationError('Batch size is not a number')

def validate_optimizer(value):
	if 'type' not in value:
		raise ModelConfigurationError("No type in optimizer field")
	if 'params' not in value:
		raise ModelConfigurationError("No params in optimizer field")
	for param in value["params"]:
		try:
			int(param)
		except Exception as e:
			raise ModelConfigurationError("Optimizer params not a number")
	optimizer = Optimizer.optimizer(value['type'], value['params'])
	if optimizer is None:
		raise ModelConfigurationError('Optimizer does not exist.')

def validate_csv_data(value):
	try:
		data = extract_csv(value)
	except Exception:
		raise ModelConfigurationError('CSV data not valid')

__all__ = ['parse_model_json', 'compile_and_fit_parsed_model', 'compile_fit_evaluate_parsed_model']