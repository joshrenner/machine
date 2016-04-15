const math = require('forwardjs-ml-math')
const Layer = require('./layer')
const Neuron = require('./neuron')
const log = console.log

var hiddenLayer = new Layer(2)
var outputLayer = new Layer(1)

var iterations = 100000

var data = [
	{
		input: [0, 0],
		output: [0]
	},
	{
		input: [1, 0],
		output: [0]
	},
	{
		input: [0, 1],
		output: [0]
	},
	{
		input: [1, 1],
		output: [1]
	}
]


for (var i = 0; i < iterations; i++) {
	var j = Math.floor(Math.random() * data.length)
	var input = data[j].input
	var output = data[j].output

	input = [1, ...input]

	var hiddenLayerOutput = hiddenLayer.forward(input)
	hiddenLayerOutput = [1, ...hiddenLayerOutput]
	var outputLayerOutput = outputLayer.forward(hiddenLayerOutput)

	var errors = math.arraySubtract(outputLayerOutput, output)

	var hiddenLayerErrors = outputLayer.backward(errors)
	hiddenLayer.backward(hiddenLayerErrors)

	hiddenLayer.updateWeights()
	outputLayer.updateWeights()

	if (i % Math.floor(iterations/100) === 0) log('accuracy at iteration %s -> %s', i, accuracy())
}

log('I AM SKYNET =============')
for (var i = 0; i < data.length; i++) {
	var input = data[i].input;
	input = [1, ...input];

	var hiddenLayerOutput = hiddenLayer.forward(input)
	hiddenLayerOutput = [1, ...hiddenLayerOutput]
	var outputLayerOutput = outputLayer.forward(hiddenLayerOutput)

	console.log('%s -> %s', data[i].input, outputLayerOutput);
}

function accuracy() {
	var correct = 0;
	for (var i = 0; i < data.length; i++) {
		var input = data[i].input
		var output = data[i].output

		input = [1, ...input]

		var hiddenLayerOutput = hiddenLayer.forward(input)
		hiddenLayerOutput = [1, ...hiddenLayerOutput]
		var outputLayerOutput = outputLayer.forward(hiddenLayerOutput)

		outputLayerOutput = outputLayerOutput[0] > 0.5 ? 1 : 0
		if (outputLayerOutput === output) correct++
	}
	return correct / data.length;
}
