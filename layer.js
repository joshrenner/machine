'use strict'

var math = require('forwardjs-ml-math')
var Neuron = require('./neuron')

module.exports = class Layer {

    constructor(size) {
        this.neurons = []
        for (var i = 0; i < size; i++) {
            var neuron = new Neuron()
            this.neurons.push(neuron)
        }
    }

    forward(inputs) {
        var outputs = this.neurons.map(n => n.forward(inputs))
        return outputs
    }

    backward(errors) {
        var allBackwardErrors = []
        for (var i = 0; i < this.neurons.length; i++) {
            var neuron = this.neurons[i]
            var error = errors[i]
            var backwardError = neuron.backward(error)
            allBackwardErrors.push(backwardError)
        }

        var totalBackwardErrors = allBackwardErrors[0]
        for (var i=0; i < allBackwardErrors.length; i++) {
            totalBackwardErrors = math.arrayAdd(
                totalBackwardErrors,
                allBackwardErrors[i]
            )
        }

        return totalBackwardErrors
    }

    updateWeights() {
        this.neurons.forEach(n => n.updateWeights())
    }
}
