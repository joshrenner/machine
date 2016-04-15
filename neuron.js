'use strict'

var math = require('forwardjs-ml-math')

var stepSize = 0.1

module.exports = class Neuron {
    constructor() {
        this.weights = []
        for (var i = 0; i < 3; i++) {
            var weight = Math.random() - 0.5
            this.weights.push(weight)
        }
    }

    forward(inputs) {
        this.inputs = inputs
        this.z = math.arrayMultiply(inputs, this.weights)
        return math.sigmoid(this.z)
    }

    /**
     * @param (Number) error
     * @return {Array}
     */
    backward(error) {
		this.error = error
        var backErrors = this.weights.map(w => w * error)
        return backErrors.slice(1)
    }

    updateWeights() {
        var deltas = this.inputs.map(input => this.error * input * math.sigmoidGradient(this.z) * stepSize)
        this.weights = math.arraySubtract(this.weights, deltas)
    }
}
