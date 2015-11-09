package nn

import (
    "strconv"
    "github.com/garretraziel/matrices"
)

// NN represents neural network to be used with backpropagation
type NN struct {
    layers []int
    weights []matrices.Matrix
    biases []matrices.Matrix
}

// InitNN creates new neural network with given number of layers, neurons in each layer and initalizes them randomly
func InitNN(layers []int) NN {
    biases := make([]matrices.Matrix, len(layers) - 1)
    weights := make([]matrices.Matrix, len(layers) - 1)

    for i := range layers[1:] {
        biases[i] = matrices.RandInitMatrix(1, layers[i + 1])
    }

    for i := range layers[1:] {
        weights[i] = matrices.RandInitMatrix(layers[i], layers[i + 1])
    }

    net := NN{layers, weights, biases}
    return net
}

func (network NN) String() (result string) {
    result = "Neural network:\n"
    result += "layers:"
    for _, layer := range network.layers {
        result += " " + strconv.Itoa(layer)
    }
    result += "\n\nweights:"
    for _, weights := range network.weights {
        result += "\n" + weights.String()
    }
    result += "\n\nbiases:"
    for _, biases := range network.biases {
        result += "\n" + biases.String()
    }

    return
}

// FeedForward returns output of given Network on given input
func (network NN) FeedForward(input matrices.Matrix) matrices.Matrix {
    lastOutput := input
    for i := range network.weights {
        weights := network.weights[i]
        biases := network.biases[i]
        multiplied, err := lastOutput.Dot(weights)
        if err != nil {
            panic(err)
        }
        added, err := multiplied.Add(biases)
        if err != nil {
            panic(err)
        }
        lastOutput = added.Sigmoid()
    }
    return lastOutput
}

// func (network NN) Train(inputs []matrices.Matrix, labels matrices.Matrix, epochs int, mini_batch_size int, eta float64) {
//
// }
