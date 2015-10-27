package nn

import (
    "math/rand"
    "github.com/garretraziel/matrices"
)

// NN represents neural network to be used with backpropagation
type NN struct {
    layers []int
    weights []matrices.Matrix
    biases []matrices.Vector
}

// InitNN creates new neural network with given number of layers, neurons in each layer and initalizes them randomly
func InitNN(layers []int, r *rand.Rand) *NN {
    biases := make([]matrices.Vector, len(layers) - 1)
    for i := range layers[1:] {
        biases[i] = matrices.RandInitVector(layers[i])
    }
    net := NN{layers: layers, weights}
    // TODO
    return &net
}
