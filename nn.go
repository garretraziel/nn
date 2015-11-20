package nn

import (
    "fmt"
    "math/rand"
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
        result += fmt.Sprintf("%2d", layer)
    }
    for i, weights := range network.weights {
        result += fmt.Sprintf("\nweights layer %d to %d:\n%s", i + 1, i, weights.String())
    }
    for i, biases := range network.biases {
        result += fmt.Sprintf("\nbiases layer %d:\n%s", i + 1, biases.String())
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

// Evaluate returns ratio of correctly clasified inputs
func (network NN) Evaluate(inputs []TrainItem) float64 {
    correct := 0
    for _, input := range inputs {
        output := network.FeedForward(input.Values)
        max, err := output.MaxAt()
        if err != nil {
            panic(err)
        }
        if float64(max) == input.Label {
            correct++
        }
    }
    return float64(correct) / float64(len(inputs))
}

// Train trains Network on given input with given settings
func (network NN) Train(inputs []TrainItem, epochs int, miniBatchSize int, eta float64) {
    inputCount := len(inputs)
    for i := 0; i < epochs; i++ {
        shuffled := make([]TrainItem, inputCount)
        perm := rand.Perm(inputCount)
        for i, v := range perm {
            shuffled[v] = inputs[i]
        }

        batchesCount := int(float64(inputCount) / float64(miniBatchSize) + 0.5)
        batches := make([][]TrainItem, batchesCount)
        for i := 0; i < batchesCount; i++ {
            if i + miniBatchSize >= inputCount {
                batches[i] = shuffled[i*miniBatchSize:]
            } else {
                batches[i] = shuffled[i*miniBatchSize:i*miniBatchSize + miniBatchSize]
            }
        }

        for _, batch := range batches {
            network.updateMiniBatch(batch, eta)
        }
    }
}

func (network NN) updateMiniBatch(batch []TrainItem, eta float64) {
    var err error
    cxw := make([]matrices.Matrix, len(network.weights))
    cxb := make([]matrices.Matrix, len(network.biases))
    for i, m := range network.weights {
        cxw[i] = matrices.InitMatrix(m.Rows(), m.Cols())
    }
    for i, m := range network.biases {
        cxb[i] = matrices.InitMatrix(m.Rows(), m.Cols())
    }

    for _, item := range batch {
        deltaW, deltaB := network.backprop(item)
        for i, delta := range deltaW {
            cxw[i], err = cxw[i].Add(delta)
            if err != nil {
                panic(err)
            }
        }
        for i, delta := range deltaB {
            cxb[i], err = cxb[i].Add(delta)
            if err != nil {
                panic(err)
            }
        }
    }
    multByConst := matrices.Mult(eta / float64(len(batch)))
    for i, w := range cxw {
        reduced := w.Apply(multByConst)
        network.weights[i], err = network.weights[i].Sub(reduced)
        if err != nil {
            panic(err)
        }
    }
    for i, b := range cxb {
        reduced := b.Apply(multByConst)
        network.biases[i], err = network.biases[i].Sub(reduced)
        if err != nil {
            panic(err)
        }
    }
}

func (network NN) backprop(item TrainItem) ([]matrices.Matrix, []matrices.Matrix) {
    deltaW := make([]matrices.Matrix, len(network.weights))
    deltaB := make([]matrices.Matrix, len(network.biases))
    for i, m := range network.weights {
        deltaW[i] = matrices.InitMatrix(m.Rows(), m.Cols())
    }
    for i, m := range network.biases {
        deltaB[i] = matrices.InitMatrix(m.Rows(), m.Cols())
    }

    activation := item.Values
    activations := make([]matrices.Matrix, len(network.weights) + 1)
    activations[0] = activation
    zs := make([]matrices.Matrix, len(network.weights))

    for i := range network.weights {
        weights := network.weights[i]
        biases := network.biases[i]
        multiplied, err := activation.Dot(weights)
        if err != nil {
            panic(err)
        }
        z, err := multiplied.Add(biases)
        if err != nil {
            panic(err)
        }
        zs[i] = z
        activation = z.Sigmoid()
        activations[i + 1] = activation
    }

    y, err := matrices.OneHotMatrix(1, network.layers[len(network.layers) - 1], 0, int(item.Label))
    if err != nil {
        panic(err)
    }

    delta, err := activations[len(activations) - 1].Sub(y)
    if err != nil {
        panic(err)
    }

    return deltaW, deltaB
}
