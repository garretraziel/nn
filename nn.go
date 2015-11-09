package nn

import (
    "strconv"
    "math/rand"
    "github.com/garretraziel/matrices"
)

// TrainItem represents one item for training of neural network
type TrainItem struct {
    Values matrices.Matrix
    Label float64
}

// InitTrainItem initializes new training item - values and label
func InitTrainItem(values []float64, label float64) (TrainItem, error) {
    matrix, err := matrices.InitMatrixWithValues(1, len(values), values)
    return TrainItem{matrix, label}, err
}

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
                batches[i] = inputs[i*miniBatchSize:]
            } else {
                batches[i] = inputs[i*miniBatchSize:i*miniBatchSize + miniBatchSize]
            }
        }

        for _, batch := range batches {
            network.updateMiniBatch(batch, eta)
        }
    }
}

func (network NN) updateMiniBatch(batch []TrainItem, eta float64) {
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
        cxw = cxw.Add(deltaW)
        cxb = cxb.Add(deltaB)
    }
    for i, w := range cxw {

        network.weights[i] = network.weights[i].Sub()
    }
}
