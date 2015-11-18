package nn

import "github.com/garretraziel/matrices"

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
