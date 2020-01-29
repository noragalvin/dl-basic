package main

import "math"

type NeutralNetwork struct {
	LearningRate   float64
	NumOfIteration int32
	Layers         []int32
	Weights        [][][]float64
	Bias           [][][]float64
}

func (nn *NeutralNetwork) New(layers []int32, learningRate float64) *NeutralNetwork {
	nn.LearningRate = learningRate
	nn.Layers = layers
	nn.Weights = [][][]float64{}
	nn.Bias = [][][]float64{}

	return nn
}

// func (nn *NeutralNetwork) Train

func (nn *NeutralNetwork) initDefaultLayers() {
	for i := 0; i < len(nn.Layers); i++ {

	}
}

func main() {

}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}
