package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
)

var nNodes []int
var activations [][]float64
var weights [][][]float64
var changes [][][]float64
var mFactor float64

func main() {
	csvfile, err := os.Open("xor.csv")
	if err != nil {
		log.Fatalln("Couldn't open the csv file", err)
	}

	// Parse the file
	r := csv.NewReader(csvfile)

	var input [][]float64
	var output [][]float64
	// Number of records
	var N int

	// Loop over first line
	r.Read()
	// Iterate through the records
	for {
		// Read each record from csv
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		var a, b, c float64
		a, _ = strconv.ParseFloat(record[0], 64)
		b, _ = strconv.ParseFloat(record[1], 64)
		c, _ = strconv.ParseFloat(record[2], 64)

		xi := []float64{a, b}

		input = append(input, xi)

		output = append(output, []float64{c})
		N++
	}

	log.Println("Input: ", input)
	log.Println("Output: ", output)

	layers := []int{len(input[0])}
	hiddenLayers := []int{8, 8}
	numOfIteration := 10000

	learningRate := 0.2
	mFactor = 0.4
	log.Println("Learning Rate: ", learningRate)
	for i := 0; i < len(hiddenLayers); i++ {
		layers = append(layers, hiddenLayers[i])
	}
	layers = append(layers, 1)

	log.Println("Layers: ", layers)

	for i := 0; i < len(layers)-1; i++ {
		nNodes = append(nNodes, layers[i]+1) // +1 for bias
	}
	nNodes = append(nNodes, layers[len(layers)-1])

	log.Println("Nodes: ", nNodes)

	// totalNodes := 0

	// Init matrix
	weights = make([][][]float64, len(nNodes)-1)
	changes = make([][][]float64, len(nNodes)-1)

	for i := 0; i < len(weights); i++ {
		weights[i] = make([][]float64, nNodes[i])
		for j := 0; j < len(weights[i]); j++ {
			weights[i][j] = make([]float64, nNodes[i+1])
		}
	}

	for i := 0; i < len(changes); i++ {
		changes[i] = make([][]float64, nNodes[i])
		for j := 0; j < len(changes[i]); j++ {
			changes[i][j] = make([]float64, nNodes[i+1])
		}
	}

	// init random weights
	rand.Seed(0)
	for i := 0; i < len(nNodes)-1; i++ {
		for j := 0; j < nNodes[i]; j++ {
			for k := 0; k < nNodes[i+1]; k++ {
				weights[i][j][k] = randFloats(-1, 1)
			}
		}
	}

	fmt.Println(weights)

	for i := 0; i < len(nNodes); i++ {
		activations = append(activations, vector(nNodes[i], 1.0))
	}

	for i := 0; i < numOfIteration; i++ {
		loss := float64(0)
		for j := 0; j < len(input); j++ {
			// Feedforwards
			feedForwards(input[j])

			// Back propagate
			loss += backPropagate(output[j], learningRate)
		}
		// log.Println("LOSS: ", loss)
	}

	for i := 0; i < len(input); i++ {
		log.Printf("PREDICT: %f. ACTUAL: %f\n", predict(input[i]), output[i])
	}
}

func predict(input []float64) []float64 {
	return feedForwards(input)
}

func feedForwards(input []float64) []float64 {
	for k := 1; k < nNodes[0]; k++ {
		activations[0][k] = input[k-1]
	}

	// Calculate activations
	for k := 1; k < len(nNodes)-1; k++ {
		for l := 1; l < nNodes[k]; l++ {
			a := float64(0)
			for m := 0; m < nNodes[k-1]; m++ {
				a = a + weights[k-1][m][l]*activations[k-1][m]
			}
			activations[k][l] = relu(a)
		}
	}

	// Calculate activations output
	for k := 0; k < nNodes[len(nNodes)-1]; k++ {
		a := float64(0)
		for l := 0; l < nNodes[len(nNodes)-2]; l++ {
			a = a + weights[len(nNodes)-2][l][k]*activations[len(nNodes)-2][l]
		}
		activations[len(nNodes)-1][k] = sigmoid(a)

	}

	return activations[len(nNodes)-1]
}

func backPropagate(output []float64, learningRate float64) float64 {
	NLayers := len(nNodes)

	deltas := make([][]float64, NLayers-1)
	deltas[NLayers-2] = vector(nNodes[NLayers-1], 0.0)
	for i := 0; i < nNodes[NLayers-1]; i++ {
		deltas[NLayers-2][i] = sigmoidDerivative(activations[NLayers-1][i]) * (output[i] - activations[NLayers-1][i])
	}

	for k := len(deltas) - 2; k >= 0; k-- {
		deltas[k] = vector(nNodes[k+1], 0.0)
		for i := 0; i < nNodes[k+1]; i++ {
			var e float64

			for j := 0; j < nNodes[k+2]-1; j++ {
				e += deltas[k+1][j] * weights[k+1][i][j]
			}

			deltas[k][i] = sigmoidDerivative(activations[k+1][i]) * e
			// deltas[k][i] = drelu(activations[k+1][i]) * e
		}
	}

	for k := NLayers - 2; k >= 0; k-- {
		for i := 0; i < nNodes[k]; i++ {
			for j := 0; j < nNodes[k+1]; j++ {
				change := deltas[k][j] * activations[k][i]
				weights[k][i][j] = weights[k][i][j] + learningRate*change + mFactor*changes[k][i][j]
				changes[k][i][j] = change
			}
		}
	}
	var err float64
	for i := 0; i < len(output); i++ {
		err += 0.5 * math.Pow(output[i]-activations[NLayers-1][i], 2)
	}
	return err
}
func vector(I int, fill float64) []float64 {
	v := make([]float64, I)
	for i := 0; i < I; i++ {
		v[i] = fill
	}
	return v
}

func randFloats(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

func relu(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func drelu(y float64) float64 {
	if y > 0 {
		return 1
	}
	return 0
}
