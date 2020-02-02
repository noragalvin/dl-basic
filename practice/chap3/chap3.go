package main

import (
	"encoding/csv"
	"io"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"
)

func main() {
	csvfile, err := os.Open("xor.csv")
	if err != nil {
		log.Fatalln("Couldn't open the csv file", err)
	}

	// Parse the file
	r := csv.NewReader(csvfile)

	var input [][]float64
	var output []float64
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

		output = append(output, c)
		N++
	}

	log.Println("Input: ", input)
	log.Println("Output: ", output)

	layers := []int{len(input[0])}
	hiddenLayers := []int{3}
	for i := 0; i < len(hiddenLayers); i++ {
		layers = append(layers, hiddenLayers[i])
	}
	layers = append(layers, 1)

	log.Println("Layers: ", layers)

	nNodes := []int{}
	for i := 0; i < len(layers)-1; i++ {
		nNodes = append(nNodes, layers[i]+1) // +1 for bias
	}
	nNodes = append(nNodes, layers[len(layers)-1])

	log.Println("Nodes: ", nNodes)

	// totalNodes := 0

	// Init matrix
	weights := make([][][]float64, len(nNodes)-1)

	for i := 0; i < len(weights); i++ {
		weights[i] = make([][]float64, nNodes[i])
		for j := 0; j < len(weights[i]); j++ {
			weights[i][j] = make([]float64, nNodes[i+1])
		}
	}

	// init random weights
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < len(nNodes)-1; i++ {
		for j := 0; j < nNodes[i]; j++ {
			for k := 0; k < nNodes[i+1]; k++ {
				// if j == 0 {
				// 	weights[i][j][k] = 1 // init bias = 1
				// 	continue
				// }
				weights[i][j][k] = randFloats(-1, 1)
			}
		}
	}

	log.Println(weights)

	activations := [][]float64{}

	// Feedforwards
	// init activations
	for i := 0; i < len(nNodes); i++ {
		activations = append(activations, vector(nNodes[i], 1.0))
	}

	// for i := 0; i < nNodes[0]-1; i++ {
	// 	activations[0][i] = input[i]
	// }

	log.Println("Activations: ", activations)


	// for i := 1; i < len(nNodes) - 1; i++ {
	// 	// activations[i] = make([]float64, layers[i])
	// 	for j := 0; j < nNodes[i];j ++ {
	// 		sum := 0
	// 		for k := 0; k < nNodes[i-1]; k++ {
	// 			sum = sum + weights[i-1]
	// 		}
	// 		activations[i] = sum
	// 	}
	// }
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
