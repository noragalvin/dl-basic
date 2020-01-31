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

		xi := []float64{1, a, b}

		input = append(input, xi)

		output = append(output, c)
		N++
	}

	// log.Println(input)
	// log.Println(output)

	layers := []int{len(input[0]) - 1}
	hiddenLayers := []int{1, 3}
	for i := 0; i < len(hiddenLayers); i++ {
		layers = append(layers, hiddenLayers[i])
	}
	layers = append(layers, 1)

	// log.Println(layers)

	// totalNodes := 0

	weights := make([][][]float64, len(layers)-1)

	for i := 0; i < len(weights); i++ {
		weights[i] = make([][]float64, layers[i])
		for j := 0; j < len(weights[i]); j++ {
			weights[i][j] = make([]float64, layers[i+1])
		}
		// log.Println(weights)
	}
	// log.Println(weights)

	// init random weights
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < len(layers)-1; i++ {
		for j := 0; j < layers[i]; j++ {
			for k := 0; k < layers[i+1]; k++ {
				weights[i][j][k] = randFloats(-1, 1)
			}
		}
	}

	log.Println(weights)
}

func randFloats(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}
