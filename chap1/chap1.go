package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
)

func main() {
	csvfile, err := os.Open("data_linear.csv")
	if err != nil {
		log.Fatalln("Couldn't open the csv file", err)
	}

	// Parse the file
	r := csv.NewReader(csvfile)

	var area []float64
	var prices []float64
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
		var a, b float64
		a, _ = strconv.ParseFloat(record[0], 64)
		b, _ = strconv.ParseFloat(record[1], 64)

		area = append(area, a)
		prices = append(prices, b)
		N++
	}

	x := make([][][]float64, 1)
	y := make([][]float64, 1)

	x[0] = make([][]float64, 0)
	y[0] = make([]float64, 0)
	for i := 0; i < N; i++ {
		x[0] = append(x[0], []float64{1, area[i]})
		y[0] = append(y[0], prices[i])
	}

	w := make([][]float64, 1)
	w[0] = make([]float64, 2)
	w[0][0] = 0
	w[0][1] = 1
	numOfIteration := 100

	cost := make([][]float64, 1)
	cost[0] = make([]float64, numOfIteration)

	learningRate := 0.000001

	sumPrices := float64(0)
	for i := range x[0] {
		sumPrices += x[0][i][1]
	}
	for i := 0; i < numOfIteration; i++ {
		r := derivativeW0(predictMatrix(x[0], w), y)
		cost[0][i] = 0.5 * sumMatrix(multiplyHadamardMatrix(r, r))
		w[0][0] -= learningRate * sumMatrix(r)
		w[0][1] -= learningRate * sumMatrix(multiplyHadamardMatrix(r, reShapeMatrix(x[0], 1)))
		// if i == 0 {
		// 	return
		// }
	}
	// predict := predictMatrix(x[0], w)
	// fmt.Println(predict)

	x1 := float64(50)
	y1 := w[0][0] + w[0][1]*x1
	fmt.Println("The prices of area 50m^2 is: ", y1)
}

func predictMatrix(a [][]float64, b [][]float64) [][]float64 {
	N := len(a)

	predict := make([][]float64, 1)
	predict[0] = make([]float64, 0)

	for i := 0; i < N; i++ {
		predict[0] = append(predict[0], float64(b[0][0]+b[0][1]*a[i][1]))

	}
	return predict
}

func multiplyHadamardMatrix(a [][]float64, b [][]float64) [][]float64 {
	N := len(a[0])
	result := make([][]float64, 1)
	result[0] = make([]float64, 0)
	// fmt.Println(a)
	// fmt.Println(b)

	for i := 0; i < N; i++ {
		result[0] = append(result[0], a[0][i]*b[0][i])
	}
	return result
}

func derivativeW0(predict [][]float64, y [][]float64) [][]float64 {
	N := len(predict[0])

	result := make([][]float64, 1)
	result[0] = make([]float64, 0)

	for i := 0; i < N; i++ {
		result[0] = append(result[0], float64(predict[0][i]-y[0][i]))
	}
	return result
}

func sumMatrix(matrix [][]float64) float64 {
	// fmt.Println(matrix)
	N := len(matrix[0])
	sum := float64(0)
	for i := 0; i < N; i++ {
		sum += matrix[0][i]
	}
	return sum
}

func reShapeMatrix(a [][]float64, col int) [][]float64 {
	result := make([][]float64, 1)
	result[0] = make([]float64, 0)
	N := len(a)
	for i := 0; i < N; i++ {
		result[0] = append(result[0], a[i][col])
	}
	return result
}
