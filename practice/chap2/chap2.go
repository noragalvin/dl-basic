package main

import (
	"encoding/csv"
	"io"
	"log"
	"math"
	"os"
	"strconv"
)

func main() {
	csvfile, err := os.Open("dataset.csv")
	if err != nil {
		log.Fatalln("Couldn't open the csv file", err)
	}

	// Parse the file
	r := csv.NewReader(csvfile)

	var x [][]float64
	var y []float64
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
		var c float64
		a, _ = strconv.ParseFloat(record[0], 64)
		b, _ = strconv.ParseFloat(record[1], 64)
		c, _ = strconv.ParseFloat(record[2], 64)

		xi := []float64{1, a, b}

		x = append(x, xi)

		y = append(y, c)
		N++
	}

	w := []float64{0, 0.1, 0.1}

	numOfIteration := 1000

	learningRate := 0.01

	cost := make([]float64, numOfIteration)

	for i := 0; i < numOfIteration; i++ {
		yPredict := predict(multiplyMatrix(x, w))

		y1 := multiplyHadamardMatrix(y, ln(yPredict))

		y2 := multiplyHadamardMatrix(decrease(y), ln(decrease(yPredict)))

		// log.Println(y1)
		// log.Println(y2)

		cost[i] = -sum(L(y1, y2))
		// log.Println(cost[i])

		w = minus(w, multiply(learningRate, multiplyMatrix(transposeMatrix(x), minus(yPredict, y))))

	}

	yPredict := predict(multiplyMatrix(x, w))
	log.Println(yPredict)
}

func L(a []float64, b []float64) []float64 {
	result := []float64{}
	for i := 0; i < len(a); i++ {
		result = append(result, a[i]+b[i])
	}
	return result
}

func transposeMatrix(a [][]float64) [][]float64 {
	m := len(a[0])
	n := len(a)

	result := make([][]float64, m)
	for i := 0; i < m; i++ {
		result[i] = make([]float64, n)
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			result[i][j] = a[j][i]
		}
	}
	return result
}

func minus(a []float64, b []float64) []float64 {
	result := []float64{}

	for i := 0; i < len(a); i++ {
		result = append(result, a[i]-b[i])
	}
	return result
}

func multiply(x float64, a []float64) []float64 {
	result := []float64{}
	for i := 0; i < len(a); i++ {
		result = append(result, a[i]*x)
	}
	return result
}

func multiplyHadamardMatrix(a []float64, b []float64) []float64 {
	N := len(a)
	result := []float64{}

	for i := 0; i < N; i++ {
		result = append(result, a[i]*b[i])
	}
	return result
}

func decrease(a []float64) []float64 {
	N := len(a)
	result := []float64{}

	for i := 0; i < N; i++ {
		result = append(result, 1-a[i])
	}
	return result
}

func ln(a []float64) []float64 {
	result := []float64{}
	for i := 0; i < len(a); i++ {
		result = append(result, math.Log(a[i]))
	}
	return result
}

func sum(a []float64) float64 {
	sum := float64(0)
	for i := 0; i < len(a); i++ {
		sum += a[i]
	}
	return sum
}

func predict(a []float64) []float64 {
	result := []float64{}
	for i := 0; i < len(a); i++ {
		result = append(result, sigmoid(a[i]))
	}
	return result
}

// func multiplyMatrix(a [][]float64, b []float64) []float64 {
// 	N := len(a)
// 	result := []float64{}
// 	for i := 0; i < N; i++ {
// 		result = append(result, b[0]+b[1]*a[i][1]+b[2]*a[i][2])
// 	}
// 	return result
// }

func multiplyMatrix(a [][]float64, b []float64) []float64 {
	N := len(a)
	result := []float64{}

	for i := 0; i < N; i++ {
		x := float64(0)
		for j := 0; j < len(a[i]); j++ {
			x += a[i][j] * b[j]
		}
		result = append(result, x)
	}
	return result
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}
