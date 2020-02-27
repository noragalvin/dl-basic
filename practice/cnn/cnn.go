package main

import (
	"fmt"
	"log"
)

func main() {
	X := [][]float64{}
	X = append(X, []float64{float64(1), float64(1), float64(1), float64(0), float64(0)})
	X = append(X, []float64{float64(0), float64(1), float64(1), float64(1), float64(0)})
	X = append(X, []float64{float64(0), float64(0), float64(1), float64(1), float64(1)})
	X = append(X, []float64{float64(0), float64(0), float64(1), float64(1), float64(0)})
	X = append(X, []float64{float64(0), float64(1), float64(1), float64(0), float64(0)})

	W := [][]float64{}
	W = append(W, []float64{float64(1), float64(0), float64(1)})
	W = append(W, []float64{float64(0), float64(1), float64(0)})
	W = append(W, []float64{float64(1), float64(0), float64(1)})

	Y := convolution(X, W, 0, 0)

	fmt.Println(Y)

}

/*
	Input:
		X: input matrix
		W: kernel matrix
		p: padding
		s: strike

	Ouput:
		Two dimensional matrix is element-wise of matrix X and W with p and s
*/
func convolution(X, W [][]float64, p int, s int) [][]float64 {

	centerW := findCenterMatrix(W)
	fmt.Println("Center W: ", centerW)

	Y := [][]float64{}

	widthX := len(X[0])
	heightX := len(X)

	sizeW := len(W)

	/*
		Matrix Y = (m - k + 1) * (n - k + 1)
			m: width X
			n: height X
			k: size W
	*/
	widthY := widthX - sizeW + 1
	heightY := heightX - sizeW + 1

	fmt.Println("Width Y, height Y: ", widthY, heightY)

	for i := 0; i < len(X); i++ {

	}

	return Y
}

func findCenterMatrix(W [][]float64) int {
	width := len(W[0])
	height := len(W)

	if width != height {
		log.Fatal("Wrong matrix W")
	}
	center := width/2 + 1
	return center
}
