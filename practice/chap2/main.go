package main

import (
	"math"
)

func main() {
	
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}
