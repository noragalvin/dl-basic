package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"
)

var nNodes []int
var activations [][]float64
var weights [][][]float64
var changes [][][]float64
var mFactor float64
var numOfIteration int
var learningRate float64
var timeExcute float64

type NeutralNetwork struct {
	NNodes         []int         `json:"nNodes"`
	Activations    [][]float64   `json:"activations"`
	Weights        [][][]float64 `json:"weights"`
	Changes        [][][]float64 `json:"changes"`
	MFactor        float64       `json:"mfactor"`
	NumOfIteration int           `json:"numOfIteration"`
	LearningRate   float64       `json:"learningRate"`
	TimeExcute     float64       `json:"timeExcute"`
}

func main() {

	args := os.Args
	if len(args) > 1 {

		if args[1] == "predict" {
			nn, err := load("model-new.json")

			nNodes = nn.NNodes
			activations = nn.Activations
			weights = nn.Weights

			if err != nil {
				log.Println(err)
				return
			}

			if len(args) > 2 {
				if args[2] == "testing" {
					predictTesting()
					return
				}
				fileName := args[2]
				filePath := "testing/" + fileName
				input := dataFromFile(filePath)

				result := predict(input)
				predictValue := findMax(result)
				fmt.Printf("\n====================\n")
				fmt.Println("FILE: ", filePath)
				fmt.Printf("PREDICT: %d.\n", predictValue)
				fmt.Printf("====================\n")
				return
			}

			predictTraining()

			return
		}
	}

	input, output := traningData()

	log.Println("Input: ", len(input))
	log.Println("Output: ", len(output))

	layers := []int{len(input[0])}
	hiddenLayers := []int{50}
	numOfIteration = 10000
	log.Println("Num of iteration: ", numOfIteration)

	learningRate = 0.03
	mFactor = 0.9
	log.Println("Learning Rate: ", learningRate)
	log.Println("Momentum: ", mFactor)

	initNetwork(hiddenLayers, layers, output)

	start := time.Now()

	train(numOfIteration, input, output, learningRate)

	elapsed := time.Since(start)
	timeExcute = elapsed.Seconds()
	log.Printf("Training took %s", elapsed)

	save("model-new.json")

}

func initNetwork(hiddenLayers []int, layers []int, output [][]float64) {
	for i := 0; i < len(hiddenLayers); i++ {
		layers = append(layers, hiddenLayers[i])
	}
	layers = append(layers, len(output[0]))

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

	for i := 0; i < len(nNodes); i++ {
		activations = append(activations, vector(nNodes[i], 1.0))
	}
}

func train(numOfIteration int, input [][]float64, output [][]float64, learningRate float64) {
	for i := 0; i < numOfIteration; i++ {
		loss := float64(0)
		for j := 0; j < len(input); j++ {
			// Feedforwards
			feedForwards(input[j])

			// Back propagate
			loss += backPropagate(output[j], learningRate)
		}
		if i%1000 == 0 {
			log.Printf("Epoch: %d  LOSS: %f", i+1, loss)
			earlyStoppingTraining()
			earlyStoppingTesting()
		}
		// if i%10 == 0 {
		// }
	}
}

func traningData() ([][]float64, [][]float64) {
	inputData := [][]float64{}
	outputData := [][]float64{}

	for i := 0; i < 10; i++ {
		var files []string

		root := fmt.Sprintf("training/%d", i)
		err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
			files = append(files, path)
			return nil
		})
		if err != nil {
			panic(err)
		}
		files = files[1:]

		// Get output data
		output := []float64{}
		for j := 0; j < 10; j++ {
			if j == i {
				output = append(output, 1)
			} else {
				output = append(output, 0)
			}
		}

		// Get input data
		for _, file := range files {
			input := []float64{}

			f0, err := os.Open(fmt.Sprintf("%s", file))
			if err != nil {
				fmt.Println(err)
			}
			defer f0.Close()
			img0, _, err := image.Decode(f0)
			if err != nil {
				fmt.Println(err)
			}

			for u := 0; u < 28; u++ {
				for v := 0; v < 28; v++ {
					r, _, _, _ := img0.At(u, v).RGBA()

					input = append(input, float64(r/257/255))
				}
			}

			inputData = append(inputData, input)
			outputData = append(outputData, output)
		}
	}

	return inputData, outputData
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
		deltas[NLayers-2][i] = sigmoidDerivative(activations[NLayers-1][i]) * (activations[NLayers-1][i] - output[i])
	}

	for k := len(deltas) - 2; k >= 0; k-- {
		deltas[k] = vector(nNodes[k+1], 0.0)
		for i := 0; i < nNodes[k+1]; i++ {
			var e float64

			for j := 0; j < nNodes[k+2]-1; j++ {
				e += deltas[k+1][j] * weights[k+1][i][j]
			}

			// deltas[k][i] = sigmoidDerivative(activations[k+1][i]) * e
			deltas[k][i] = reluDerivative(activations[k+1][i]) * e
		}
	}

	// weightsDecay := float64(0)
	// numOfWeights := float64(0)

	// for k := NLayers - 2; k >= 0; k-- {
	// 	for i := 0; i < nNodes[k]; i++ {
	// 		for j := 0; j < nNodes[k+1]; j++ {
	// 			numOfWeights++
	// 			// weightsDecay = weightsDecay + math.Pow(weights[k][i][j], 2)
	// 		}
	// 	}
	// }

	// weightsDecay = math.Sqrt(weightsDecay)

	// log.Println(weightsDecay)
	// log.Println((0.1 / numOfWeights) * weightsDecay)

	for k := NLayers - 2; k >= 0; k-- {
		for i := 0; i < nNodes[k]; i++ {
			for j := 0; j < nNodes[k+1]; j++ {
				change := deltas[k][j]*activations[k][i] + 0.01*weights[k][i][j]
				if j == nNodes[k+1]-1 {
					change = deltas[k][j] * activations[k][i]
				}
				m := (1-mFactor)*change + mFactor*changes[k][i][j]

				weights[k][i][j] = weights[k][i][j] - learningRate*m
				// weights[k][i][j] = weights[k][i][j] - learningRate*change
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

func reluDerivative(y float64) float64 {
	if y > 0 {
		return 1
	}
	return 0
}

func marshal(v interface{}) (io.Reader, error) {
	b, err := json.MarshalIndent(v, "", "\t")
	if err != nil {
		return nil, err
	}
	return bytes.NewReader(b), nil
}

func unmarshal(r io.Reader, v interface{}) error {
	return json.NewDecoder(r).Decode(v)
}

// save neural network to file
func save(path string) error {
	data := make(map[string]interface{})

	data["nNodes"] = nNodes
	data["activations"] = activations
	data["weights"] = weights
	data["mFactor"] = mFactor
	data["numberOfIteration"] = numOfIteration
	data["learningRate"] = learningRate
	data["timeExcute"] = timeExcute

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	r, err := marshal(data)
	if err != nil {
		return err
	}
	_, err = io.Copy(f, r)
	return err
}

// load neural network from file
func load(path string) (NeutralNetwork, error) {
	nn := NeutralNetwork{}

	f, err := os.Open(path)
	if err != nil {
		return nn, err
	}
	defer f.Close()
	err = unmarshal(f, &nn)
	return nn, err
}

func findMax(arr []float64) int {
	max := arr[0]
	index := 0
	for i := 0; i < len(arr); i++ {
		if arr[i] > max {
			max = arr[i]
			index = i
		}
	}
	return index
}

func calcPerCent(result []int) float64 {
	n := len(result)

	n1 := 0

	for _, number := range result {
		if number == 1 {
			n1++
		}
	}

	return float64(n1) / float64(n) * 100
}

func predictTraining() {

	output := []int{}

	for i := 0; i < 10; i++ {
		var files []string

		root := fmt.Sprintf("training/%d", i)
		err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
			files = append(files, path)
			return nil
		})
		files = files[1:]
		if err != nil {
			panic(err)
		}

		for _, file := range files {
			input := []float64{}

			f0, err := os.Open(fmt.Sprintf("%s", file))
			if err != nil {
				fmt.Println(err)
			}
			defer f0.Close()
			img0, _, err := image.Decode(f0)
			if err != nil {
				fmt.Println(err)
			}

			for u := 0; u < 28; u++ {
				for v := 0; v < 28; v++ {
					r, _, _, _ := img0.At(u, v).RGBA()

					input = append(input, float64(r/257/255))
				}
			}

			result := predict(input)

			predictValue := findMax(result)
			fmt.Printf("\n====================\n")
			fmt.Println("FILE: ", file)
			fmt.Printf("PREDICT: %d. ACTUALLY: %d\n", predictValue, i)
			fmt.Printf("====================\n")
			if predictValue == i {
				output = append(output, 1)
			} else {
				output = append(output, 0)
			}
		}
	}

	fmt.Printf("\n\nResult: %.2f%%\n", calcPerCent(output))
}

func predictTesting() {

	output := []int{}

	for i := 0; i < 10; i++ {
		var files []string

		root := fmt.Sprintf("testingwithlabel/%d", i)
		err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
			files = append(files, path)
			return nil
		})
		files = files[1:]
		if err != nil {
			panic(err)
		}

		for _, file := range files {
			input := []float64{}

			f0, err := os.Open(fmt.Sprintf("%s", file))
			if err != nil {
				fmt.Println(err)
			}
			defer f0.Close()
			img0, _, err := image.Decode(f0)
			if err != nil {
				fmt.Println(err)
			}

			for u := 0; u < 28; u++ {
				for v := 0; v < 28; v++ {
					r, _, _, _ := img0.At(u, v).RGBA()

					input = append(input, float64(r/257/255))
				}
			}

			result := predict(input)

			predictValue := findMax(result)
			fmt.Printf("\n====================\n")
			fmt.Println("FILE: ", file)
			fmt.Printf("PREDICT: %d. ACTUALLY: %d\n", predictValue, i)
			fmt.Printf("====================\n")
			if predictValue == i {
				output = append(output, 1)
			} else {
				output = append(output, 0)
			}
		}
	}

	fmt.Printf("\n\nResult: %.2f%%\n", calcPerCent(output))
}

func earlyStoppingTraining() {

	output := []int{}

	for i := 0; i < 10; i++ {
		var files []string

		root := fmt.Sprintf("training/%d", i)
		err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
			files = append(files, path)
			return nil
		})
		files = files[1:]
		if err != nil {
			panic(err)
		}

		for _, file := range files {
			input := []float64{}

			f0, err := os.Open(fmt.Sprintf("%s", file))
			if err != nil {
				fmt.Println(err)
			}
			defer f0.Close()
			img0, _, err := image.Decode(f0)
			if err != nil {
				fmt.Println(err)
			}

			for u := 0; u < 28; u++ {
				for v := 0; v < 28; v++ {
					r, _, _, _ := img0.At(u, v).RGBA()

					input = append(input, float64(r/257/255))
				}
			}

			result := predict(input)

			predictValue := findMax(result)
			// fmt.Printf("\n====================\n")
			// fmt.Println("FILE: ", file)
			// fmt.Printf("PREDICT: %d. ACTUALLY: %d\n", predictValue, i)
			// fmt.Printf("====================\n")
			if predictValue == i {
				output = append(output, 1)
			} else {
				output = append(output, 0)
			}
		}
	}

	fmt.Printf("Result Training: %.2f%%\n", calcPerCent(output))
}

func earlyStoppingTesting() {

	output := []int{}

	for i := 0; i < 10; i++ {
		var files []string

		root := fmt.Sprintf("testingwithlabel/%d", i)
		err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
			files = append(files, path)
			return nil
		})
		files = files[1:]
		if err != nil {
			panic(err)
		}

		for _, file := range files {
			input := []float64{}

			f0, err := os.Open(fmt.Sprintf("%s", file))
			if err != nil {
				fmt.Println(err)
			}
			defer f0.Close()
			img0, _, err := image.Decode(f0)
			if err != nil {
				fmt.Println(err)
			}

			for u := 0; u < 28; u++ {
				for v := 0; v < 28; v++ {
					r, _, _, _ := img0.At(u, v).RGBA()

					input = append(input, float64(r/257/255))
				}
			}

			result := predict(input)

			predictValue := findMax(result)
			// fmt.Printf("\n====================\n")
			// fmt.Println("FILE: ", file)
			// fmt.Printf("PREDICT: %d. ACTUALLY: %d\n", predictValue, i)
			// fmt.Printf("====================\n")
			if predictValue == i {
				output = append(output, 1)
			} else {
				output = append(output, 0)
			}
		}
	}

	fmt.Printf("Result Testing: %.2f%%\n", calcPerCent(output))
}

func dataFromFile(fileName string) []float64 {
	// Get input data
	input := []float64{}

	f0, err := os.Open(fileName)
	if err != nil {
		fmt.Println(err)
	}
	defer f0.Close()
	img0, _, err := image.Decode(f0)
	if err != nil {
		fmt.Println(err)
	}

	for u := 0; u < 28; u++ {
		for v := 0; v < 28; v++ {
			r, _, _, _ := img0.At(u, v).RGBA()

			input = append(input, float64(r/257/255))
		}
	}

	return input
}
