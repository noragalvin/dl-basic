package libs

import (
	"fmt"
	"strconv"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/linear_models"
)

func PredictWithLibs() {
	trainData, _ := base.ParseCSVToInstances("data_linear.csv", true)
	testData, _ := base.ParseCSVToInstances("data_test.csv", true)
	lr := linear_models.NewLinearRegression()
	lr.Fit(trainData)

	predictions, _ := lr.Predict(testData)

	_, rows := predictions.Size()
	// fmt.Println(predictions)

	for i := 0; i < rows; i++ {
		// actualValue, _ := strconv.ParseFloat(base.GetClass(testData, i), 64)
		expectedValue, _ := strconv.ParseFloat(base.GetClass(predictions, i), 64)
		fmt.Println("The house's price for 50m^2 is: ", expectedValue)
	}
}
