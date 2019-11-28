/*
 * Filename: /Users/hackintosh/work/personal/dl-basic/chap1/chap1.go
 * Path: /Users/hackintosh/work/personal/dl-basic/chap1
 * Created Date: Thursday, November 28th 2019, 4:34:10 pm
 * Author: Nora
 *
 * Copyright (c) 2019 Your Company
 */

package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
)

func main() {
	csvfile, err := os.Open("data_linear.csv")
	if err != nil {
		log.Fatalln("Couldn't open the csv file", err)
	}

	// Parse the file
	r := csv.NewReader(csvfile)
	//r := csv.NewReader(bufio.NewReader(csvfile))

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
		fmt.Printf("Question: %s Answer %s\n", record[0], record[1])
	}

	// y = w1*x + w0
}
