/*
 * Copyright (c) 2021 Oracle and/or its affiliates. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tribuo.interop.tensorflow;

import org.tensorflow.ndarray.FloatNdArray;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.datasource.IDXDataSource;
import org.tribuo.interop.tensorflow.example.CNNExamples;
import org.tribuo.interop.tensorflow.example.GraphTuple;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class TFTrainerTest {

    private static final int PIXEL_DEPTH = 255;
    private static final int IMAGE_SIZE = 28;
    private static final int NUM_LABELS = 10;
    private static final String INPUT_NAME = "inputplaceholder";

    private static String ndArrToString(FloatNdArray ndarray) {
        StringBuffer sb = new StringBuffer();
        ndarray.scalars().forEachIndexed((idx,array) -> sb.append(Arrays.toString(idx)).append(" = ").append(array.getFloat()).append("\n"));
        return sb.toString();
    }

    public static void main(String[] args) throws IOException {
        LabelFactory labelFactory = new LabelFactory();
        String base = "./tutorials/";

        System.out.println("Loading data");
        IDXDataSource<Label> trainMNIST = new IDXDataSource<>(Paths.get(base,"train-images-idx3-ubyte.gz"),Paths.get(base,"train-labels-idx1-ubyte.gz"),labelFactory);
        IDXDataSource<Label> testMNIST = new IDXDataSource<>(Paths.get(base,"t10k-images-idx3-ubyte.gz"),Paths.get(base,"t10k-labels-idx1-ubyte.gz"),labelFactory);

        Dataset<Label> train = new MutableDataset<>(trainMNIST);
        Dataset<Label> test = new MutableDataset<>(testMNIST);

        System.out.println("Building graph");
        GraphTuple graphTuple = CNNExamples.buildLeNetGraph(INPUT_NAME,IMAGE_SIZE,PIXEL_DEPTH,NUM_LABELS);

        Map<String, Float> gradientParams = new HashMap<>();
        gradientParams.put("learningRate", 0.01f);
        gradientParams.put("initialAccumulatorValue", 0.1f);

        ExampleTransformer<Label> imageTransformer = new ImageTransformer<>(INPUT_NAME, 28, 28, 1);
        OutputTransformer<Label> outputTransformer = new LabelTransformer();

        TFTrainer<Label> trainer = new TFTrainer<>(graphTuple.graph,
                graphTuple.outputName,
                graphTuple.initName,
                GradientOptimiser.ADAGRAD,
                gradientParams,
                imageTransformer,
                outputTransformer,
                16,
                2,
                16);

        System.out.println("Training model");
        Model<Label> model = trainer.train(train);

        System.out.println("Evaluating model");
        LabelEvaluation eval = new LabelEvaluator().evaluate(model,test);

        System.out.println(eval.toString());

        System.out.println(eval.getConfusionMatrix().toString());
    }
}