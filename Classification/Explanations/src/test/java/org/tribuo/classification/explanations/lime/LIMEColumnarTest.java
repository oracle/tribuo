/*
 * Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.classification.explanations.lime;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.SparseTrainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.xgboost.XGBoostClassificationTrainer;
import org.tribuo.data.columnar.ColumnarFeature;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.ResponseProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor;
import org.tribuo.data.columnar.processors.field.IdentityProcessor;
import org.tribuo.data.columnar.processors.field.TextFieldProcessor;
import org.tribuo.data.columnar.processors.response.FieldResponseProcessor;
import org.tribuo.data.csv.CSVDataSource;
import org.tribuo.data.text.impl.BasicPipeline;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.rtree.CARTJointRegressionTrainer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.tribuo.util.tokens.Tokenizer;
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;

import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 *
 */
public class LIMEColumnarTest {

    private static final Tokenizer tokenizer = new BreakIteratorTokenizer(Locale.US);

    @BeforeEach
    public void setupTest() {
        Logger logger = Logger.getLogger(org.tribuo.regression.slm.SLMTrainer.class.getName());
        logger.setLevel(Level.WARNING);
    }

    private Pair<RowProcessor<Label>,Dataset<Label>> generateBinarisedDataset() throws URISyntaxException {
        LabelFactory labelFactory = new LabelFactory();

        ResponseProcessor<Label> responseProcessor = new FieldResponseProcessor<>("Response","N",labelFactory);
        Map<String, FieldProcessor> fieldProcessors = new HashMap<>();
        fieldProcessors.put("A",new IdentityProcessor("A"));
        fieldProcessors.put("B",new DoubleFieldProcessor("B"));
        fieldProcessors.put("C",new DoubleFieldProcessor("C"));
        fieldProcessors.put("D",new IdentityProcessor("D"));
        fieldProcessors.put("TextField",new TextFieldProcessor("TextField",new BasicPipeline(tokenizer,2)));

        RowProcessor<Label> rp = new RowProcessor<>(responseProcessor,fieldProcessors);

        CSVDataSource<Label> source = new CSVDataSource<>(LIMEColumnarTest.class.getResource("/org/tribuo/classification/explanations/lime/test-columnar.csv").toURI(),rp,true);

        Dataset<Label> dataset = new MutableDataset<>(source);

        return new Pair<>(rp,dataset);
    }

    private Pair<RowProcessor<Label>,Dataset<Label>> generateCategoricalDataset() throws URISyntaxException {
        LabelFactory labelFactory = new LabelFactory();

        ResponseProcessor<Label> responseProcessor = new FieldResponseProcessor<>("Response","N",labelFactory);
        Map<String, FieldProcessor> fieldProcessors = new HashMap<>();
        fieldProcessors.put("A",new IdentityProcessor("A") {
            @Override
            public GeneratedFeatureType getFeatureType() {
                return GeneratedFeatureType.CATEGORICAL;
            }});
        fieldProcessors.put("B",new DoubleFieldProcessor("B"));
        fieldProcessors.put("C",new DoubleFieldProcessor("C"));
        fieldProcessors.put("D",new IdentityProcessor("D"){
            @Override
            public GeneratedFeatureType getFeatureType() {
                return GeneratedFeatureType.CATEGORICAL;
            }});
        fieldProcessors.put("TextField",new TextFieldProcessor("TextField",new BasicPipeline(tokenizer,2)));

        RowProcessor<Label> rp = new RowProcessor<>(responseProcessor,fieldProcessors);

        CSVDataSource<Label> source = new CSVDataSource<>(LIMEColumnarTest.class.getResource("/org/tribuo/classification/explanations/lime/test-columnar.csv").toURI(),rp,true);

        Dataset<Label> dataset = new MutableDataset<>(source);

        return new Pair<>(rp,dataset);
    }

    @Test
    public void testBinarisedCategorical() throws URISyntaxException {
        Pair<RowProcessor<Label>,Dataset<Label>> pair = generateBinarisedDataset();

        RowProcessor<Label> rp = pair.getA();
        Dataset<Label> dataset = pair.getB();

        XGBoostClassificationTrainer trainer = new XGBoostClassificationTrainer(50);

        Model<Label> model = trainer.train(dataset);

        SparseTrainer<Regressor> sparseTrainer = new CARTJointRegressionTrainer(4,true);

        LIMEColumnar lime = new LIMEColumnar(new SplittableRandom(1),model,sparseTrainer,1000,rp,tokenizer);

        Map<String,String> testExample = new HashMap<>();

        testExample.put("A","Small");
        testExample.put("B","4.0");
        testExample.put("C","4.0");
        testExample.put("D","Red");
        testExample.put("TextField","The full text field has more words in it than other fields.");

        Pair<LIMEExplanation, List<Example<Regressor>>> explanation = lime.explainWithSamples(testExample);

        for (Example<Regressor> e : explanation.getB()) {
            int aCounter = 0;
            int bCounter = 0;
            int cCounter = 0;
            int dCounter = 0;
            int textCounter = 0;

            for (Feature f : e) {
                String featureName = f.getName();
                if (featureName.startsWith("A")) {
                    aCounter++;
                } else if (featureName.startsWith("B")) {
                    bCounter++;
                } else if (featureName.startsWith("C")) {
                    cCounter++;
                } else if (featureName.startsWith("D")) {
                    dCounter++;
                } else if (featureName.startsWith("TextField")) {
                    textCounter++;
                } else {
                    fail("Unknown feature with name " + featureName);
                }
            }

            if (aCounter != 1) {
                fail("Should only sample one A feature");
            }
            if (bCounter != 1) {
                fail("Should only sample one B feature");
            }
            if (cCounter != 1) {
                fail("Should only sample one C feature");
            }
            if (dCounter != 1) {
                fail("Should only sample one D feature");
            }
        }
    }

    @Test
    public void testCategorical() throws URISyntaxException {
        Pair<RowProcessor<Label>,Dataset<Label>> pair = generateCategoricalDataset();

        RowProcessor<Label> rp = pair.getA();
        Dataset<Label> dataset = pair.getB();

        XGBoostClassificationTrainer trainer = new XGBoostClassificationTrainer(50);

        Model<Label> model = trainer.train(dataset);

        SparseTrainer<Regressor> sparseTrainer = new CARTJointRegressionTrainer(4,true);

        LIMEColumnar lime = new LIMEColumnar(new SplittableRandom(1),model,sparseTrainer,5000,rp,tokenizer);

        Map<String,String> testExample = new HashMap<>();

        testExample.put("A","Small");
        testExample.put("B","4.0");
        testExample.put("C","4.0");
        testExample.put("D","Red");
        testExample.put("TextField","The full text field has more words in it than other fields.");

        Pair<LIMEExplanation, List<Example<Regressor>>> explanation = lime.explainWithSamples(testExample);

        List<String> activeFeatures = explanation.getA().getActiveFeatures();
        assertNotNull(activeFeatures);

        int[] aSampleCount = new int[3];
        int[] dSampleCount = new int[3];

        int aPresentCounter = 0;
        int dPresentCounter = 0;

        for (Example<Regressor> e : explanation.getB()) {
            boolean aPresent = false;
            boolean dPresent = false;
            int aCounter = 0;
            int bCounter = 0;
            int cCounter = 0;
            int dCounter = 0;
            int textCounter = 0;

            for (Feature f : e) {
                String featureName = f.getName();
                if (featureName.equals("A"+ ColumnarFeature.JOINER+"Small")) {
                    aSampleCount[0]++;
                    aCounter++;
                    aPresent = true;
                } else if (featureName.equals("A"+ ColumnarFeature.JOINER+"Medium")) {
                    aSampleCount[1]++;
                    aCounter++;
                    aPresent = true;
                } else if (featureName.equals("A"+ ColumnarFeature.JOINER+"Large")) {
                    aSampleCount[2]++;
                    aCounter++;
                    aPresent = true;
                } else if (featureName.startsWith("B")) {
                    bCounter++;
                } else if (featureName.startsWith("C")) {
                    cCounter++;
                } else if (featureName.equals("D"+ ColumnarFeature.JOINER+"Red")) {
                    dSampleCount[0]++;
                    dCounter++;
                    dPresent = true;
                } else if (featureName.equals("D"+ ColumnarFeature.JOINER+"Yellow")) {
                    dSampleCount[1]++;
                    dCounter++;
                    dPresent = true;
                } else if (featureName.equals("D"+ ColumnarFeature.JOINER+"Green")) {
                    dSampleCount[2]++;
                    dCounter++;
                    dPresent = true;
                } else if (featureName.startsWith("TextField")) {
                    textCounter++;
                } else {
                    fail("Unknown feature with name " + featureName);
                }
            }
            if (!aPresent) {
                aPresentCounter++;
            }
            if (!dPresent) {
                dPresentCounter++;
            }

            // Real features should be sampled correctly
            if (bCounter != 1) {
                fail("Should only sample one B feature");
            }
            if (cCounter != 1) {
                fail("Should only sample one C feature");
            }

            // Categorical features may be sampled multiple times as they are not specified as BINARISED_CATEGORICAL
            if ((aCounter > 3) || (aCounter < 0)) {
                fail("Should sample between 0 and 3 A features");
            }
            if ((dCounter > 3) || (dCounter < 0)) {
                fail("Should sample between 0 and 3 D features");
            }
        }

        assertTrue(aSampleCount[0] > 1000);
        assertTrue(aSampleCount[0] < 2500);
        //System.out.println(aSampleCount[0]);
        assertTrue(aSampleCount[1] > 1000);
        assertTrue(aSampleCount[1] < 2500);
        //System.out.println(aSampleCount[1]);
        assertTrue(aSampleCount[2] > 1000);
        assertTrue(aSampleCount[2] < 2500);
        //System.out.println(aSampleCount[2]);
        assertTrue(aPresentCounter > 1000);
        assertTrue(aPresentCounter < 2500);
        //System.out.println(aPresentCounter);

        assertTrue(dSampleCount[0] > 1000);
        assertTrue(dSampleCount[0] < 2500);
        //System.out.println(dSampleCount[0]);
        assertTrue(dSampleCount[1] > 1000);
        assertTrue(dSampleCount[1] < 2500);
        //System.out.println(dSampleCount[1]);
        assertTrue(dSampleCount[2] > 1000);
        assertTrue(dSampleCount[2] < 2500);
        //System.out.println(dSampleCount[2]);
        assertTrue(dPresentCounter > 1000);
        assertTrue(dPresentCounter < 2500);
        //System.out.println(dPresentCounter);
    }

}
