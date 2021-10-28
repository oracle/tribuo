/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.reproducibility;

import com.oracle.labs.mlrg.olcut.config.property.SimpleProperty;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.FileProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;
import org.junit.jupiter.api.io.TempDir;
import org.tribuo.Dataset;
import org.tribuo.DataSource;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.linear.LinearSGDModel;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.common.sgd.AbstractLinearSGDTrainer;
import org.tribuo.common.sgd.AbstractSGDTrainer;
import org.tribuo.common.tree.RandomForestTrainer;
import org.tribuo.data.columnar.FieldExtractor;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.columnar.extractors.DateExtractor;
import org.tribuo.data.columnar.extractors.FloatExtractor;
import org.tribuo.data.columnar.extractors.IntExtractor;
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor;
import org.tribuo.data.columnar.processors.field.IdentityProcessor;
import org.tribuo.data.columnar.processors.field.TextFieldProcessor;
import org.tribuo.data.columnar.processors.response.FieldResponseProcessor;
import org.tribuo.data.csv.CSVDataSource;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.data.csv.CSVSaver;
import org.tribuo.data.text.impl.BasicPipeline;
import org.tribuo.ensemble.BaggingTrainer;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.ensemble.AveragingCombiner;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.tribuo.regression.rtree.CARTRegressionTrainer;
import org.tribuo.regression.rtree.impurity.MeanSquaredError;
import org.tribuo.regression.sgd.linear.LinearSGDTrainer;
import org.tribuo.transform.TransformTrainer;
import org.tribuo.transform.TransformationMap;
import org.tribuo.transform.transformations.LinearScalingTransformation;
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.tribuo.common.tree.AbstractCARTTrainer.MIN_EXAMPLES;

public class ReproUtilTest {
    @TempDir
    static Path tempDir;

    static Path tempFile;

    private static final Class<?>[] silencedClasses = new Class<?>[]{BaggingTrainer.class, AbstractSGDTrainer.class, AbstractLinearSGDTrainer.class, LinearSGDTrainer.class};

    @BeforeAll
    public static void setup() throws IOException {
        for (Class<?> c : silencedClasses) {
            Logger logger = Logger.getLogger(c.getName());
            logger.setLevel(Level.WARNING);
        }

        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();

        tempFile = Files.createFile(tempDir.resolve("dense.csv"));

        CSVSaver saver = new CSVSaver();
        saver.save(tempFile, p.getA(), "response");
    }

    private static CSVDataSource<Label> getCSVDataSource() throws URISyntaxException {
        URL u = ReproUtilTest.class.getResource("/org/tribuo/reproducibility/test.csv");
        Path csvPath = Paths.get(u.toURI());
        //List<String> csvLines = Files.readAllLines(csvPath, StandardCharsets.UTF_8);

        BasicPipeline textPipeline = new BasicPipeline(new BreakIteratorTokenizer(Locale.US),2);
        HashMap<String, FieldProcessor> fieldProcessors = new HashMap<>();
        fieldProcessors.put("height",new DoubleFieldProcessor("height"));
        fieldProcessors.put("description",new TextFieldProcessor("description",textPipeline));
        fieldProcessors.put("transport",new IdentityProcessor("transport"));

        Map<String,FieldProcessor> regexMappingProcessors = Collections.singletonMap("extra.*", new DoubleFieldProcessor("extra.*"));

        FieldResponseProcessor<Label> responseProcessor = new FieldResponseProcessor<>("disposition","UNK",new LabelFactory());

        ArrayList<FieldExtractor<?>> metadataExtractors = new ArrayList<>();
        metadataExtractors.add(new IntExtractor("id"));
        metadataExtractors.add(new DateExtractor("timestamp","timestamp","dd/MM/yyyy HH:mm"));

        FloatExtractor weightExtractor = new FloatExtractor("example-weight");

        RowProcessor<Label> rowProcessor = new RowProcessor<>(metadataExtractors,weightExtractor,responseProcessor,fieldProcessors,regexMappingProcessors, Collections.emptySet());

        return new CSVDataSource<>(csvPath,rowProcessor,true);
    }

    // Generates a CSV file, writes it to disk, and loads it back in using a CSVLoader for the purposes of
    // testing the reproducibility of CSVLoader objects.
    // Data originally generated by LabelledDataGenerator.denseTrainTest()
    private static DataSource<Label> getCSVLoaderSource() throws IOException {
        String tempData = """
                "Foo","1.0","0.5","1.0","-1.0"
                "Foo","1.5","0.35","1.3","-1.2"
                "Foo","1.2","0.45","1.5","-1.0"
                "Bar","-1.1","0.55","-1.5","0.5"
                "Bar","-1.5","0.25","-1.0","0.125"
                "Bar","-1.0","0.5","-1.123","0.123"
                "Baz","1.5","5.0","0.5","4.5"
                "Baz","1.234","5.1235","0.1235","6.0"
                "Baz","1.734","4.5","0.5123","5.5"
                "Quux","-1.0","0.25","5.0","10.0"
                "Quux","-1.4","0.55","5.65","12.0"
                "Quux","-1.9","0.25","5.9","15.0"
                """;

        Path tempFile = null;

        tempFile = Files.createFile(tempDir.resolve("testLoader.csv"));

        Files.write(tempFile, tempData.getBytes(StandardCharsets.UTF_8));

        LabelFactory labelFactory = new LabelFactory();
        CSVLoader<Label> csvLoader = new CSVLoader<>(labelFactory);

        String[] tempHeaders = new String[]{"response", "A", "B", "C", "D"};

        return csvLoader.loadDataSource(tempFile,"response", tempHeaders);
    }

    private static DataSource<Regressor> getConfigurableRegressionDenseTrain() throws IOException {

        RegressionFactory regressionFactory = new RegressionFactory();
        CSVLoader<Regressor> csvLoader = new CSVLoader<>(regressionFactory);

        return csvLoader.loadDataSource(tempFile,"response");
    }

    @Test
    public void testReproduceFromProvenanceWithSplitter() throws URISyntaxException, ClassNotFoundException {
        CSVDataSource<Label> csvSource = getCSVDataSource();

        TrainTestSplitter<Label> trainTestSplitter = new TrainTestSplitter<>(csvSource,0.7,1L);
        MutableDataset<Label> trainingDataset = new MutableDataset<>(trainTestSplitter.getTrain());

        LogisticRegressionTrainer trainer = new LogisticRegressionTrainer();
        LinearSGDModel model = (LinearSGDModel) trainer.train(trainingDataset);
        model = (LinearSGDModel) trainer.train(trainingDataset);
        model = (LinearSGDModel) trainer.train(trainingDataset);

        ReproUtil<Label> reproUtil = new ReproUtil<>(model.getProvenance(),Label.class);

        LinearSGDModel newModel = (LinearSGDModel) reproUtil.reproduceFromProvenance();

        assertEquals(newModel.getWeightsCopy(), model.getWeightsCopy());
    }

    @Test
    public void testReproduceFromProvenanceNoSplitter() throws URISyntaxException, ClassNotFoundException {
        CSVDataSource<Label> csvSource = getCSVDataSource();
        MutableDataset<Label> datasetFromCSV = new MutableDataset<>(csvSource);

        LogisticRegressionTrainer trainer = new LogisticRegressionTrainer();
        LinearSGDModel model = (LinearSGDModel) trainer.train(datasetFromCSV);
        model = (LinearSGDModel) trainer.train(datasetFromCSV);
        model = (LinearSGDModel) trainer.train(datasetFromCSV);

        ReproUtil<Label> reproUtil = new ReproUtil<>(model.getProvenance(),Label.class);
        LinearSGDModel newModel = (LinearSGDModel) reproUtil.reproduceFromProvenance();

        assertEquals(newModel.getWeightsCopy(), model.getWeightsCopy());
    }

    @Test
    public void testReproduceFromModel() throws IOException, URISyntaxException, ClassNotFoundException {
        CSVDataSource<Label> csvSource = getCSVDataSource();
        MutableDataset<Label> datasetFromCSV = new MutableDataset<>(csvSource);

        LogisticRegressionTrainer trainer = new LogisticRegressionTrainer();
        LinearSGDModel model = (LinearSGDModel) trainer.train(datasetFromCSV);
        model = (LinearSGDModel) trainer.train(datasetFromCSV);
        model = (LinearSGDModel) trainer.train(datasetFromCSV);

        ReproUtil<Label> reproUtil = new ReproUtil<>(model);
        ReproUtil.ModelReproduction<Label> modelReproduction = reproUtil.reproduceFromModel();
        LinearSGDModel newModel = (LinearSGDModel) modelReproduction.model();

        assertEquals(0, modelReproduction.featureDiff().reproducedFeatures().size());
        assertEquals(0, modelReproduction.featureDiff().originalFeatures().size());

        assertEquals(0, modelReproduction.outputDiff().reproducedOutput().size());
        assertEquals(0, modelReproduction.outputDiff().originalOutput().size());

        assertEquals(newModel.getWeightsCopy(), model.getWeightsCopy());
    }

    @Test
    public void testOverrideConfigurableProperty() throws URISyntaxException, ClassNotFoundException {
        CSVDataSource<Label> csvSource = getCSVDataSource();
        MutableDataset<Label> datasetFromCSV = new MutableDataset<>(csvSource);

        LogisticRegressionTrainer trainer = new LogisticRegressionTrainer();
        LinearSGDModel model = (LinearSGDModel) trainer.train(datasetFromCSV);
        model = (LinearSGDModel) trainer.train(datasetFromCSV);
        model = (LinearSGDModel) trainer.train(datasetFromCSV);

        URL u = ReproUtilTest.class.getResource("/org/tribuo/reproducibility/test/new_data.csv");
        Path csvPath = Paths.get(u.toURI());

        ReproUtil<Label> reproUtil = new ReproUtil<>(model.getProvenance(),Label.class);

        reproUtil.getConfigurationManager().overrideConfigurableProperty("csvdatasource-1", "dataPath", new SimpleProperty(csvPath.toString()));
        LinearSGDModel newModel = (LinearSGDModel) reproUtil.reproduceFromProvenance();

        assertEquals(newModel.getWeightsCopy(), model.getWeightsCopy());

        for (Pair<String, Provenance> provPair : newModel.getProvenance().getDatasetProvenance().getSourceProvenance()) {
            if (provPair.getA().equals("dataPath")) {
                assertEquals("new_data.csv", ((FileProvenance) provPair.getB()).getValue().getName());
            }
        }
    }


    @Test
    public void testProvDiff() throws IOException, URISyntaxException, ClassNotFoundException {
        //TODO: Expand this to actually assert something
        CSVDataSource<Label> csvSource = getCSVDataSource();
        MutableDataset<Label> datasetFromCSV = new MutableDataset<>(csvSource);

        LogisticRegressionTrainer trainer = new LogisticRegressionTrainer();
        LinearSGDModel model1 = (LinearSGDModel) trainer.train(datasetFromCSV);
        LinearSGDModel model2 = (LinearSGDModel) trainer.train(datasetFromCSV);

        ReproUtil<Label> repro = new ReproUtil<>(model1);

        LinearSGDModel model3 = (LinearSGDModel) repro.reproduceFromProvenance();
        String report = ReproUtil.diffProvenance(model1.getProvenance(), model3.getProvenance());
        // TODO: Evaluate report value, this requires address fact that timestamps will change so can't just
        //  encode the expected report as String
    }

    @Test
    public void reproduceTransformTrainer() throws URISyntaxException, ClassNotFoundException {
        CSVDataSource<Label> csvSource = getCSVDataSource();
        TrainTestSplitter<Label> splitter = new TrainTestSplitter<>(csvSource);
        MutableDataset<Label> datasetFromCSV = new MutableDataset<>(splitter.getTrain());
        MutableDataset<Label> testData = new MutableDataset<>(splitter.getTest());

        LogisticRegressionTrainer trainer = new LogisticRegressionTrainer();
        TransformationMap transformations = new TransformationMap(List.of(new LinearScalingTransformation(0,1)));
        TransformTrainer<Label> transformed = new TransformTrainer<>(trainer, transformations);
        Model<Label> transformedModel = transformed.train(datasetFromCSV);

        ReproUtil<Label> reproUtil = new ReproUtil<>(transformedModel.getProvenance(),Label.class);
        Model<Label> newModel = reproUtil.reproduceFromProvenance();

        LabelEvaluator evaluator = new LabelEvaluator();

        LabelEvaluation oldEvaluation = evaluator.evaluate(transformedModel, testData);
        LabelEvaluation newEvaluation = evaluator.evaluate(newModel, testData);
        assertEquals(oldEvaluation.toString(), newEvaluation.toString());
    }

    @Test
    public void testProvDiffWithTransformTrainer() throws IOException, URISyntaxException {
        //TODO: Expand this to actually assert something
        CSVDataSource<Label> csvSource = getCSVDataSource();
        MutableDataset<Label> datasetFromCSV = new MutableDataset<>(csvSource);

        LogisticRegressionTrainer trainer = new LogisticRegressionTrainer();
        LinearSGDModel model = (LinearSGDModel) trainer.train(datasetFromCSV);

        TransformationMap transformations = new TransformationMap(List.of(new LinearScalingTransformation(0,1)));
        TransformTrainer<Label> transformed = new TransformTrainer<>(trainer, transformations);
        Model<Label> transformedModel = transformed.train(datasetFromCSV);

        String report = ReproUtil.diffProvenance(model.getProvenance(), transformedModel.getProvenance());
        // TODO: Evaluate report value, this requires address fact that timestamps will change so can't just
        //  encode the expected report as String
        //System.out.println(report);
    }


    @Test
    public void testBaggingTrainer() throws IOException, ClassNotFoundException {
        //TODO: Will need more extensive testing here
        CARTRegressionTrainer subsamplingTree = new CARTRegressionTrainer(Integer.MAX_VALUE,
                MIN_EXAMPLES, 0.0f, 0.5f, false, new MeanSquaredError(), Trainer.DEFAULT_SEED);
        RandomForestTrainer<Regressor> rfT = new RandomForestTrainer<>(subsamplingTree,new AveragingCombiner(),10);
        Dataset<Regressor> trainData = new MutableDataset<>(getConfigurableRegressionDenseTrain());
        Model<Regressor> model = rfT.train(trainData);

        ReproUtil<Regressor> reproUtil = new ReproUtil<>(model.getProvenance(),Regressor.class);

        Model<Regressor> reproducedModel = reproUtil.reproduceFromProvenance();

        assertEquals(model.getProvenance().getTrainerProvenance().getInstanceValues().get("train-invocation-count"),
                reproducedModel.getProvenance().getTrainerProvenance().getInstanceValues().get("train-invocation-count"));

    }

    @Test
    public void testBaggingTrainerAllInvocationsChange() throws IOException, ClassNotFoundException {
        // This example has multiple trainers in the form of an ensemble, and all need to be set to the correct value
        CARTRegressionTrainer subsamplingTree = new CARTRegressionTrainer(Integer.MAX_VALUE,
                MIN_EXAMPLES, 0.0f, 0.5f, false, new MeanSquaredError(), Trainer.DEFAULT_SEED);
        RandomForestTrainer<Regressor> rfT = new RandomForestTrainer<>(subsamplingTree,new AveragingCombiner(),10);
        Dataset<Regressor> trainData = new MutableDataset<>(getConfigurableRegressionDenseTrain());
        Model<Regressor> model1 = rfT.train(trainData);
        subsamplingTree.setInvocationCount(15);
        Model<Regressor> model2 = rfT.train(trainData);

        ReproUtil<Regressor> reproUtil = new ReproUtil<>(model2.getProvenance(),Regressor.class);

        Model<Regressor> reproducedModel = reproUtil.reproduceFromProvenance();

        // Make sure the inner trainer's setinvocation count has occurred
        assertEquals(((TrainerProvenanceImpl) model2.getProvenance()
                        .getTrainerProvenance()
                        .getConfiguredParameters()
                        .get("innerTrainer"))
                        .getInstanceValues()
                        .get("train-invocation-count"),
                ((TrainerProvenanceImpl) reproducedModel.getProvenance()
                        .getTrainerProvenance()
                        .getConfiguredParameters()
                        .get("innerTrainer"))
                .getInstanceValues()
                .get("train-invocation-count"));

        // make sure the main rft setInvocationCount has occurred correctly.
        assertEquals(model2.getProvenance().getTrainerProvenance().getInstanceValues().get("train-invocation-count"),
                reproducedModel.getProvenance().getTrainerProvenance().getInstanceValues().get("train-invocation-count"));
    }

    @Test
    public void testCSVLoader() throws IOException {
        DataSource<Label> tempSource = getCSVLoaderSource();

        MutableDataset<Label> trainingDataset = new MutableDataset<>(tempSource);

        Trainer<Label> trainer = new LogisticRegressionTrainer();
        Model<Label> tempModel = trainer.train(trainingDataset);
        ReproUtil<Label> repro = new ReproUtil<>(tempModel);

        assertDoesNotThrow((Executable) repro::reproduceFromProvenance);
    }

}
