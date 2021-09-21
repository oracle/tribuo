package org.tribuo.reproducibility;

import com.oracle.labs.mlrg.olcut.config.property.SimpleProperty;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.FileProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.tribuo.Dataset;
import org.tribuo.DataSource;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
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
import java.util.Iterator;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.tribuo.common.tree.AbstractCARTTrainer.MIN_EXAMPLES;

class ReproUtilTest {
    @TempDir
    static Path tempDir;

    static Path tempFile;

    @BeforeAll
    public static void setup() {
        Class<?>[] classes = new Class<?>[]{AbstractSGDTrainer.class, AbstractLinearSGDTrainer.class,LinearSGDTrainer.class, BaggingTrainer.class};
        for (Class<?> c : classes) {
            Logger logger = Logger.getLogger(c.getName());
            logger.setLevel(Level.WARNING);
        }

        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();

        tempFile = null;
        try {
            tempFile = Files.createFile(tempDir.resolve("dense.csv"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        CSVSaver saver = new CSVSaver();

        try {
            saver.save(tempFile, p.getA(), "response");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    private CSVDataSource getCSVDatasource(){
        URL u = ReproUtilTest.class.getResource("/test.csv");
        Path csvPath = null;
        try {
            csvPath = Paths.get(u.toURI());
        } catch (URISyntaxException e) {
            e.printStackTrace();
        }
        List<String> csvLines = null;
        try {
            csvLines = Files.readAllLines(csvPath, StandardCharsets.UTF_8);
        } catch (IOException e) {
            e.printStackTrace();
        }

        BasicPipeline textPipeline = new BasicPipeline(new BreakIteratorTokenizer(Locale.US),2);
        HashMap<String, FieldProcessor> fieldProcessors = new HashMap<String, FieldProcessor>();
        fieldProcessors.put("height",new DoubleFieldProcessor("height"));
        fieldProcessors.put("description",new TextFieldProcessor("description",textPipeline));
        fieldProcessors.put("transport",new IdentityProcessor("transport"));

        HashMap<String,FieldProcessor> regexMappingProcessors = new HashMap<String,FieldProcessor>();
        regexMappingProcessors.put("extra.*", new DoubleFieldProcessor("extra.*"));

        FieldResponseProcessor responseProcessor = new FieldResponseProcessor("disposition","UNK",new LabelFactory());

        ArrayList<FieldExtractor<?>> metadataExtractors = new ArrayList<FieldExtractor<?>>();
        metadataExtractors.add(new IntExtractor("id"));
        metadataExtractors.add(new DateExtractor("timestamp","timestamp","dd/MM/yyyy HH:mm"));

        FloatExtractor weightExtractor = new FloatExtractor("example-weight");

        RowProcessor<Label> rowProcessor = new RowProcessor<Label>(metadataExtractors,weightExtractor,responseProcessor,fieldProcessors,regexMappingProcessors, Collections.emptySet());

        CSVDataSource dataSource = new CSVDataSource<Label>(csvPath,rowProcessor,true);

        return(dataSource);
    }

    // Generates a CSV file, writes it to disk, and loads it back in using a CSVLoader for the purposes of
    // testing the reproducibility of CSVLoader objects.
    // Data originally generated by LabelledDataGenerator.denseTrainTest()
    private DataSource getCSVLoadersource() {
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
        try {
            tempFile = Files.createFile(tempDir.resolve("test.csv"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            Files.write(tempFile, tempData.getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            e.printStackTrace();
        }
        LabelFactory labelFactory = new LabelFactory();
        CSVLoader csvLoader = new CSVLoader<>(labelFactory);

        var tempHeaders = new String[]{"response", "A", "B", "C", "D"};
        DataSource tempSource = null;

        try {
            tempSource = csvLoader.loadDataSource(tempFile,"response", tempHeaders);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return tempSource;
    }

    private DataSource getConfigurableRegressionDenseTrain(){
        DataSource tempSource = null;
        RegressionFactory regressionFactory = new RegressionFactory();
        CSVLoader csvLoader = new CSVLoader(regressionFactory);
        try {
            tempSource = csvLoader.loadDataSource(tempFile,"response");
        } catch (IOException e) {
            e.printStackTrace();
        }

        return tempSource;
    }

    @Test
    public void testReproduceFromProvenanceWithSplitter(){
        CSVDataSource csvSource = getCSVDatasource();

        TrainTestSplitter trainTestSplitter = new TrainTestSplitter<>(csvSource,0.7,1L);
        MutableDataset trainingDataset = new MutableDataset<>(trainTestSplitter.getTrain());


        LogisticRegressionTrainer trainer = new LogisticRegressionTrainer();
        LinearSGDModel model = (LinearSGDModel) trainer.train(trainingDataset);
        model = (LinearSGDModel) trainer.train(trainingDataset);
        model = (LinearSGDModel) trainer.train(trainingDataset);

        ReproUtil reproUtil = null;
        try {
            reproUtil = new ReproUtil(model.getProvenance());
        } catch (Exception e) {
            e.printStackTrace();
        }
        LinearSGDModel newModel = (LinearSGDModel) reproUtil.reproduceFromProvenance();

        assertTrue(newModel.getWeightsCopy().equals(model.getWeightsCopy()));
    }

    @Test
    public void testReproduceFromProvenanceNoSplitter(){
        CSVDataSource csvSource = getCSVDatasource();
        MutableDataset datasetFromCSV = new MutableDataset<Label>(csvSource);

        LogisticRegressionTrainer trainer = new LogisticRegressionTrainer();
        LinearSGDModel model = (LinearSGDModel) trainer.train(datasetFromCSV);
        model = (LinearSGDModel) trainer.train(datasetFromCSV);
        model = (LinearSGDModel) trainer.train(datasetFromCSV);

        ReproUtil reproUtil = null;
        try {
            reproUtil = new ReproUtil(model.getProvenance());
        } catch (Exception e) {
            e.printStackTrace();
        }
        LinearSGDModel newModel = (LinearSGDModel) reproUtil.reproduceFromProvenance();

        assertTrue(newModel.getWeightsCopy().equals(model.getWeightsCopy()));
    }

    @Test
    public void testReproduceFromModel(){
        CSVDataSource csvSource = getCSVDatasource();
        MutableDataset datasetFromCSV = new MutableDataset<Label>(csvSource);

        LogisticRegressionTrainer trainer = new LogisticRegressionTrainer();
        LinearSGDModel model = (LinearSGDModel) trainer.train(datasetFromCSV);
        model = (LinearSGDModel) trainer.train(datasetFromCSV);
        model = (LinearSGDModel) trainer.train(datasetFromCSV);

        ReproUtil reproUtil = null;
        try {
            reproUtil = new ReproUtil(model);
        } catch (Exception e) {
            e.printStackTrace();
        }
        LinearSGDModel newModel = null;
        try {
            newModel = (LinearSGDModel) reproUtil.reproduceFromModel();
        } catch (Exception e) {
            e.printStackTrace();
        }

        assertTrue(newModel.getWeightsCopy().equals(model.getWeightsCopy()));
    }

    @Test
    public void testOverrideConfigurableProperty(){
        CSVDataSource csvSource = getCSVDatasource();
        MutableDataset datasetFromCSV = new MutableDataset<Label>(csvSource);

        LogisticRegressionTrainer trainer = new LogisticRegressionTrainer();
        LinearSGDModel model = (LinearSGDModel) trainer.train(datasetFromCSV);
        model = (LinearSGDModel) trainer.train(datasetFromCSV);
        model = (LinearSGDModel) trainer.train(datasetFromCSV);

        URL u = ReproUtilTest.class.getResource("/new_dir/new_data.csv");
        Path csvPath = null;
        try {
            csvPath = Paths.get(u.toURI());
        } catch (URISyntaxException e) {
            e.printStackTrace();
        }
        ReproUtil reproUtil = null;
        try {
            reproUtil = new ReproUtil(model.getProvenance());
        } catch (Exception e) {
            e.printStackTrace();
        }
        reproUtil.getConfigurationManager().overrideConfigurableProperty("csvdatasource-1", "dataPath", new SimpleProperty(csvPath.toString()));
        LinearSGDModel newModel = (LinearSGDModel) reproUtil.reproduceFromProvenance();

        assertTrue(newModel.getWeightsCopy().equals(model.getWeightsCopy()));

        Iterator<Pair<String, Provenance>> sourceProv = newModel.getProvenance().getDatasetProvenance().getSourceProvenance().iterator();

        while (sourceProv.hasNext()){
            Pair<String, Provenance> provPair = sourceProv.next();
            if(provPair.getA() == "dataPath"){
                assertEquals("new_data.csv", ((FileProvenance) provPair.getB()).getValue().getName());
            }
        }
    }


    @Test
    public void testProvDiff(){
        //TODO: Expand this to actually assert something
        CSVDataSource csvSource = getCSVDatasource();
        MutableDataset datasetFromCSV = new MutableDataset<Label>(csvSource);

        LogisticRegressionTrainer trainer = new LogisticRegressionTrainer();
        LinearSGDModel model_1 = (LinearSGDModel) trainer.train(datasetFromCSV);
        LinearSGDModel model_2 = (LinearSGDModel) trainer.train(datasetFromCSV);
        ReproUtil repro = null;
        try {
            repro = new ReproUtil(model_1);
        } catch (Exception e) {
            e.printStackTrace();
        }
        LinearSGDModel model_3 = (LinearSGDModel) repro.reproduceFromProvenance();
        String report = ReproUtil.diffProvenance(model_1.getProvenance(), model_3.getProvenance());
        System.out.println(report);
    }

    @Test
    public void testProvDiffWithTransformTrainer(){
        //TODO: Expand this to actually assert something
        CSVDataSource csvSource = getCSVDatasource();
        MutableDataset datasetFromCSV = new MutableDataset<Label>(csvSource);

        LogisticRegressionTrainer trainer = new LogisticRegressionTrainer();
        LinearSGDModel model_1 = (LinearSGDModel) trainer.train(datasetFromCSV);

        TransformationMap transformations = new TransformationMap(List.of(new LinearScalingTransformation(0,1)));
        TransformTrainer transformed = new TransformTrainer(trainer, transformations);
        Model transformedModel = transformed.train(datasetFromCSV);

        String report = ReproUtil.diffProvenance(model_1.getProvenance(), transformedModel.getProvenance());
        System.out.println(report);
    }


    @Test
    public void testBaggingTrainer(){
        //TODO: Will need more extensive testing here
        CARTRegressionTrainer subsamplingTree = new CARTRegressionTrainer(Integer.MAX_VALUE,
                MIN_EXAMPLES, 0.0f, 0.5f, false, new MeanSquaredError(), Trainer.DEFAULT_SEED);
        RandomForestTrainer<Regressor> rfT = new RandomForestTrainer<>(subsamplingTree,new AveragingCombiner(),10);
        Dataset<Regressor> trainData = new MutableDataset<>(getConfigurableRegressionDenseTrain());
        Model<Regressor> model = rfT.train(trainData);

        ReproUtil reproUtil = null;
        try {
            reproUtil = new ReproUtil(model.getProvenance());
        } catch (Exception e) {
            e.printStackTrace();
        }

        Model<Regressor> reproducedModel = reproUtil.reproduceFromProvenance();

        assertEquals(model.getProvenance().getTrainerProvenance().getInstanceValues().get("train-invocation-count"),
                reproducedModel.getProvenance().getTrainerProvenance().getInstanceValues().get("train-invocation-count"));

    }

    @Test
    public void testBaggingTrainerInnerInvocationChange(){
        //TODO: Modify reproutil or bagging trainer to account for this case
        CARTRegressionTrainer subsamplingTree = new CARTRegressionTrainer(Integer.MAX_VALUE,
                MIN_EXAMPLES, 0.0f, 0.5f, false, new MeanSquaredError(), Trainer.DEFAULT_SEED);
        RandomForestTrainer<Regressor> rfT = new RandomForestTrainer<>(subsamplingTree,new AveragingCombiner(),10);
        Dataset<Regressor> trainData = new MutableDataset<>(getConfigurableRegressionDenseTrain());
        Model<Regressor> model1 = rfT.train(trainData);
        subsamplingTree.setInvocationCount(15);
        Model<Regressor> model2 = rfT.train(trainData);

        ReproUtil reproUtil = null;
        try {
            reproUtil = new ReproUtil(model2.getProvenance());
        } catch (Exception e) {
            e.printStackTrace();
        }

        Model<Regressor> reproducedModel = reproUtil.reproduceFromProvenance();

        /*
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

         */
    }

    @Test
    public void testCSVLoader(){
        DataSource<?> tempSource = getCSVLoadersource();

        MutableDataset trainingDataset = new MutableDataset<>(tempSource);

        Trainer<Label> trainer = new LogisticRegressionTrainer();
        Model<Label> tempModel = trainer.train(trainingDataset);
        ReproUtil repro = null;
        try {
            repro = new ReproUtil(tempModel);
        } catch (Exception e) {
            e.printStackTrace();
        }
        ReproUtil finalRepro = repro;
        assertDoesNotThrow(() -> {
            finalRepro.reproduceFromProvenance();
        });

    }

}
