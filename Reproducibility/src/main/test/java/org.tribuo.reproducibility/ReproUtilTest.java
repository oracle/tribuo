package org.tribuo.reproducibility;

import com.oracle.labs.mlrg.olcut.config.property.SimpleProperty;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.FileProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
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
import org.tribuo.data.text.impl.BasicPipeline;
import org.tribuo.ensemble.BaggingTrainer;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.ensemble.AveragingCombiner;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.tribuo.regression.rtree.CARTRegressionTrainer;
import org.tribuo.regression.rtree.impurity.MeanSquaredError;
import org.tribuo.regression.sgd.linear.LinearSGDTrainer;
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.tribuo.common.tree.AbstractCARTTrainer.MIN_EXAMPLES;

class ReproUtilTest {

    @BeforeAll
    public static void setup() {
        Class<?>[] classes = new Class<?>[]{AbstractSGDTrainer.class, AbstractLinearSGDTrainer.class,LinearSGDTrainer.class, BaggingTrainer.class};
        for (Class<?> c : classes) {
            Logger logger = Logger.getLogger(c.getName());
            logger.setLevel(Level.WARNING);
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
    public void testChangeDataPath(){
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
        List<Pair<String, String>> newConfigs = Arrays.asList(new Pair<String, String>("dataPath", csvPath.toString()));
        ReproUtil reproUtil = null;
        try {
            reproUtil = new ReproUtil(model.getProvenance());
        } catch (Exception e) {
            e.printStackTrace();
        }
        LinearSGDModel newModel = (LinearSGDModel) reproUtil.reproduceFromProvenance(newConfigs);

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
    public void testBaggingTrainer(){
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();

        CARTRegressionTrainer subsamplingTree = new CARTRegressionTrainer(Integer.MAX_VALUE,
                MIN_EXAMPLES, 0.0f, 0.5f, false, new MeanSquaredError(), Trainer.DEFAULT_SEED);
        RandomForestTrainer<Regressor> rfT = new RandomForestTrainer<>(subsamplingTree,new AveragingCombiner(),10);

        Model<Regressor> model = rfT.train(p.getA());
        rfT.train(p.getA());
        Model<Regressor> diff_model = rfT.train(p.getA());

        ReproUtil reproUtil = null;
        try {
            reproUtil = new ReproUtil(model.getProvenance());
        } catch (Exception e) {
            e.printStackTrace();
        }
        RandomForestTrainer<Regressor> new_rfT = (RandomForestTrainer<Regressor>) reproUtil.recoverTrainer();
        Model<Regressor> new_model = new_rfT.train(p.getA());

    }


}
