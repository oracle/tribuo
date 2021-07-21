package org.tribuo.reproducibility;

import com.oracle.labs.mlrg.olcut.config.property.SimpleProperty;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.FileProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.sgd.linear.LinearSGDModel;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.common.sgd.AbstractLinearSGDModel;
import org.tribuo.common.sgd.AbstractLinearSGDTrainer;
import org.tribuo.common.sgd.AbstractSGDTrainer;
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
import org.tribuo.data.text.impl.BasicPipeline;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.tribuo.regression.sgd.linear.LinearSGDTrainer;
import org.tribuo.regression.sgd.objectives.SquaredLoss;
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

class ReproUtilTest {

    @BeforeAll
    public static void setup() {
        Class<?>[] classes = new Class<?>[]{AbstractSGDTrainer.class, AbstractLinearSGDTrainer.class,LinearSGDTrainer.class};
        for (Class c : classes) {
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

        ReproUtil reproUtil = new ReproUtil(model.getProvenance());
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

        ReproUtil reproUtil = new ReproUtil(model.getProvenance());
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

        ReproUtil reproUtil = new ReproUtil(model);
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
        ReproUtil reproUtil = new ReproUtil(model.getProvenance());
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
    public void testoverrideConfigurableProperty(){
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
        ReproUtil reproUtil = new ReproUtil(model.getProvenance());
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
}