package org.tribuo.reproducibility;

import com.oracle.labs.mlrg.olcut.config.ConfigurationData;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.property.SimpleProperty;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DoubleProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.LongProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.*;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.provenance.DataSourceProvenance;
import org.tribuo.provenance.ModelProvenance;

import java.lang.reflect.InvocationTargetException;
import java.util.Iterator;
import java.util.List;

public final class ReproUtil {
    
    private ReproUtil () {}

    private static Trainer recoverTrainer(ModelProvenance provenance, ConfigurationManager CM){
        Class trainerClass = null;

        // Recover the name of the trainer class from the model's provenance
        // Convert to a class object so it can be passed to the config manager to recover the trainer object
        try {
            trainerClass = Class.forName(provenance.getTrainerProvenance().getClassName());
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        // Use the class identified from the prov to create a new Trainer from the config manager that
        // is the same as the original trainer used.
        String trainerName = (String) CM.listAll(trainerClass).get(0);
        Trainer newTrainer = (Trainer) CM.lookup(trainerName);

        // RNG changes state each time train is called, so examine prov for how many invocations of train
        // had been called when the original model was trained. Then, set the RNG to the same state
        newTrainer.setInvocationCount((int) provenance
                .getTrainerProvenance()
                .getInstanceValues()
                .get("train-invocation-count")
                .getValue());

        return newTrainer;
    }

    private static DataSource getDatasourceFromCM(Class dataSourceClass, ConfigurationManager CM, List<Pair<String, String>> propertyNameAndValues){
        List sources = CM.listAll(dataSourceClass);
        String sourceName = null;
        if (sources.size() > 0){
            sourceName = (String) sources.get(0);
        }

        // If the data source can be recovered from CM do it here.
        DataSource dataSource = null;
        if (sourceName != null){
            if(propertyNameAndValues != null){
                for(Pair<String, String> propertyNameAndValue : propertyNameAndValues){
                    CM.overrideConfigurableProperty(sourceName, propertyNameAndValue.getA(), new SimpleProperty(propertyNameAndValue.getB()));
                }
            }
            dataSource  = (DataSource) CM.lookup(sourceName);
        }

        if (dataSource == null) {
            throw new IllegalArgumentException("The provided provenance has no data source");
        }

        return dataSource;
    }

    private static Dataset datasetReflection(DataSource modelSource, ModelProvenance provenance){
        Dataset modelDataset = null;
        Iterator<Pair<String, Provenance>> sourceProv = provenance.getDatasetProvenance().iterator();
        String datasetClassname = null;
        while (sourceProv.hasNext()){
            Pair<String, Provenance> provPair = sourceProv.next();
            if(provPair.getA() == "class-name"){
                datasetClassname = ((StringProvenance) provPair.getB()).getValue();
            }
        }

        try {
            modelDataset = (Dataset) Class.forName(datasetClassname).getConstructor(DataSource.class).newInstance(modelSource);
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.printStackTrace();
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        return modelDataset;
    }

    private static Dataset recoverDataset(ModelProvenance provenance, ConfigurationManager CM, List<Pair<String, String>> propertyNameAndValues){
        Class dataSourceClass = null;
        Dataset modelDataset = null;

        try {
            dataSourceClass = Class.forName(provenance.getDatasetProvenance().getSourceProvenance().getClassName());
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        if (provenance.getDatasetProvenance().getSourceProvenance() instanceof TrainTestSplitter.SplitDataSourceProvenance) {
            Iterator<Pair<String, Provenance>> sourceProv = provenance.getDatasetProvenance().getSourceProvenance().iterator();
            long seed = 0;
            double trainProportion = 0;
            boolean isTrain = true;
            DataSourceProvenance innerProvenance = null;
            while (sourceProv.hasNext()) {
                Pair<String, Provenance> provPair = sourceProv.next();
                if (provPair.getA() == "source") {
                    innerProvenance = (DataSourceProvenance) provPair.getB();
                } else if (provPair.getA() == "seed") {
                    seed = ((LongProvenance) provPair.getB()).getValue();
                } else if (provPair.getA() == "train-proportion") {
                    trainProportion = ((DoubleProvenance) provPair.getB()).getValue();
                } else if (provPair.getA() == "is-train") {
                    isTrain = ((BooleanProvenance) provPair.getB()).getValue();
                }
            }

            try {
                dataSourceClass = Class.forName(innerProvenance.getClassName());
            } catch (ClassNotFoundException e) {
                e.printStackTrace();
            }

            DataSource innerSource = getDatasourceFromCM(dataSourceClass, CM, propertyNameAndValues);

            TrainTestSplitter trainTestSplitter = new TrainTestSplitter<>(innerSource,trainProportion,seed);

            if(isTrain){
                modelDataset = datasetReflection(trainTestSplitter.getTrain(), provenance);
            } else {
                modelDataset = datasetReflection(trainTestSplitter.getTest(), provenance);
            }

        } else {

            DataSource modelSource = getDatasourceFromCM(dataSourceClass, CM, propertyNameAndValues);
            modelDataset = datasetReflection(modelSource, provenance);
        }


        return modelDataset;
    }

    /**
     * Using a supplied {@link ModelProvenance} object, recreates a model object that the provenance describes.
     * <p>
     * Recovers the trainer and dataset information before training an identical model.
     * @param provenance The provenance describing the model that is to be reproduced.
     * @return A reproduced model identical to the one described in the provenance.
     */
    public static Model reproduceFromProvenance(ModelProvenance provenance){
        return(reproduceFromProvenance(provenance, null));
    }

    /**
     * Using a supplied {@link ModelProvenance} object, recreates a model object that the provenance describes.
     * <p>
     * Recovers the trainer and dataset information before training an identical model.
     * @param provenance The provenance describing the model that is to be reproduced.
     * @param propertyNameAndValues List of pairs where each pair has a configuration name, and
     *                             a new value for that configuration.
     * @return A reproduced model identical to the one described in the provenance.
     */
    public static Model reproduceFromProvenance(ModelProvenance provenance, List<Pair<String, String>> propertyNameAndValues){
        // Load provenance into the config manager so we can extract the necessary classes from config
        List<ConfigurationData> provConfig = ProvenanceUtil.extractConfiguration(provenance);
        ConfigurationManager newCM = new ConfigurationManager();
        newCM.addConfiguration(provConfig);

        Trainer newTrainer = recoverTrainer(provenance, newCM);

        // recoverDataset returns the dataset used to train the model
        // and handles TrainTestSplitters, so if a model uses one this
        // function will return the training data specifically
        Dataset newDataset = recoverDataset(provenance, newCM, propertyNameAndValues);



        Model newModel = newTrainer.train(newDataset);

        return newModel;
    }

    /**
     * Using a supplied {@link Model} object, recreates an identical model object that the provenance describes.
     * <p>
     * Recovers the trainer and dataset information before training an identical model.
     * @param originalModel The provenance describing the model that is to be reproduced.
     * @return A reproduced model identical to the one described in the provenance.
     */
    public static Model reproduceFromModel(Model originalModel) throws Exception {

        Model newModel = reproduceFromProvenance(originalModel.getProvenance());

        ImmutableFeatureMap newFeatureMap = newModel.getFeatureIDMap();
        ImmutableFeatureMap oldFeatureMap = originalModel.getFeatureIDMap();
        if(newFeatureMap.size() == oldFeatureMap.size()){
            for(int featureIndex = 0; featureIndex < newFeatureMap.size(); featureIndex++){
                if(!newFeatureMap.get(featureIndex).toString().equals(oldFeatureMap.get(featureIndex).toString())){
                    throw new Exception("Recreated model has different feature map");
                }
            }
        }

        if(!originalModel.getOutputIDInfo().toString().equals(newModel.getOutputIDInfo().toString())){
            throw new Exception("Recreated model has different output domain");
        }

        return newModel;
    }
}
