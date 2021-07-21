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
import org.tribuo.DataSource;
import org.tribuo.Dataset;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;
import org.tribuo.Trainer;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.provenance.DataSourceProvenance;
import org.tribuo.provenance.ModelProvenance;

import java.lang.reflect.InvocationTargetException;
import java.util.Iterator;
import java.util.List;

public class ReproUtil {

    private ConfigurationManager CM;

    private ModelProvenance modelProvenance;

    private Model originalModel;

    private ReproUtil () {}

    public ReproUtil(ModelProvenance provenance){
        this(provenance, null);
    }

    public ReproUtil(Model originalModel){
        this(originalModel.getProvenance(), originalModel);
    }

    private ReproUtil(ModelProvenance provenance, Model originalModel){
        this.modelProvenance = provenance;

        // Load configurations from provenance so we can re-instantiate objects using the ConfigManager
        // Additionally allows us to change the values of certain configurable fields before re-instantiation
        List<ConfigurationData> provConfig = ProvenanceUtil.extractConfiguration(this.modelProvenance);
        this.CM = new ConfigurationManager();
        this.CM.addConfiguration(provConfig);

        this.originalModel = originalModel;
    }

    private Trainer recoverTrainer(){
        Class trainerClass = null;

        // Recover the name of the trainer class from the model's provenance
        // Convert to a class object so it can be passed to the config manager to recover the trainer object
        try {
            trainerClass = Class.forName(this.modelProvenance.getTrainerProvenance().getClassName());
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        // Use the class identified from the prov to create a new Trainer from the config manager that
        // is the same as the original trainer used.
        String trainerName = (String) CM.listAll(trainerClass).get(0);
        Trainer newTrainer = (Trainer) CM.lookup(trainerName);

        // RNG changes state each time train is called, so examine prov for how many invocations of train
        // had been called when the original model was trained. Then, set the RNG to the same state
        newTrainer.setInvocationCount((int) this.modelProvenance
                .getTrainerProvenance()
                .getInstanceValues()
                .get("train-invocation-count")
                .getValue());

        return newTrainer;
    }

    private DataSource getDatasourceFromCM(Class dataSourceClass, List<Pair<String, String>> propertyNameAndValues){
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

    private Dataset datasetReflection(DataSource modelSource){
        Dataset modelDataset = null;
        Iterator<Pair<String, Provenance>> sourceProv = this.modelProvenance.getDatasetProvenance().iterator();
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

    private Dataset recoverDataset(List<Pair<String, String>> propertyNameAndValues){
        Class dataSourceClass = null;
        Dataset modelDataset = null;

        try {
            dataSourceClass = Class.forName(this.modelProvenance.getDatasetProvenance().getSourceProvenance().getClassName());
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        if (this.modelProvenance.getDatasetProvenance().getSourceProvenance() instanceof TrainTestSplitter.SplitDataSourceProvenance) {
            Iterator<Pair<String, Provenance>> sourceProv = this.modelProvenance.getDatasetProvenance().getSourceProvenance().iterator();
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

            DataSource innerSource = getDatasourceFromCM(dataSourceClass, propertyNameAndValues);

            TrainTestSplitter trainTestSplitter = new TrainTestSplitter<>(innerSource,trainProportion,seed);

            if(isTrain){
                modelDataset = datasetReflection(trainTestSplitter.getTrain());
            } else {
                modelDataset = datasetReflection(trainTestSplitter.getTest());
            }

        } else {

            DataSource modelSource = getDatasourceFromCM(dataSourceClass, propertyNameAndValues);
            modelDataset = datasetReflection(modelSource);
        }


        return modelDataset;
    }

    public ConfigurationManager getConfigurationManager(){
        return CM;
    }

    /**
     * Recreates a model object using the {@link ModelProvenance} supplied when the ReproUtil object was created.
     * <p>
     * Recovers the trainer and dataset information before training an identical model.
     * @return A reproduced model identical to the one described in the provenance.
     */
    public Model reproduceFromProvenance(){
        return(reproduceFromProvenance(null));
    }

    /**
     * Recreates a model object using the {@link ModelProvenance} supplied when the ReproUtil object was created.
     * Additionally allows calling code to pass a list of new values for configurable properties
     * <p>
     * Recovers the trainer and dataset information before training an identical model.
     * @param propertyNameAndValues List of pairs where each pair has a configuration name, and
     *                             a new value for that configuration.
     * @return A reproduced model identical to the one described in the provenance.
     */
    public Model reproduceFromProvenance(List<Pair<String, String>> propertyNameAndValues){

        // Until now the object only holds the configuration for these objects, the following
        // functions will actually re-instantiate them.
        Trainer newTrainer = recoverTrainer();
        Dataset newDataset = recoverDataset(propertyNameAndValues);

        // This function actually re-trains a model rather than copy the original
        Model newModel = newTrainer.train(newDataset);

        return newModel;
    }

    /**
     * Using a supplied {@link Model} object, recreates an identical model object that the provenance describes.
     * <p>
     * Recovers the trainer and dataset information before training an identical model.
     * @return A reproduced model identical to the one described in the provenance.
     */
    public Model reproduceFromModel() throws Exception {
        // TODO: Create types for these exceptions
        if(this.originalModel == null){
            throw new Exception("No model to reproduce was provided");
        }

        Model newModel = reproduceFromProvenance();

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
