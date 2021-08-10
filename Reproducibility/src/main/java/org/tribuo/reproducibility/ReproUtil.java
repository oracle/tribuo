package org.tribuo.reproducibility;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.oracle.labs.mlrg.olcut.config.ConfigurationData;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.property.SimpleProperty;
import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import com.oracle.labs.mlrg.olcut.provenance.MapProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.PrimitiveProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
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
import java.lang.reflect.Method;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;

public class ReproUtil {

    // These fields are used to denote which value came from which provenance in a diff
    private final static String OLD = "original";
    private final static String NEW = "reproduced";

    private ConfigurationManager CM;

    private ModelProvenance modelProvenance;
    private Model originalModel;

    private ReproUtil () {}

    // This method takes an iterator from a prov object and converts it into a sorted map
    // It uses a sorted map just in case two prov objects are the same type but iterate in a different order.
    private static TreeMap<String, Provenance> iterToMap(Iterator<Pair<String, Provenance>> iter){
        TreeMap<String, Provenance> provMap = new TreeMap<>();
        while (iter.hasNext()){
            Pair<String, Provenance> provPair = iter.next();
            provMap.put(provPair.getA(), provPair.getB());
        }

        return new TreeMap<>(provMap);
    }

    // This method takes two provenance iterators and returns a diff in the form of a JSON ObjectNode
    private static ObjectNode compareModelProvenances(Iterator<Pair<String, Provenance>> iterA, Iterator<Pair<String, Provenance>> iterB){
        // The report ObjectNode is the ultimate return value of the method,
        // All diff values are put in it.
        ObjectMapper mapper = new ObjectMapper();
        ObjectNode report = mapper.createObjectNode();

        // Convert the iterators to a sorted map, so it can iterate through the keys to find matching values
        TreeMap<String, Provenance> provMapA = iterToMap(iterA);
        TreeMap<String, Provenance> provMapB = iterToMap(iterB);

        Set<String> keys = provMapA.keySet();
        for(String key : keys){
            // When provenance is primitive and they share a key simply compare their values
            // If different, place each prov's value in the report
            if(provMapA.get(key) instanceof PrimitiveProvenance && provMapB.get(key) instanceof PrimitiveProvenance){
                if(!provMapA.get(key).equals(provMapB.get(key))){
                    ObjectNode val_diff = mapper.createObjectNode();
                    val_diff.put(OLD, ((PrimitiveProvenance<?>) provMapA.get(key)).getValue().toString());
                    val_diff.put(NEW, ((PrimitiveProvenance<?>) provMapB.get(key)).getValue().toString());
                    report.set(key, val_diff);
                }
            }
            // In the event it's a different prov object, and they are the same time, recursively call
            // this method on the iterators of the respective prov objects, and then merge the resulting
            // ObjectNode into the report in this frame.
            else if(provMapA.get(key).getClass() == provMapB.get(key).getClass()){

                // ObjectProvenance and MapProvenance will always return Iterator<Pair<String, Provenance>> so can
                // be handled simply here through recursion of this method.
                if (provMapA.get(key) instanceof ObjectProvenance || provMapA.get(key) instanceof MapProvenance){

                    // There is no abstract provenance object that has the iterator method, so use reflection to determine
                    // what the type of the provenance is (Dataset, Datasource, Trainer, etc), then get the iterator method
                    Method iterator_A = null;
                    Method iterator_B = null;
                    try {
                        iterator_A = provMapA.get(key).getClass().getMethod("iterator", (Class<?>[]) null);
                        iterator_B = provMapB.get(key).getClass().getMethod("iterator", (Class<?>[]) null);
                    } catch (NoSuchMethodException e) {
                        //TODO: Do more to handle this?
                        e.printStackTrace();
                    }

                    // Use the method identified in the previous step to actually get the iterator for each prov object.
                    Iterator<Pair<String, Provenance>> subIterA = null;
                    Iterator<Pair<String, Provenance>> subIterB = null;
                    try {
                        subIterA = (Iterator<Pair<String, Provenance>>) iterator_A.invoke(provMapA.get(key), (Object[]) null);
                        subIterB = (Iterator<Pair<String, Provenance>>) iterator_B.invoke(provMapB.get(key), (Object[]) null);
                    } catch (IllegalAccessException e) {
                        //TODO: Do more to handle these?
                        e.printStackTrace();
                    } catch (InvocationTargetException e) {
                        e.printStackTrace();
                    }

                    // Recursively identify any diffs down to primitive objects
                    ObjectNode sub_report = compareModelProvenances(subIterA, subIterB);

                    // Only add the new ObjectNode if it is not empty, to prevent unwanted keys in the resulting JSON
                    if(!sub_report.isEmpty()){
                        report.set(key, sub_report);
                    }
                }
                // ListProvenance might contain Pair<String, Provenance> but it might also contain a list of
                // ConfiguredObjectProvenanceImpl. This handles that situation directly.
                else if (provMapA.get(key) instanceof ListProvenance){

                    ListProvenance listProvA = ((ListProvenance<?>) provMapA.get(key));
                    ListProvenance listProvB = ((ListProvenance<?>) provMapB.get(key));

                    // If it is a Pair, handle it with recursion as if it was an Object or Map Provenance.
                    if(listProvA.getList().size() > 0 && listProvA.getList().get(0) instanceof Pair){
                        Iterator subIterA = ((ListProvenance<?>) provMapA.get(key)).iterator();
                        Iterator subIterB = ((ListProvenance<?>) provMapB.get(key)).iterator();

                        ObjectNode sub_report = compareModelProvenances(subIterA, subIterB);
                        if(!sub_report.isEmpty()){
                            report.set(key, sub_report);
                        }
                    }
                    // If the list contains ConfiguredObjectProvenanceImpl then do something different.
                    //TODO: Iterate through these?
                    else if (listProvA.getList().size() > 0 && listProvA.getList().get(0) instanceof ConfiguredObjectProvenanceImpl){

                        if(!listProvA.equals(listProvB)){
                            ObjectNode val_diff = mapper.createObjectNode();
                            val_diff.put(OLD, listProvA.toString());
                            val_diff.put(NEW, listProvB.toString());
                            report.set(key, val_diff);
                        }

                    }

                } else {
                    //TODO: Error handling here
                    System.out.println("Unrecognized Provenance: ");
                    System.out.println(provMapA.get(key).getClass());
                }
            }
        }

        return report;
    }

    /**
     * Creates a JSON String diff of two {@link ModelProvenance} objects. Only the differences will appear
     * in the resulting diff.
     * <p>
     * Recovers the trainer and dataset information before training an identical model.
     * @param originalProvenance The first of the two provenance objects to diff
     * @param newProvenance The second of the two provenance objects to diff
     * @return A String JSON report displaying the differences in the model.
     */
    public static String diffProvenance(ModelProvenance originalProvenance, ModelProvenance newProvenance){

        Iterator<Pair<String, Provenance>> originalIter = originalProvenance.iterator();
        Iterator<Pair<String, Provenance>> newIter = newProvenance.iterator();


        String report = compareModelProvenances(originalIter, newIter).toPrettyString();
        return report;
    }

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

    public Trainer recoverTrainer(){
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

    // This method takes a class object and will recover that datasource from the ConfigurationManager
    // It also can take a list of properties to overwrite, allowing users to change the datapath to their data
    // or other similar properties.
    private DataSource getDatasourceFromCM(Class dataSourceClass, List<Pair<String, String>> propertyNameAndValues){

        // Since this utility created a CM from a model, contained within should only be the datasource used to
        // create the model. Unless a user has manually added another datasource.
        List sources = CM.listAll(dataSourceClass);
        String sourceName = null;
        if (sources.size() > 0){
            sourceName = (String) sources.get(0);
        }

        // If the data source can be recovered from CM, override the properties before the dataSource is
        // instantiated, and then instantiate the source.
        DataSource dataSource = null;
        if (sourceName != null){
            if(propertyNameAndValues != null){
                for(Pair<String, String> propertyNameAndValue : propertyNameAndValues){
                    CM.overrideConfigurableProperty(sourceName, propertyNameAndValue.getA(), new SimpleProperty(propertyNameAndValue.getB()));
                }
            }
            dataSource  = (DataSource) CM.lookup(sourceName);
        }

        // Throw error if the source cannot be instantiated.
        if (dataSource == null) {
            throw new IllegalArgumentException("The provided provenance has no data source");
        }

        return dataSource;
    }


    // Attempts to use reflection to determine the correct type a dataset should be and then instantiates it
    // returns null if cannot be done.
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

    // Return a dataset used when a model was trained.
    private Dataset recoverDataset(List<Pair<String, String>> propertyNameAndValues){
        Class dataSourceClass = null;
        Dataset modelDataset = null;

        // The class is used to query the ConfigurationManager for the datasource object
        try {
            dataSourceClass = Class.forName(this.modelProvenance.getDatasetProvenance().getSourceProvenance().getClassName());
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        // If the source is a TrainTestSplitter, we need to extract the correct data from the Splitter first.
        // While it is likely they trained on the "train" dataset, in the future they also might have used cross-validation.
        if (this.modelProvenance.getDatasetProvenance().getSourceProvenance() instanceof TrainTestSplitter.SplitDataSourceProvenance) {

            Iterator<Pair<String, Provenance>> sourceProv = this.modelProvenance.getDatasetProvenance().getSourceProvenance().iterator();

            // First collect all the information about the TrainTestSplitter possible including the inner data source
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

            // Recreating the trainTestSplitter with the parameters gathered from the provenance, including the
            // innerSource, should return the same Dataset used to train the model
            DataSource innerSource = getDatasourceFromCM(dataSourceClass, propertyNameAndValues);
            TrainTestSplitter trainTestSplitter = new TrainTestSplitter<>(innerSource,trainProportion,seed);

            if(isTrain){
                modelDataset = datasetReflection(trainTestSplitter.getTrain());
            } else {
                modelDataset = datasetReflection(trainTestSplitter.getTest());
            }

        } else {
            // When it's not a splitter, recover the Datasource and then Dataset.
            DataSource modelSource = getDatasourceFromCM(dataSourceClass, propertyNameAndValues);
            modelDataset = datasetReflection(modelSource);
        }

        return modelDataset;
    }

    /**
     * Returns the ConfigurationManager the ReproUtility is using to manage the reproduced models.
     * <p>
     * @return a ConfigurationManager the ReproUtility is managing.
     */
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

        // TODO: Expand upon this
        if(!originalModel.getOutputIDInfo().toString().equals(newModel.getOutputIDInfo().toString())){
            throw new Exception("Recreated model has different output domain");
        }

        return newModel;
    }
}
