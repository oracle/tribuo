package org.tribuo.reproducibility;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.config.ConfigurationData;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import com.oracle.labs.mlrg.olcut.provenance.PrimitiveProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DoubleProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.LongProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ConfigurableDataSource;
import org.tribuo.DataSource;
import org.tribuo.Dataset;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Trainer;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.interop.ExternalTrainerProvenance;
import org.tribuo.provenance.DataSourceProvenance;
import org.tribuo.provenance.ModelProvenance;

import java.lang.reflect.InvocationTargetException;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;


public class ReproUtil {

    // These fields are used to denote which value came from which provenance in a diff
    private final static String OLD = "original";
    private final static String NEW = "reproduced";

    private ConfigurationManager CM;

    private ModelProvenance modelProvenance;
    private Model<?> originalModel;

    private ReproUtil () {}

    public ReproUtil(ModelProvenance provenance) throws Exception {
        this(provenance, null);
    }

    public ReproUtil(Model<?> originalModel) throws Exception {
        this(originalModel.getProvenance(), originalModel);
    }

    private ReproUtil(ModelProvenance provenance, Model<?> originalModel) throws Exception {
        if (provenance.getTrainerProvenance() instanceof ExternalTrainerProvenance){
            throw new Exception("This version of this tool cannot reproduce external models.");
        }

        this.modelProvenance = provenance;

        // Load configurations from provenance so it can re-instantiate objects using the ConfigManager
        // Additionally allows us to change the values of certain configurable fields before re-instantiation
        List<ConfigurationData> provConfig = ProvenanceUtil.extractConfiguration(this.modelProvenance);
        this.CM = new ConfigurationManager();
        this.CM.addConfiguration(provConfig);

        this.originalModel = originalModel;
    }

    public <T extends Output<T>> Trainer<T> recoverTrainer(){
        Class<? extends Trainer<T>> trainerClass = null;

        // Recover the name of the trainer class from the model's provenance
        // Convert to a class object so it can be passed to the config manager to recover the trainer object
        try {
            trainerClass = (Class<? extends Trainer<T>>) Class.forName(this.modelProvenance.getTrainerProvenance().getClassName());
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        // Use the class identified from the prov to create a new Trainer from the config manager that
        // is the same as the original trainer used.
        String trainerName = CM.listAll(trainerClass).get(0);
        Trainer<T> newTrainer = (Trainer<T>) CM.lookup(trainerName);

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
    private <T extends Output<T>> DataSource<T> getDatasourceFromCM(Class<? extends Configurable> dataSourceClass){

        // Since this utility created a CM from a model, contained within should only be the datasource used to
        // create the model. Unless a user has manually added another datasource.
        List<String> sources = CM.listAll(dataSourceClass);
        String sourceName = null;
        if (sources.size() > 0){
            //TODO: what if there are more than one sources?
            //If configuration data only comes from a model prov there should only be one source
            //but since we expose the configuration manager, a user theoretically could add a new source
            sourceName = sources.get(0);
        }

        // If the data source can be recovered from CM, override the properties before the dataSource is
        // instantiated, and then instantiate the source.
        DataSource<T> dataSource = null;
        if (sourceName != null){
            dataSource  = (DataSource<T>) CM.lookup(sourceName);
        }

        // Throw error if the source cannot be instantiated.
        if (dataSource == null) {
            throw new IllegalArgumentException("The provided provenance has no data source");
        }

        return dataSource;
    }


    // Attempts to use reflection to determine the correct type a dataset should be and then instantiates it
    // returns null if cannot be done.
    private <T extends Output<T>> Dataset<T> datasetReflection(DataSource<?> modelSource){
        // The first step is to find the classname as a String so it can use reflection to instantiate the correct type
        // The class name is contained within the provenance accessible through the iterator.
        Dataset<T> modelDataset = null;
        Iterator<Pair<String, Provenance>> sourceProv = this.modelProvenance.getDatasetProvenance().iterator();
        String datasetClassname = null;
        while (sourceProv.hasNext()){
            Pair<String, Provenance> provPair = sourceProv.next();
            if(provPair.getA() == "class-name"){
                datasetClassname = ((StringProvenance) provPair.getB()).getValue();
            }
        }

        // Once the class is identified, reflection will allow us to search for a constructor for that class and instantiate it
        try {
            modelDataset = (Dataset<T>) Class.forName(datasetClassname).getConstructor(DataSource.class).newInstance(modelSource);
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
    private <T extends Output<T>> Dataset<T> recoverDataset() throws Exception {
        Class<? extends ConfigurableDataSource<?>> dataSourceClass = null;
        Dataset<T> modelDataset = null;

        // The class is used to query the ConfigurationManager for the datasource object
        try {
            dataSourceClass = (Class<? extends ConfigurableDataSource<?>>) Class.forName(this.modelProvenance.getDatasetProvenance().getSourceProvenance().getClassName());
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        // If the source is a TrainTestSplitter, we need to extract the correct data from the Splitter first.
        // While it is likely they trained on the "train" dataset, in the future they also might have used cross-validation.
        if (this.modelProvenance.getDatasetProvenance().getSourceProvenance() instanceof
                TrainTestSplitter.SplitDataSourceProvenance splitterSource) {

            Iterator<Pair<String, Provenance>> sourceProv = splitterSource.iterator();

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

            if(!(innerProvenance instanceof ConfiguredObjectProvenance)){
                // TODO: expand this to be more informative
                throw new Exception("Datasource is not configurable and cannot be recovered.");
            }

            try {
                dataSourceClass = (Class<? extends ConfigurableDataSource<?>>) Class.forName(innerProvenance.getClassName());
            } catch (ClassNotFoundException e) {
                e.printStackTrace();
            }

            // Recreating the trainTestSplitter with the parameters gathered from the provenance, including the
            // innerSource, should return the same Dataset used to train the model
            DataSource<?> innerSource = getDatasourceFromCM(dataSourceClass);
            TrainTestSplitter<?> trainTestSplitter = new TrainTestSplitter<>(innerSource,trainProportion,seed);

            if(isTrain){
                modelDataset = datasetReflection(trainTestSplitter.getTrain());
            } else {
                modelDataset = datasetReflection(trainTestSplitter.getTest());
            }

        } else {
            // When it's not a splitter, recover the Datasource and then Dataset.

            if(!(this.modelProvenance.getDatasetProvenance().getSourceProvenance() instanceof ConfiguredObjectProvenance)){
                // TODO: expand this to be more informative
                throw new Exception("Datasource is not configurable and cannot be recovered.");
            }

            DataSource<?> modelSource = getDatasourceFromCM(dataSourceClass);
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
    public <T extends Output<T>> Model<T> reproduceFromProvenance(){

        // Until now the object only holds the configuration for these objects, the following
        // functions will actually re-instantiate them.
        Trainer<T> newTrainer = recoverTrainer();
        Dataset<T> newDataset = null;
        try {
            newDataset = recoverDataset();
        } catch (Exception e) {

            //TODO Decide what to do when this exception is encountered
            e.printStackTrace();
        }

        // This function actually re-trains a model rather than copy the original
        Model<T> newModel = newTrainer.train(newDataset);

        return newModel;
    }

    /**
     * Using a supplied {@link Model} object, recreates an identical model object that the provenance describes.
     * <p>
     * Recovers the trainer and dataset information before training an identical model.
     * @return A reproduced model identical to the one described in the provenance.
     */
    public <T extends Output<T>> Model<T> reproduceFromModel() throws Exception {
        // TODO: Create types for these exceptions
        if(this.originalModel == null){
            throw new Exception("No model to reproduce was provided");
        }

        Model<T> newModel = reproduceFromProvenance();

        ImmutableFeatureMap newFeatureMap = newModel.getFeatureIDMap();
        ImmutableFeatureMap oldFeatureMap = originalModel.getFeatureIDMap();
        if(newFeatureMap.size() == oldFeatureMap.size()){
            for(int featureIndex = 0; featureIndex < newFeatureMap.size(); featureIndex++){
                if(!newFeatureMap.get(featureIndex).toString().equals(oldFeatureMap.get(featureIndex).toString())){
                    throw new Exception("Recreated model has different feature map");
                }
            }
        }

        // TODO: Expand upon this exception
        if(!originalModel.getOutputIDInfo().toString().equals(newModel.getOutputIDInfo().toString())){
            throw new Exception("Recreated model has different output domain");
        }

        return newModel;
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


        String report = diffProvenanceIterators(originalIter, newIter).toPrettyString();
        return report;
    }

    // This method takes an iterator from a prov object and converts it into a sorted map
    // It uses a sorted map just in case two prov objects are the same type but iterate in a different order.
    private static TreeMap<String, Provenance> iterToMap(Iterator<Pair<String, Provenance>> iter){
        TreeMap<String, Provenance> provMap = new TreeMap<>();
        while (iter.hasNext()){
            Pair<String, Provenance> provPair = iter.next();
            provMap.put(provPair.getA(), provPair.getB());
        }

        return provMap;
    }


    /*
    private static void addProvWithoutDiff(ObjectNode report, Set<String> keys, TreeMap<String, Provenance> provMap, String provIdentifier){
        ObjectMapper mapper = new ObjectMapper();
        for(String key : keys){
            if(provMap.get(key) instanceof PrimitiveProvenance primitiveProvenance){
                ObjectNode provVal = mapper.createObjectNode();
                provVal.put(provIdentifier, primitiveProvenance.getValue().toString());
                report.set(key, provVal);
            } else {
                if (provMap.get(key) instanceof ListProvenance listProvenance){
                    //TODO: Fill out here, is it best to do it this way?
                } else if (provMap.get(key) instanceof Iterable provIterable){
                    ObjectNode subNode = mapper.createObjectNode();
                    Iterator<Pair<String, Provenance>> subIter = provIterable.iterator();

                    TreeMap<String, Provenance> subProvMap = iterToMap(subIter);
                    addSingleProv(subNode, subProvMap.keySet(), subProvMap, provIdentifier);
                    report.set(key, subNode);
                } else {
                    //TODO: Probably throw an exception here
                    System.out.println("Missing type of provenance");
                }

            }
        }
    }
    * */


    private static ObjectNode addProvWithoutDiff(Set<String> keys,  TreeMap<String, Provenance> provMap){
        ObjectMapper mapper = new ObjectMapper();
        ObjectNode provNode = mapper.createObjectNode();

        for (String key : keys) {
            if (provMap.get(key) instanceof PrimitiveProvenance primitiveProvenance) {
                provNode.put(key, primitiveProvenance.getValue().toString());
            } else if (provMap.get(key) instanceof ListProvenance listProvenance) {
                //TODO: Fill out here, is it best to do it this way?
            } else if (provMap.get(key) instanceof Iterable provIterable) {
                Iterator<Pair<String, Provenance>> provIter = provIterable.iterator();
                TreeMap<String, Provenance> subProvMap = iterToMap(provIter);

                provNode.set(key, addProvWithoutDiff(subProvMap.keySet(), subProvMap));
            } else {
                //TODO: Probably throw an exception here
                System.out.println("Missing type of provenance");
            }
        }

        return provNode;
    }

        // This method takes two provenance iterators and returns a diff in the form of a JSON ObjectNode
    private static <T extends Provenance> ObjectNode diffProvenanceIterators(Iterator<Pair<String, Provenance>> iterA, Iterator<Pair<String, Provenance>> iterB){
        // The report ObjectNode is the ultimate return value of the method,
        // All diff values are put in it.
        ObjectMapper mapper = new ObjectMapper();
        ObjectNode report = mapper.createObjectNode();

        // Convert the iterators to a sorted map, so it can iterate through the keys to find matching values
        TreeMap<String, Provenance> provMapA = iterToMap(iterA);
        TreeMap<String, Provenance> provMapB = iterToMap(iterB);

        TreeSet<String> mapAkeys = new TreeSet<>(provMapA.keySet());
        TreeSet<String> mapBkeys = new TreeSet<>(provMapB.keySet());
        TreeSet<String> intersectionOfKeys = new TreeSet<>(provMapA.keySet());

        intersectionOfKeys.retainAll(mapBkeys);

        for(String key : intersectionOfKeys){
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
            // In the event it's a different prov object, and they are the same class, recursively call
            // this method on the iterators of the respective prov objects, and then merge the resulting
            // ObjectNode into the report in this frame.
            else if(provMapA.get(key).getClass() == provMapB.get(key).getClass()){

                if (provMapA.get(key) instanceof ListProvenance){

                    ListProvenance<T> listProvA = ((ListProvenance<T>) provMapA.get(key));
                    ListProvenance<T> listProvB = ((ListProvenance<T>) provMapB.get(key));

                    // If it is a Pair, handle it with recursion as if it was an Object or Map Provenance.
                    if(listProvA.getList().size() > 0 && listProvA.getList().get(0) instanceof Pair){
                        Iterator<Pair<String, Provenance>> subIterA = (Iterator<Pair<String, Provenance>>) ((ListProvenance<?>) provMapA.get(key)).iterator();
                        Iterator<Pair<String, Provenance>> subIterB = (Iterator<Pair<String, Provenance>>) ((ListProvenance<?>) provMapB.get(key)).iterator();

                        ObjectNode sub_report = diffProvenanceIterators(subIterA, subIterB);
                        if(!sub_report.isEmpty()){
                            report.set(key, sub_report);
                        }
                    }
                    // If the list contains ConfiguredObjectProvenanceImpl, they are the same size, and not empty
                    // do an element-wise comparison of the objects and add any diffs to an array that is reported
                    else if (listProvA.getList().size() > 0 &&
                            listProvB.getList().size() > 0 &&
                            listProvA.getList().get(0) instanceof ConfiguredObjectProvenanceImpl &&
                            listProvB.getList().get(0) instanceof ConfiguredObjectProvenanceImpl &&
                            listProvA.getList().size() == listProvB.getList().size()){

                        ArrayNode provArray =  mapper.createArrayNode();

                        // Since each ConfiguredObjectProvenanceImpl will have an iterator, recurse again
                        // Then check if empty and add to array if a diff exists
                        for (int provListIndex = 0; provListIndex < listProvA.getList().size(); provListIndex++){
                            ObjectNode singleProv = diffProvenanceIterators(((ConfiguredObjectProvenanceImpl) listProvA.getList().get(provListIndex)).iterator(),
                                    ((ConfiguredObjectProvenanceImpl) listProvB.getList().get(provListIndex)).iterator());
                            if(!singleProv.isEmpty()){
                                provArray.add(singleProv);
                            }
                        }

                        // After iteration, if diffs exists between the two provenance lists add it to the overall report
                        if(!provArray.isEmpty()){
                            report.set(key, provArray);
                        }
                    }
                }
                // If it's not a ListProvenance it should be an ObjectProvenance or MapProvenance, in which case the
                // iterator will return a Pair<String,Provenance>
                else if (provMapA.get(key) instanceof Iterable provIterableA &&
                         provMapB.get(key) instanceof Iterable provIterableB){

                    // Use the method identified in the previous step to actually get the iterator for each prov object.
                    Iterator<Pair<String, Provenance>> subIterA = provIterableA.iterator();
                    Iterator<Pair<String, Provenance>> subIterB = provIterableB.iterator();

                    // Recursively identify any diffs down to primitive objects
                    ObjectNode sub_report = diffProvenanceIterators(subIterA, subIterB);

                    // Only add the new ObjectNode if it is not empty, to prevent unwanted keys in the resulting JSON
                    if(!sub_report.isEmpty()){
                        report.set(key, sub_report);
                    }

                }
                // ListProvenance might contain Pair<String, Provenance> but it might also contain a list of
                // ConfiguredObjectProvenanceImpl. This handles that situation directly.
                else {
                    //TODO: Error handling here
                    System.out.println("Unrecognized Provenance: ");
                    System.out.println(provMapA.get(key).getClass());
                }
            }
        }

        mapAkeys.removeAll(intersectionOfKeys);
        mapBkeys.removeAll(intersectionOfKeys);

        ObjectNode provNode = addProvWithoutDiff(mapAkeys, provMapA);
        if (!provNode.isEmpty()){
            report.set(OLD, provNode);
        }
        provNode = addProvWithoutDiff(mapBkeys, provMapB);
        if (!provNode.isEmpty()){
            report.set(NEW, provNode);
        }

        return report;
    }

}
