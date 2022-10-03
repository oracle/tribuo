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

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.config.ConfigurationData;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import com.oracle.labs.mlrg.olcut.provenance.MapProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.PrimitiveProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DoubleProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.LongProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ConfigurableDataSource;
import org.tribuo.DataSource;
import org.tribuo.Dataset;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Trainer;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.interop.ExternalTrainerProvenance;
import org.tribuo.interop.onnx.ONNXExternalModel;
import org.tribuo.provenance.DataProvenance;
import org.tribuo.provenance.DataSourceProvenance;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;

import java.lang.reflect.InvocationTargetException;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Reproducibility utility based on Tribuo's provenance objects.
 * <p>
 * Note: this class is designed to be used to reproduce a single object.
 * Repeated calls to {@link #reproduceFromModel()} or {@link #reproduceFromProvenance()}
 * may produce different outputs due to internal state changes.
 * <p>
 * Note: this class's API is experimental and may change in Tribuo minor releases as we work
 * to make it more robust and increase coverage.
 * <p>
 * At the moment the reproducibility system supports {@link ConfigurableDataSource}s and a single level of splitting
 * using a {@link TrainTestSplitter}. It does not support {@link org.tribuo.dataset.DatasetView} which forms the basis
 * of the cross-validation support in Tribuo, nor other datasets which take a subset of their input (e.g.,
 * {@link org.tribuo.dataset.MinimumCardinalityDataset}), we will add this support in future releases.
 * @param <T> The output type of the model being reproduced.
 */
public final class ReproUtil<T extends Output<T>> {
    private static final Logger logger = Logger.getLogger(ReproUtil.class.getName());

    // These fields are used to denote which value came from which provenance in a diff
    private static final String OLD = "original";
    private static final String NEW = "reproduced";

    // Configure the object mapper
    private static final ObjectMapper mapper = new ObjectMapper();
    static {
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
        mapper.enable(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);
    }

    private final ConfigurationManager cm;

    private final ModelProvenance modelProvenance;
    private final Model<T> originalModel;
    private final Class<T> outputClass;

    /**
     * Creates a ReproUtil instance
     * <p>
     * Throws {@link IllegalArgumentException} if the model is an external model trained outside of Tribuo.
     * <p>
     * The output class is validated when the model is reproduced.
     * @param provenance The ReproUtil will re-train a model based on the information contained in this {@link ModelProvenance}.
     * @param outputClass The output class for the model.
     */
    public ReproUtil(ModelProvenance provenance, Class<T> outputClass) {
        this(provenance, null, outputClass);
    }

    /**
     * Creates a ReproUtil instance.
     * <p>
     * Throws {@link IllegalArgumentException} if the model is an external model trained outside of Tribuo.
     * @param originalModel The ReproUtil will re-train a model based on the provenance contained in this {@link Model}.
     */
    public ReproUtil(Model<T> originalModel) {
        this(originalModel.getProvenance(), originalModel, extractOutputClass(originalModel));
    }

    /**
     * Creates a ReproUtil instance.
     * <p>
     * Throws {@link IllegalArgumentException} if the model is an external model trained outside of Tribuo.
     * @param provenance The ReproUtil will re-train a model based on the information contained in this {@link ModelProvenance}.
     * @param originalModel Optional argument, but if it's not null than the provenance used is from this {@link Model}.
     * @param outputClass The output class for the model.
     */
    private ReproUtil(ModelProvenance provenance, Model<T> originalModel, Class<T> outputClass) {
        if (originalModel instanceof ONNXExternalModel<T> onnxModel && onnxModel.getTribuoProvenance().isPresent()) {
            this.modelProvenance = onnxModel.getTribuoProvenance().get();
        } else if (provenance.getTrainerProvenance() instanceof ExternalTrainerProvenance){
            throw new IllegalArgumentException("This version of the tool cannot reproduce external models.");
        } else {
            this.modelProvenance = provenance;
        }

        // Load configurations from provenance so it can re-instantiate objects using the ConfigManager
        // Additionally allows us to change the values of certain configurable fields before re-instantiation
        List<ConfigurationData> provConfig = ProvenanceUtil.extractConfiguration(this.modelProvenance);

        this.cm = new ConfigurationManager();
        this.cm.addConfiguration(provConfig);

        this.originalModel = originalModel;

        this.outputClass = outputClass;
    }

    /**
     * Extracts the output class from the supplied model.
     * @param model The model to extract the output from.
     * @param <T> The output type.
     * @return The output class.
     */
    private static <T extends Output<T>> Class<T> extractOutputClass(Model<T> model) {
        ImmutableOutputInfo<T> outputInfo = model.getOutputIDInfo();
        @SuppressWarnings("unchecked") // the model is typed by this domain so it must be of type <T extends Output<T>>
        Class<T> outputClass = (Class<T>) outputInfo.getDomain().iterator().next().getClass();
        return outputClass;
    }

    /**
     * Extract the trainer from this repro util.
     * <p>
     * Note calling {@link Trainer#train} on the returned trainer object may distort any future reproductions
     * produced by this instance of {@code ReproUtil}.
     * @return A {@link Trainer} found in the configuration manager, used to train the originalModel.
     */
    public Trainer<T> recoverTrainer() {
        // We need to set the state of the RNG for each trainer used in the provenance.
        // ProvenanceUtil.orderProvenances allows us to iterate through the provObjects,
        // construct the correct component name using ProvenanceUtil.computeName, and
        // link a provObject to its corresponding configuration in the cm.
        ProvenanceUtil.ProvenanceOrdering ordering = ProvenanceUtil.orderProvenances(this.modelProvenance);

        for (int i = 0; i < ordering.traversalOrder.size(); i++){
            if(ordering.traversalOrder.get(i) instanceof TrainerProvenance trainerProvenance){
                String componentName = ProvenanceUtil.computeName(trainerProvenance, i);
                Configurable configurableObject = cm.lookup(componentName);
                // Once a Trainer is identified we need to set the invocationCount as identified
                // in the provenance. Invocation count is not configurable since it is a provenance value,
                // it is an immutable value mapping one-to-one to a single execution.
                if(configurableObject instanceof Trainer<?> trainer){
                    //TODO: More instanceof match?
                    trainer.setInvocationCount((int) trainerProvenance.getInstanceValues().get("train-invocation-count").getValue());
                } else {
                    throw new IllegalStateException("Object that is supposed to be a Trainer recovered from Configuration Manager is not a trainer");
                }
            }
        }

        // Recover the name of the trainer class from the model's provenance
        // Convert to a class object so it can be passed to the config manager to recover the trainer object
        try {
            @SuppressWarnings("unchecked")
            Class<? extends Trainer<T>> trainerClass = (Class<? extends Trainer<T>>) Class.forName(this.modelProvenance.getTrainerProvenance().getClassName());
            // Use the class identified from the prov to create a new Trainer from the config manager that
            // is the same as the original trainer used.
            String trainerName = cm.listAll(trainerClass).get(0);
            @SuppressWarnings("unchecked")
            Trainer<T> newTrainer = (Trainer<T>) cm.lookup(trainerName);

            return newTrainer;
        } catch (ClassNotFoundException e) {
            // This exception should not occur since an instance of this class is instantiated in the loop above
            throw new IllegalStateException("Unexpected class cast exception",e);
        }
    }

    /**
     * This method takes a class object and will recover that datasource from the {@link ConfigurationManager}.
     * <p>
     * Throws {@link IllegalStateException} if the data source cannot be instantiated.
     * @param dataSourceClass A configurable class that this method will search for in the configuration manager.
     * @param <D> The type of the data source.
     * @return A {@link DataSource} object that was used to load data from te original model.
     */
    private <D extends ConfigurableDataSource<T>> D getDataSourceFromCM(Class<D> dataSourceClass){

        // Since this utility created a cm from a model, contained within should only be the datasource used to
        // create the model. Unless a user has manually added another datasource.
        List<String> sources = cm.listAll(dataSourceClass);
        String sourceName = null;
        if (sources.size() > 0){
            //TODO: There can be more than one source with AggregateConfigurableDataSource, should potentially
            // search for that and handle it.
            //If configuration data only comes from a model prov there should only be one source
            //but since we expose the configuration manager, a user theoretically could add a new source
            sourceName = sources.get(0);
            if(sources.size() > 1){
                logger.log(Level.INFO, "More than one DataSource found");
            }
        }

        // If the data source can be recovered from cm, override the properties before the dataSource is
        // instantiated, and then instantiate the source.
        if (sourceName == null){
            throw new IllegalStateException("Failed to find source with expected class " + dataSourceClass);
        }

        // Type is restricted by source construction
        D dataSource = dataSourceClass.cast(cm.lookup(sourceName));

        if (!dataSource.getOutputFactory().getUnknownOutput().getClass().equals(outputClass)) {
            throw new IllegalStateException("Supplied output class " + outputClass.getName() + " did not match the data source output class " + dataSource.getOutputFactory().getUnknownOutput().getClass().getName());
        }

        return dataSource;
    }


    /**
     * Attempts to use reflection to determine the correct type a dataset should be and then instantiates it.
     * <p>
     * Throws {@link IllegalStateException} if the dataset could not be instantiated.
     * @param modelSource A datasource class that is passed to the dataset upon construction
     * @param datasetClass The Class of Dataset to be instantiated
     * @return A new {@link Dataset} with modelSource as its {@link DataSource}.
     */
    private Dataset<T> datasetReflection(DataSource<T> modelSource, Class<? extends Dataset<T>> datasetClass){
        // The first step is to find the classname as a String so it can use reflection to instantiate the correct type
        // The class name is contained within the provenance accessible through the iterator.

        // TODO: Determine how to handle other dataset types, ImmutableDataset, DatasetView, and MinimumCardinalityDataset.
        // Once the class is identified, reflection will allow us to search for a constructor for that class and instantiate it
        try {
            return datasetClass.getConstructor(DataSource.class).newInstance(modelSource);
        } catch (InstantiationException | IllegalAccessException | InvocationTargetException | NoSuchMethodException e) {
            throw new IllegalStateException(datasetClass.getName() + " not supported for reproduction", e);
        }
    }

    /**
     * Since {@link Dataset}s can nest inside of {@link Dataset}s, this method recurses through a
     * {@link DatasetProvenance} until it finds the {@link Dataset} that contains a {@link DataSource}
     * and then it returns the classes of the {@link Dataset} and {@link DataSource}.
     * @param datasetProvenance The {@link DatasetProvenance} to search.
     * @return An array of Classes of length two, where the first element is the {@link DataSource} Class and
     * the second element is the {@link Dataset} Class.
     * @throws ClassNotFoundException If the datasource class could not be loaded.
     */
    private Pair<Class<? extends DataSource<T>>,Class<? extends Dataset<T>>> getSourcesClassNames(DatasetProvenance datasetProvenance) throws ClassNotFoundException {
        if (datasetProvenance.getSourceProvenance() instanceof DatasetProvenance innerDatasetProvenance){
            return getSourcesClassNames(innerDatasetProvenance);
        } else {
            Class<?> bareDataClass = Class.forName(datasetProvenance.getSourceProvenance().getClassName());
            if (!DataSource.class.isAssignableFrom(bareDataClass) && !bareDataClass.equals(TrainTestSplitter.class)) {
                throw new IllegalStateException("Unable to instantiate data source, it is not configurable. Found " + bareDataClass);
            }
            // Alternatively we could extract the output factory to guarantee the output class
            @SuppressWarnings("unchecked") // Type guaranteed by provenance and there's a check against the witness in getDataSourceFromCM
            Class<? extends DataSource<T>> dataSourceClass = (Class<? extends DataSource<T>>) bareDataClass;
            Iterator<Pair<String, Provenance>> sourceProv = datasetProvenance.iterator();
            String datasetClassname = null;
            while (sourceProv.hasNext()) {
                Pair<String, Provenance> provPair = sourceProv.next();
                if (provPair.getA().equals("class-name")) {
                    datasetClassname = ((StringProvenance) provPair.getB()).getValue();
                }
            }
            @SuppressWarnings("unchecked") // Type guaranteed by provenance as it's dependent on the source
            Class<? extends Dataset<T>> datasetClass = (Class<? extends Dataset<T>>) Class.forName(datasetClassname);
            return new Pair<>(dataSourceClass, datasetClass);
        }

    }

    /**
     * Since {@link Dataset}s can nest inside of {@link Dataset}s, this method recurses through a
     * {@link DatasetProvenance} until it finds the {@link Dataset} that contains a {@link DataSource}
     * and then it returns the {@link Dataset} and {@link DataSource}.
     * @param datasetProvenance The {@link DatasetProvenance} to search.
     * @return An array of {@link DataProvenance} where the first element is the {@link DataSource} and
     * the second element is the {@link Dataset}.
     */
    private static DataProvenance[] getSources(DatasetProvenance datasetProvenance) {
        DataProvenance[] sourceProvenance = null;
        if(datasetProvenance.getSourceProvenance() instanceof DatasetProvenance innerDatasetProvenance){
            sourceProvenance = getSources(innerDatasetProvenance);
        } else {
            DataSourceProvenance dataSource = null;

            if(datasetProvenance.getSourceProvenance() instanceof DataSourceProvenance dataSourceProvenance){
                dataSource = dataSourceProvenance;
            }
            sourceProvenance = new DataProvenance[] {dataSource, datasetProvenance};
        }

        return sourceProvenance;
    }

    /**
     * Return a {@link Dataset} used when a model was trained.
     * <p>
     * Throws {@link IllegalStateException} if the dataset could not be recovered or one of the classes could not be
     * instantiated.
     * <p>
     * Note transforming or otherwise mutating the returned {@link Dataset} object may distort any future reproductions
     * produced by this instance of {@code ReproUtil}.
     * <p>
     * At the moment this function supports {@link ConfigurableDataSource}s and a single level of splitting using
     * a {@link TrainTestSplitter}. It does not support {@link org.tribuo.dataset.DatasetView} which forms the basis
     * of the cross-validation support in Tribuo, nor other datasets which take a subset of their input (e.g.,
     * {@link org.tribuo.dataset.MinimumCardinalityDataset}), we will add this support in future releases.
     * @return A new {@link Dataset}.
     */
    public Dataset<T> recoverDataset() {
        Provenance[] sourceProvenance = getSources(this.modelProvenance.getDatasetProvenance());
        DataSourceProvenance dataSourceProv = null;
        if(sourceProvenance[0] instanceof DataSourceProvenance dataSourceProvenance){
            dataSourceProv = dataSourceProvenance;
        }

        Class<? extends DataSource<T>> dataSourceClass;
        Class<? extends Dataset<T>> datasetClass;
        try {
            // Get the class names for the dataset and datasource
            var classPair = getSourcesClassNames(this.modelProvenance.getDatasetProvenance());
            dataSourceClass = classPair.getA();
            datasetClass = classPair.getB();
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException("Failed to load datasource or dataset class",e);
        }

        Dataset<T> modelDataset;

        // If the source is a TrainTestSplitter, we need to extract the correct data from the Splitter first.
        // While it is likely they trained on the "train" dataset, in the future they also might have used cross-validation.
        if (dataSourceProv instanceof
                TrainTestSplitter.SplitDataSourceProvenance splitterSource) {

            Iterator<Pair<String, Provenance>> sourceProv = splitterSource.iterator();

            // First collect all the information about the TrainTestSplitter possible including the inner data source
            long seed = 0;
            double trainProportion = 0;
            boolean isTrain = true;
            DataSourceProvenance innerProvenance = null;
            while (sourceProv.hasNext()) {
                Pair<String, Provenance> provPair = sourceProv.next();
                if (provPair.getA().equals("source")) {
                    innerProvenance = (DataSourceProvenance) provPair.getB();
                } else if (provPair.getA().equals("seed")) {
                    seed = ((LongProvenance) provPair.getB()).getValue();
                } else if (provPair.getA().equals("train-proportion")) {
                    trainProportion = ((DoubleProvenance) provPair.getB()).getValue();
                } else if (provPair.getA().equals("is-train")) {
                    isTrain = ((BooleanProvenance) provPair.getB()).getValue();
                }
            }

            if(!(innerProvenance instanceof ConfiguredObjectProvenance)){
                throw new IllegalStateException("Datasource is not configurable and cannot be recovered.");
            }

            try {
                Class<?> bareDataClass = Class.forName(innerProvenance.getClassName());
                if (!ConfigurableDataSource.class.isAssignableFrom(bareDataClass)) {
                    throw new IllegalStateException("Unable to instantiate data source, it is not configurable. Found " + bareDataClass);
                }
                @SuppressWarnings("unchecked") // Guarded by is assignable check
                Class<? extends ConfigurableDataSource<T>> splitterSourceClass = (Class<? extends ConfigurableDataSource<T>>) bareDataClass;
                // Recreating the trainTestSplitter with the parameters gathered from the provenance, including the
                // innerSource, should return the same Dataset used to train the model
                ConfigurableDataSource<T> innerSource = getDataSourceFromCM(splitterSourceClass);
                TrainTestSplitter<T> trainTestSplitter = new TrainTestSplitter<>(innerSource,trainProportion,seed);

                if(isTrain){
                    modelDataset = datasetReflection(trainTestSplitter.getTrain(), datasetClass);
                } else {
                    modelDataset = datasetReflection(trainTestSplitter.getTest(), datasetClass);
                }

            } catch (ClassNotFoundException e) {
                throw new IllegalStateException("Identified DataSource " + innerProvenance.getClassName() + " does not exist", e);
            }

        } else {
            // When it's not a splitter, recover the Datasource and then Dataset.
            if(!(ConfigurableDataSource.class.isAssignableFrom(dataSourceClass))){
                throw new IllegalStateException("Datasource is not configurable and cannot be recovered.");
            }
            @SuppressWarnings("unchecked") // Guarded by is assignable check
            Class<? extends ConfigurableDataSource<T>> configDataSourceClass = (Class<? extends ConfigurableDataSource<T>>) dataSourceClass;
            ConfigurableDataSource<T> modelSource = getDataSourceFromCM(configDataSourceClass);
            modelDataset = datasetReflection(modelSource, datasetClass);
        }

        return modelDataset;
    }

    /**
     * Returns the ConfigurationManager the ReproUtil is using to manage the reproduced models.
     * <p>
     * Note modifying the returned {@code ConfigurationManager} will distort the results of any future reproductions
     * performed by this {@code ReproUtil}.
     * @return a ConfigurationManager the ReproUtil is managing.
     */
    public ConfigurationManager getConfigurationManager(){
        return cm;
    }

    /**
     * Recreates a model object using the {@link ModelProvenance} supplied when the ReproUtil object was created.
     * <p>
     * Recovers the trainer and dataset information before training an identical model.
     * <p>
     * Throws {@link IllegalStateException} if the source or trainer can not be loaded or are not configurable.
     * @return A reproduced model identical to the one described in the provenance.
     * @throws ClassNotFoundException If the dataset or trainer could not be instantiated.
     */
    public Model<T> reproduceFromProvenance() throws ClassNotFoundException {

        // Until now the object only holds the configuration for these objects, the following
        // functions will actually re-instantiate them.
        Trainer<T> newTrainer = recoverTrainer();

        Dataset<T> newDataset = recoverDataset();

        // Exposing the configuration manager means there could be an edge case were
        // the invocation count is changed before the model is trained.
        // Pass through a desired invocation count to prevent this behavior
        // TODO: does not apply to inner trainers, figure out how to address this or if it needs to be addressed
        int trainedInvocationCount = (int) this.modelProvenance
                .getTrainerProvenance()
                .getInstanceValues()
                .get("train-invocation-count")
                .getValue();

        // This function actually re-trains a model rather than copy the original
        return newTrainer.train(newDataset);
    }

    /**
     * Record for any differences between feature sets.
     * @param originalFeatures The unique features from the original model.
     * @param reproducedFeatures The unique features from the reproduced model.
     */
    public record FeatureDiff(Set<String> originalFeatures, Set<String> reproducedFeatures){}

    /**
     * Record for any differences between output domains.
     * @param <T> The type of the output.
     * @param originalOutput The unique output dimensions from the original model.
     * @param reproducedOutput The unique output dimensions from the reproduced model.
     */
    public record OutputDiff<T extends Output<T>>(Set<T> originalOutput, Set<T> reproducedOutput){}

    /**
     * Record for a model reproduction.
     * @param <T> The output type.
     * @param model The reproduced model.
     * @param featureDiff Any differences between the features.
     * @param outputDiff Any differences between the output domain.
     * @param provenanceDiff The provenance diff.
     */
    public record ModelReproduction<T extends Output<T>>(Model<T> model, FeatureDiff featureDiff, OutputDiff<T> outputDiff, String provenanceDiff){}

    /**
     * Using a supplied {@link Model} object, recreates an identical model object that the provenance describes.
     * <p>
     * Recovers the trainer and dataset information before training an identical model.
     * <p>
     * Throws {@link IllegalStateException} if the model provenance is malformed or not defined.
     * @return A reproduced model identical to the one described in the provenance.
     * @throws ClassNotFoundException If the trainer or datasource class cannot be instantiated.
     * @throws JsonProcessingException If the json diff could not be created.
     */
    public ModelReproduction<T> reproduceFromModel() throws ClassNotFoundException, JsonProcessingException {
        if(this.originalModel == null){
            throw new IllegalStateException("No model to reproduce was provided");
        }

        Model<T> newModel = reproduceFromProvenance();

        TreeSet<String> newFeatureKeys = new TreeSet<>(newModel.getFeatureIDMap().keySet());
        TreeSet<String> oldFeatureKeys = new TreeSet<>(originalModel.getFeatureIDMap().keySet());

        TreeSet<String> intersectionOfKeys = new TreeSet<>(newModel.getFeatureIDMap().keySet());
        intersectionOfKeys.retainAll(oldFeatureKeys);
        newFeatureKeys.removeAll(intersectionOfKeys);
        oldFeatureKeys.removeAll(intersectionOfKeys);

        Set<T> oldDomain = originalModel.getOutputIDInfo().generateMutableOutputInfo().getDomain();
        Set<T> newDomain = newModel.getOutputIDInfo().generateMutableOutputInfo().getDomain();
        Set<T> intersectionOfDomains = newModel.getOutputIDInfo().generateMutableOutputInfo().getDomain();
        intersectionOfDomains.retainAll(oldDomain);
        newDomain.removeAll(intersectionOfDomains);
        oldDomain.removeAll(intersectionOfDomains);

        ModelReproduction<T> modelReproduction = new ModelReproduction<>(
                newModel,
                new FeatureDiff(oldFeatureKeys, newFeatureKeys),
                new OutputDiff<>(oldDomain, newDomain),
                ReproUtil.diffProvenance(newModel.getProvenance(), originalModel.getProvenance()));

        return modelReproduction;
    }

    /**
     * Creates a JSON String diff of two {@link ModelProvenance} objects. Only the differences will appear
     * in the resulting diff.
     * <p>
     * Recovers the trainer and dataset information before training an identical model.
     * <p>
     * Throws {@link IllegalStateException} if the model provenances could not be parsed.
     * @param originalProvenance The first of the two provenance objects to diff
     * @param newProvenance The second of the two provenance objects to diff
     * @return A String JSON report displaying the differences in the model.
     * @throws JsonProcessingException If the json diff could not be created.
     */
    public static String diffProvenance(ModelProvenance originalProvenance, ModelProvenance newProvenance) throws JsonProcessingException {

        Iterator<Pair<String, Provenance>> originalIter = originalProvenance.iterator();
        Iterator<Pair<String, Provenance>> newIter = newProvenance.iterator();

        String report = mapper.writeValueAsString(diffProvenanceIterators(originalIter, newIter));
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

    /**
     * This function is used to recurse through prov objects and add them to a report when there is nothing to compare them to.
     * It does not return anything, rather adds values to its first argument.
     * @param report An ObjectNode that this method will add values to.
     * @param keys A set of keys to use to look into provMap, this set is pulled into its own variable
     *             as the keys might be a subset of provMap.keySet().
     * @param provMap A Map of provenance objects
     * @param provIdentifier An identifier indicating how the values should added to the diff,
     *                       either under the ReproUtil.OLD or ReproUtil.NEW designation
     */
    private static void addProvWithoutDiff(ObjectNode report, Set<String> keys, TreeMap<String, Provenance> provMap, String provIdentifier){

        for (String key : keys){
            if(provMap.get(key) instanceof PrimitiveProvenance primitiveProvenance){
                ObjectNode provVal = mapper.createObjectNode();
                provVal.put(provIdentifier, primitiveProvenance.getValue().toString());
                report.set(key, provVal);
            } else {
                // Not all provenance iterators return the same type of iterator,
                // since we cannot pattern match on Iterator<Type>, we need to
                // catch the different iterators based on the type of provenance.
                // ListProvenance can return a list of ConfiguredObjectProvenances
                if (provMap.get(key) instanceof ListProvenance<?> listProvenance){
                    if(listProvenance.getList().size() > 0 && listProvenance.getList().get(0) instanceof ConfiguredObjectProvenance){
                        ArrayNode provArray =  mapper.createArrayNode();

                        // Iterate through each ConfiguredObjectProvenance and recurse on each provenance object
                        for (int provListIndex = 0; provListIndex < listProvenance.getList().size(); provListIndex++){
                            TreeMap<String, Provenance> subProvMap = iterToMap(((ConfiguredObjectProvenance) listProvenance.getList().get(provListIndex)).iterator());
                            ObjectNode subNode = mapper.createObjectNode();
                            addProvWithoutDiff(subNode, subProvMap.keySet(), subProvMap, provIdentifier);
                            if(!subNode.isEmpty()){
                                provArray.add(subNode);
                            }
                        }

                        if(!(provArray.isEmpty())){
                            report.set(key, provArray);
                        }
                    }
                } else if (provMap.get(key) instanceof Iterable<?> provIterable){
                    ObjectNode subNode = mapper.createObjectNode();
                    @SuppressWarnings("unchecked") // Provenance is sealed to map, object and list. map and object are both Iterable<Pair<String,Provenance>>
                    Iterator<Pair<String, Provenance>> subIter = (Iterator<Pair<String,Provenance>>) provIterable.iterator();

                    TreeMap<String, Provenance> subProvMap = iterToMap(subIter);
                    addProvWithoutDiff(subNode, subProvMap.keySet(), subProvMap, provIdentifier);
                    report.set(key, subNode);
                } else {
                    throw new IllegalStateException("Unknown type of provenance: " + provMap.get(key).toString());
                }

            }
        }
    }

    //

    /**
     * This method takes two provenance iterators and returns a diff in the form of a JSON ObjectNode
     * <p>
     * Throws {@link IllegalStateException} if the provenance could not be compared.
     * @param iterA An iterator from a provenance object
     * @param iterB An iterator from a provenance object
     * @return An ObjectNode that will be printed to string in diffProvenance.
     */
    private static ObjectNode diffProvenanceIterators(Iterator<Pair<String, Provenance>> iterA, Iterator<Pair<String, Provenance>> iterB){
        // The report ObjectNode is the ultimate return value of the method,
        // All diff values are put in it.
        ObjectNode report = mapper.createObjectNode();

        // Convert the iterators to a sorted map, so it can iterate through the keys to find matching values
        TreeMap<String, Provenance> provMapA = iterToMap(iterA);
        TreeMap<String, Provenance> provMapB = iterToMap(iterB);

        // Get the keys of the maps and perform an intersection to find all provenance that exists in both models
        TreeSet<String> mapAkeys = new TreeSet<>(provMapA.keySet());
        TreeSet<String> mapBkeys = new TreeSet<>(provMapB.keySet());
        TreeSet<String> intersectionOfKeys = new TreeSet<>(provMapA.keySet());

        intersectionOfKeys.retainAll(mapBkeys);

        // Loop through all of the keys, and when a difference in primitive provenances is found, record both values
        for(String key : intersectionOfKeys){
            Provenance provenanceA = provMapA.get(key);
            Provenance provenanceB = provMapB.get(key);

            // When provenance is primitive and they share a key simply compare their values
            // If different, place each prov's value in the report
            if(provenanceA instanceof PrimitiveProvenance primProvA && provenanceB instanceof PrimitiveProvenance primProvB){
                if(!primProvA.equals(primProvB)){
                    ObjectNode valDiff = mapper.createObjectNode();
                    valDiff.put(OLD, primProvA.getValue().toString());
                    valDiff.put(NEW, primProvB.getValue().toString());
                    report.set(key, valDiff);
                }
            } else if(provenanceA.getClass() == provenanceB.getClass()){
                // In the event it's a different prov object, and they are the same class, recursively call
                // this method on the iterators of the respective prov objects, and then merge the resulting
                // ObjectNode into the report in this frame.

                // When prov is a ListProvenance, we have to make sure we get the right iterator,
                // as ListProvenances cannot recurse normally as they hold prov objects rather
                // than something with "keys"
                if (provenanceA instanceof ListProvenance<?> listProvA){
                    ListProvenance<?> listProvB = ((ListProvenance<?>) provenanceB);
                    List<? extends Provenance> listA = listProvA.getList();
                    List<? extends Provenance> listB = listProvB.getList();

                    ArrayNode provArray =  mapper.createArrayNode();
                    // If both lists are not empty
                    // do an element-wise comparison of the objects and add any diffs to an array that is reported
                    // TODO: add explanation suffix
                    if (listA.size() > 0 && listB.size() > 0) {
                        if (listA.get(0) instanceof ObjectProvenance && listB.get(0) instanceof ObjectProvenance){

                            int length = Math.min(listA.size(), listB.size());
                            // Since each ConfiguredObjectProvenanceImpl will have an iterator, recurse again
                            // Then check if empty and add to array if a diff exists
                            for (int provListIndex = 0; provListIndex < length; provListIndex++){
                                ObjectNode singleProv = diffProvenanceIterators(((ObjectProvenance) listA.get(provListIndex)).iterator(),
                                        ((ObjectProvenance) listB.get(provListIndex)).iterator());
                                if(!singleProv.isEmpty()){
                                    provArray.add(singleProv);
                                }
                            }

                            // Add unmatched elements of A if A is longer
                            for (int i = length; i < listA.size(); i++) {
                                ObjectNode singleProv = diffProvenanceIterators(((ObjectProvenance) listA.get(i)).iterator(), Collections.emptyIterator());
                                if(!singleProv.isEmpty()){
                                    provArray.add(singleProv);
                                }
                            }

                            // Add unmatched elements of B if B is longer
                            for (int i = length; i < listB.size(); i++) {
                                ObjectNode singleProv = diffProvenanceIterators(Collections.emptyIterator(), ((ObjectProvenance) listB.get(i)).iterator());
                                if(!singleProv.isEmpty()){
                                    provArray.add(singleProv);
                                }
                            }
                        } else {
                            // List of maps, list of lists, or list of primitives
                        }
                    } else if (listA.size() > 0) {
                        if (listA.get(0) instanceof ObjectProvenance) {
                            for (Provenance provenance : listA) {
                                ObjectNode singleProv = diffProvenanceIterators(((ObjectProvenance) provenance).iterator(), Collections.emptyIterator());
                                if (!singleProv.isEmpty()) {
                                    provArray.add(singleProv);
                                }
                            }
                        } else if (listA.get(0) instanceof MapProvenance) {
                            for (Provenance provenance : listA) {
                                @SuppressWarnings("unchecked") // wildcard casting issues if we use ? extends Provenance
                                ObjectNode singleProv = diffProvenanceIterators(((MapProvenance<Provenance>) provenance).iterator(), Collections.emptyIterator());
                                if (!singleProv.isEmpty()) {
                                    provArray.add(singleProv);
                                }
                            }
                        } else if (listA.get(0) instanceof PrimitiveProvenance) {
                            for (Provenance provenance : listA) {
                                ObjectNode valDiff = mapper.createObjectNode();
                                valDiff.put(OLD, ((PrimitiveProvenance<?>)provenance).getValue().toString());
                                provArray.add(valDiff);
                            }
                        } else if (listA.get(0) instanceof ListProvenance) {
                            throw new IllegalStateException("Nested lists not supported at key " + key);
                        }
                    } else if (listB.size() > 0) {
                        if (listB.get(0) instanceof ObjectProvenance) {
                            for (Provenance provenance : listB) {
                                ObjectNode singleProv = diffProvenanceIterators(Collections.emptyIterator(), ((ObjectProvenance) provenance).iterator());
                                if (!singleProv.isEmpty()) {
                                    provArray.add(singleProv);
                                }
                            }
                        } else if (listB.get(0) instanceof MapProvenance) {
                            for (Provenance provenance : listB) {
                                @SuppressWarnings("unchecked") // wildcard casting issues if we use ? extends Provenance
                                ObjectNode singleProv = diffProvenanceIterators(Collections.emptyIterator(), ((MapProvenance<Provenance>) provenance).iterator());
                                if (!singleProv.isEmpty()) {
                                    provArray.add(singleProv);
                                }
                            }
                        } else if (listB.get(0) instanceof PrimitiveProvenance) {
                            for (Provenance provenance : listB) {
                                ObjectNode valDiff = mapper.createObjectNode();
                                valDiff.put(NEW, ((PrimitiveProvenance<?>)provenance).getValue().toString());
                                provArray.add(valDiff);
                            }
                        } else if (listB.get(0) instanceof ListProvenance) {
                            throw new IllegalStateException("Nested lists not supported at key " + key);
                        }
                    }

                    // After iteration, if diffs exists between the two provenance lists add it to the overall report
                    if(!provArray.isEmpty()){
                        report.set(key, provArray);
                    }
                } else if (provenanceA instanceof Iterable provIterableA &&
                         provenanceB instanceof Iterable provIterableB){
                    // If it's not a ListProvenance it should be an ObjectProvenance or MapProvenance, in which case the
                    // iterator will return a Pair<String,Provenance>
                    //TODO: Can we provide a better enforcement of this assumption? Yes, once OLCUT is on Java 17

                    if(!(provenanceA instanceof ObjectProvenance || provenanceA instanceof MapProvenance)){
                        throw new IllegalStateException("Unrecognized provenance type");
                    }
                    if(!(provenanceB instanceof ObjectProvenance || provenanceB instanceof MapProvenance)){
                        throw new IllegalStateException("Unrecognized provenance type");
                    }

                    // Use the method identified in the previous step to actually get the iterator for each prov object.
                    @SuppressWarnings("unchecked") // only object and map provenances should reach here
                    Iterator<Pair<String, Provenance>> subIterA = provIterableA.iterator();
                    @SuppressWarnings("unchecked") // only object and map provenances should reach here
                    Iterator<Pair<String, Provenance>> subIterB = provIterableB.iterator();

                    // Recursively identify any diffs down to primitive objects
                    ObjectNode subReport = diffProvenanceIterators(subIterA, subIterB);

                    // Only add the new ObjectNode if it is not empty, to prevent unwanted keys in the resulting JSON
                    if(!subReport.isEmpty()){
                        report.set(key, subReport);
                    }

                } else {
                    // ListProvenance might contain Pair<String, Provenance> but it might also contain a list of
                    // ConfiguredObjectProvenanceImpl. This handles that situation directly.
                    throw new IllegalStateException("Unrecognized Provenance: \n" + provMapA.get(key).getClass());
                }
            } else {
                // TODO: Recurse diff for provenances of different types
                //throw new IllegalStateException("Provenances with the same name have different types, key = " + key + ", provA " + provenanceA.getClass() + ", provB " + provenanceB.getClass());
            }
        }

        mapAkeys.removeAll(intersectionOfKeys);
        mapBkeys.removeAll(intersectionOfKeys);

        addProvWithoutDiff(report, mapAkeys, provMapA, OLD);
        addProvWithoutDiff(report, mapBkeys, provMapB, NEW);

        return report;
    }

}
