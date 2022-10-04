/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.common.xgboost;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Trainer;
import org.tribuo.WeightedExamples;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.SkeletalTrainerProvenance;
import org.tribuo.util.Util;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.logging.Logger;

/**
 * A {@link Trainer} which wraps the XGBoost training procedure.
 * <p>
 * This only exposes a few of XGBoost's training parameters.
 * <p>
 * It uses pthreads outside of the JVM to parallelise the computation.
 * <p>
 * See:
 * <pre>
 * Chen T, Guestrin C.
 * "XGBoost: A Scalable Tree Boosting System"
 * Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016.
 * </pre>
 * and for the original algorithm:
 * <pre>
 * Friedman JH.
 * "Greedy Function Approximation: a Gradient Boosting Machine"
 * Annals of statistics, 2001.
 * </pre>
 * N.B.: XGBoost4J wraps the native C implementation of xgboost that links to various C libraries, including libgomp
 * and glibc (on Linux). If you're running on Alpine, which does not natively use glibc, you'll need to install glibc
 * into the container.
 * On the macOS binary on Maven Central is compiled without
 * OpenMP support, meaning that XGBoost is single threaded on macOS. You can recompile the macOS binary with
 * OpenMP support after installing libomp from homebrew if necessary.
 */
public abstract class XGBoostTrainer<T extends Output<T>> implements Trainer<T>, WeightedExamples {
    /* Alpine install command
     * <pre>
     *    $ apk --no-cache add ca-certificates wget
     *    $ wget -q -O /etc/apk/keys/sgerrand.rsa.pub https://alpine-pkgs.sgerrand.com/sgerrand.rsa.pub
     *    $ wget https://github.com/sgerrand/alpine-pkg-glibc/releases/download/2.30-r0/glibc-2.30-r0.apk
     *    $ apk add glibc-2.30-r0.apk
     * </pre>
     */

    private static final Logger logger = Logger.getLogger(XGBoostTrainer.class.getName());

    /**
     * The tree building algorithm.
     */
    public enum TreeMethod {
        /**
         * XGBoost chooses between {@link TreeMethod#EXACT} and {@link TreeMethod#APPROX}
         * depending on dataset size.
         */
        AUTO("auto"),
        /**
         * Exact greedy algorithm, enumerates all split candidates.
         */
        EXACT("exact"),
        /**
         * Approximate greedy algorithm, using a quantile sketch of the data and a gradient histogram.
         */
        APPROX("approx"),
        /**
         * Faster histogram optimized approximate algorithm.
         */
        HIST("hist"),
        /**
         * GPU implementation of the {@link TreeMethod#HIST} algorithm.
         * <p>
         * Note: GPU computation may not be supported on all platforms, and Tribuo is not tested with XGBoost GPU support.
         */
        GPU_HIST("gpu_hist");

        /**
         * The parameter name used by the XGBoost native library.
         */
        public final String paramName;

        TreeMethod(String paramName) {
            this.paramName = paramName;
        }
    }

    /**
     * The logging verbosity of the native library.
     */
    public enum LoggingVerbosity {
        /**
         * No logging.
         */
        SILENT(0),
        /**
         * Only warnings are logged.
         */
        WARNING(1),
        /**
         * Tree building info is logged as well as warnings.
         */
        INFO(2),
        /**
         * All the logging.
         */
        // You've never seen so much logging.
        DEBUG(3);

        /**
         * The log value used by the XGBoost native library.
         */
        public final int value;

        LoggingVerbosity(int value) {
            this.value = value;
        }
    }

    /**
     * The type of XGBoost model.
     */
    public enum BoosterType {
        /**
         * A boosted linear model.
         */
        LINEAR("gblinear"),
        /**
         * A gradient boosted decision tree.
         */
        GBTREE("gbtree"),
        /**
         * A gradient boosted decision tree using dropout.
         */
        DART("dart");

        /**
         * The parameter value used by the XGBoost native library.
         */
        public final String paramName;

        BoosterType(String paramName) {
            this.paramName = paramName;
        }
    }

    /**
     * The XGBoost parameter map, only accessed internally.
     */
    protected final Map<String, Object> parameters = new HashMap<>();

    /**
     * Override for the parameter map, must contain all parameters, including the objective function.
     */
    @Config(description = "Override for parameters, if used must contain all the relevant parameters, including the objective")
    protected Map<String, String> overrideParameters = new HashMap<>();

    /**
     * The number of trees to build.
     */
    @Config(mandatory = true,description="The number of trees to build.")
    protected int numTrees;

    /**
     * The learning rate.
     */
    @Config(description = "The learning rate, shrinks the new tree output to prevent overfitting.")
    private double eta = 0.3;

    /**
     * Minimum loss reduction to split a node.
     */
    @Config(description = "Minimum loss reduction needed to split a tree node.")
    private double gamma = 0.0;

    /**
     * Max tree depth.
     */
    @Config(description="The maximum depth of any tree.")
    private int maxDepth = 6;

    /**
     * Minimum weight in each child node before the split is valid.
     */
    @Config(description = "The minimum weight in each child node before a split is valid.")
    private double minChildWeight = 1.0;

    /**
     * Subsample the examples (i.e., bagging).
     */
    @Config(description="Independently subsample the examples for each tree.")
    private double subsample = 1.0;

    /**
     * Subsample the columns.
     */
    @Config(description="Independently subsample the features available for each node of each tree.")
    private double featureSubsample = 1.0;

    /**
     * L2 regularisation term
     */
    @Config(description="l2 regularisation term on the weights.")
    private double lambda = 1.0;

    /**
     * L1 regularisation term.
     */
    @Config(description="l1 regularisation term on the weights.")
    private double alpha = 1.0;

    /**
     * Number of training threads.
     */
    @Config(description="The number of threads to use at training time.")
    private int nThread = 4;

    /**
     * Deprecated by XGBoost in favour of the verbosity field.
     */
    @Deprecated
    @Config(description="Quiesce all the logging output from the XGBoost C library. Deprecated in favour of 'verbosity'.")
    private int silent = 1;

    @Config(description="Logging verbosity, 0 is silent, 3 is debug.")
    private LoggingVerbosity verbosity = LoggingVerbosity.SILENT;

    @Config(description="Type of the weak learner.")
    private BoosterType booster = BoosterType.GBTREE;

    @Config(description="The tree building algorithm to use.")
    private TreeMethod treeMethod = TreeMethod.AUTO;

    @Config(description="The RNG seed.")
    private long seed = Trainer.DEFAULT_SEED;

    /**
     * Number of times the {@code train} method has been called on this object.
     */
    protected int trainInvocationCounter = 0;

    /**
     * Constructs an XGBoost trainer using the specified number of trees.
     * @param numTrees The number of trees.
     */
    protected XGBoostTrainer(int numTrees) {
        this(numTrees, 0.3, 0, 6, 1, 1, 1, 1, 0, 4, true, Trainer.DEFAULT_SEED);
    }

    /**
     * Constructs an XGBoost trainer using the specified number of trees.
     * @param numTrees The number of trees.
     * @param numThreads The number of training threads.
     * @param silent Should the logging be silenced?
     */
    protected XGBoostTrainer(int numTrees, int numThreads, boolean silent) {
        this(numTrees, 0.3, 0, 6, 1, 1, 1, 1, 0, numThreads, silent, Trainer.DEFAULT_SEED);
    }

    /**
     * Create an XGBoost trainer.
     * <p>
     * Sets the boosting algorithm to {@link BoosterType#GBTREE} and the tree building algorithm to {@link TreeMethod#AUTO}.
     *
     * @param numTrees Number of trees to boost.
     * @param eta Step size shrinkage parameter (default 0.3, range [0,1]).
     * @param gamma Minimum loss reduction to make a split (default 0, range
     * [0,inf]).
     * @param maxDepth Maximum tree depth (default 6, range [1,inf]).
     * @param minChildWeight Minimum sum of instance weights needed in a leaf
     * (default 1, range [0, inf]).
     * @param subsample Subsample size for each tree (default 1, range (0,1]).
     * @param featureSubsample Subsample features for each tree (default 1,
     * range (0,1]).
     * @param lambda L2 regularization term on weights (default 1).
     * @param alpha L1 regularization term on weights (default 0).
     * @param nThread Number of threads to use (default 4).
     * @param silent Silence the training output text.
     * @param seed RNG seed.
     */
    protected XGBoostTrainer(int numTrees, double eta, double gamma, int maxDepth, double minChildWeight, double subsample, double featureSubsample, double lambda, double alpha, int nThread, boolean silent, long seed) {
        this(BoosterType.GBTREE,TreeMethod.AUTO,numTrees,eta,gamma,maxDepth,minChildWeight,subsample,featureSubsample,lambda,alpha,nThread,silent ? LoggingVerbosity.SILENT : LoggingVerbosity.INFO,seed);
    }

    /**
     * Create an XGBoost trainer.
     *
     * @param boosterType The base learning algorithm.
     * @param treeMethod The tree building algorithm if using a tree booster.
     * @param numTrees Number of trees to boost.
     * @param eta Step size shrinkage parameter (default 0.3, range [0,1]).
     * @param gamma Minimum loss reduction to make a split (default 0, range
     * [0,inf]).
     * @param maxDepth Maximum tree depth (default 6, range [1,inf]).
     * @param minChildWeight Minimum sum of instance weights needed in a leaf
     * (default 1, range [0, inf]).
     * @param subsample Subsample size for each tree (default 1, range (0,1]).
     * @param featureSubsample Subsample features for each tree (default 1,
     * range (0,1]).
     * @param lambda L2 regularization term on weights (default 1).
     * @param alpha L1 regularization term on weights (default 0).
     * @param nThread Number of threads to use (default 4).
     * @param verbosity Set the logging verbosity of the native library.
     * @param seed RNG seed.
     */
    protected XGBoostTrainer(BoosterType boosterType, TreeMethod treeMethod, int numTrees, double eta, double gamma, int maxDepth, double minChildWeight, double subsample, double featureSubsample, double lambda, double alpha, int nThread, LoggingVerbosity verbosity, long seed) {
        if (numTrees < 1) {
            throw new IllegalArgumentException("Must supply a positive number of trees. Received " + numTrees);
        }
        this.booster = boosterType;
        this.treeMethod = treeMethod;
        this.numTrees = numTrees;
        this.eta = eta;
        this.gamma = gamma;
        this.maxDepth = maxDepth;
        this.minChildWeight = minChildWeight;
        this.subsample = subsample;
        this.featureSubsample = featureSubsample;
        this.lambda = lambda;
        this.alpha = alpha;
        this.nThread = nThread;
        this.verbosity = verbosity;
        this.silent = 0; // silent is deprecated
        this.seed = seed;
    }

    /**
     * This gives direct access to the XGBoost parameter map.
     * <p>
     * It lets you pick things that we haven't exposed like dropout trees, binary classification etc.
     * <p>
     * This sidesteps the validation that Tribuo provides for the hyperparameters, and so can produce unexpected results.
     * @param numTrees Number of trees to boost.
     * @param parameters A map from string to object, where object can be Number or String.
     */
    protected XGBoostTrainer(int numTrees, Map<String,Object> parameters) {
        if (numTrees < 1) {
            throw new IllegalArgumentException("Must supply a positive number of trees. Received " + numTrees);
        }
        this.numTrees = numTrees;
        for (Map.Entry<String,Object> e : parameters.entrySet()) {
            this.overrideParameters.put(e.getKey(),e.getValue().toString());
        }
    }

    /**
     * For olcut.
     */
    protected XGBoostTrainer() { }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        parameters.put("eta", eta);
        parameters.put("gamma", gamma);
        parameters.put("max_depth", maxDepth);
        parameters.put("min_child_weight", minChildWeight);
        parameters.put("subsample", subsample);
        parameters.put("colsample_bytree", featureSubsample);
        parameters.put("lambda", lambda);
        parameters.put("alpha", alpha);
        parameters.put("nthread", nThread);
        parameters.put("seed", seed);
        if (silent == 1) {
            parameters.put("verbosity", 0);
        } else {
            parameters.put("verbosity", verbosity.value);
        }
        parameters.put("booster", booster.paramName);
        parameters.put("tree_method", treeMethod.paramName);
        if (!overrideParameters.isEmpty() && !overrideParameters.containsKey("objective")) {
            throw new PropertyException("","overrideParameters","When using the override parameters must supply an objective");
        }
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder();

        buffer.append("XGBoostTrainer(numTrees=");
        buffer.append(numTrees);
        buffer.append(",parameters");
        buffer.append(parameters.toString());
        buffer.append(")");

        return buffer.toString();
    }

    /**
     * Creates an XGBoost model from the booster list.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     * @param models The boosters.
     * @param converter The converter from XGBoost's output to Tribuo predictions.
     * @return An XGBoost model.
     */
    protected XGBoostModel<T> createModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo, List<Booster> models, XGBoostOutputConverter<T> converter) {
        return new XGBoostModel<>(name,provenance,featureIDMap,outputIDInfo,models,converter);
    }

    /**
     * Returns a copy of the supplied parameter map which
     * has the appropriate type for passing to XGBoost.train.
     * @param input The parameter map.
     * @return A (shallow) copy of the supplied map.
     */
    protected Map<String,Object> copyParams(Map<String, ?> input) {
        return new HashMap<>(input);
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public synchronized void setInvocationCount(int invocationCount) {
        if(invocationCount < 0){
            throw new IllegalArgumentException("The supplied invocationCount is less than zero.");
        }

        this.trainInvocationCounter = invocationCount;
    }

    /**
     * Converts a dataset into a DMatrix.
     * @param examples The examples to convert.
     * @param responseExtractor The extraction function for the output.
     * @param <T> The type of the output.
     * @return A DMatrixTuple.
     * @throws XGBoostError If the native library failed to construct the DMatrix.
     */
    protected static <T extends Output<T>> DMatrixTuple<T> convertDataset(Dataset<T> examples, Function<T,Float> responseExtractor) throws XGBoostError {
        return convertExamples(examples.getData(), examples.getFeatureIDMap(), responseExtractor);
    }

    /**
     * Converts a dataset into a DMatrix.
     * @param examples The examples to convert.
     * @param <T> The type of the output.
     * @return A DMatrixTuple.
     * @throws XGBoostError If the native library failed to construct the DMatrix.
     */
    protected static <T extends Output<T>> DMatrixTuple<T> convertDataset(Dataset<T> examples) throws XGBoostError {
        return convertExamples(examples.getData(), examples.getFeatureIDMap(), null);
    }

    /**
     * Converts an iterable of examples into a DMatrix.
     * @param examples The examples to convert.
     * @param featureMap The feature id map which supplies the indices.
     * @param <T> The type of the output.
     * @return A DMatrixTuple.
     * @throws XGBoostError If the native library failed to construct the DMatrix.
     */
    protected static <T extends Output<T>> DMatrixTuple<T> convertExamples(Iterable<Example<T>> examples, ImmutableFeatureMap featureMap) throws XGBoostError {
        return convertExamples(examples, featureMap, null);
    }

    /**
     * Converts an iterable of examples into a DMatrix.
     * @param examples The examples to convert.
     * @param featureMap The feature id map which supplies the indices.
     * @param responseExtractor The extraction function for the output.
     * @param <T> The type of the output.
     * @return A DMatrixTuple.
     * @throws XGBoostError If the native library failed to construct the DMatrix.
     */
    protected static <T extends Output<T>> DMatrixTuple<T> convertExamples(Iterable<Example<T>> examples, ImmutableFeatureMap featureMap, Function<T,Float> responseExtractor) throws XGBoostError {
        // headers = array of start points for a row
        // indices = array of feature indices for all data
        // data = array of feature values for all data
        // SparseType = DMatrix.SparseType.CSR
        //public DMatrix(long[] headers, int[] indices, float[] data, SparseType st) throws XGBoostError
        //
        // then call
        //public void setLabel(float[] labels) throws XGBoostError

        boolean labelled = responseExtractor != null;
        ArrayList<Float> labelsList = new ArrayList<>();
        ArrayList<Float> dataList = new ArrayList<>();
        ArrayList<Long> headersList = new ArrayList<>();
        ArrayList<Integer> indicesList = new ArrayList<>();
        ArrayList<Float> weightsList = new ArrayList<>();
        ArrayList<Integer> numValidFeatures = new ArrayList<>();
        ArrayList<Example<T>> examplesList = new ArrayList<>();

        long rowHeader = 0;
        headersList.add(rowHeader);
        for (Example<T> e : examples) {
            if (labelled) {
                labelsList.add(responseExtractor.apply(e.getOutput()));
                weightsList.add(e.getWeight());
            }
            examplesList.add(e);
            long newRowHeader = convertSingleExample(e,featureMap,dataList,indicesList,headersList,rowHeader);
            numValidFeatures.add((int) (newRowHeader-rowHeader));
            rowHeader = newRowHeader;
        }

        float[] data = Util.toPrimitiveFloat(dataList);
        int[] indices = Util.toPrimitiveInt(indicesList);
        long[] headers = Util.toPrimitiveLong(headersList);

        DMatrix dataMatrix = new DMatrix(headers, indices, data, DMatrix.SparseType.CSR,featureMap.size());
        if (labelled) {
            float[] labels = Util.toPrimitiveFloat(labelsList);
            dataMatrix.setLabel(labels);
            float[] weights = Util.toPrimitiveFloat(weightsList);
            dataMatrix.setWeight(weights);
        }
        @SuppressWarnings("unchecked") // Generic array creation
        Example<T>[] exampleArray = (Example<T>[])examplesList.toArray(new Example[0]);
        return new DMatrixTuple<>(dataMatrix,Util.toPrimitiveInt(numValidFeatures),exampleArray);
    }

    /**
     * Converts an example into a DMatrix.
     * @param example The example to convert.
     * @param featureMap The feature id map which supplies the indices.
     * @param <T> The type of the output.
     * @return A DMatrixTuple.
     * @throws XGBoostError If the native library failed to construct the DMatrix.
     */
    protected static <T extends Output<T>> DMatrixTuple<T> convertExample(Example<T> example, ImmutableFeatureMap featureMap) throws XGBoostError {
        return convertExample(example,featureMap,null);
    }

    /**
     * Converts an example into a DMatrix.
     * @param example The example to convert.
     * @param featureMap The feature id map which supplies the indices.
     * @param responseExtractor The extraction function for the output.
     * @param <T> The type of the output.
     * @return A DMatrixTuple.
     * @throws XGBoostError If the native library failed to construct the DMatrix.
     */
    protected static <T extends Output<T>> DMatrixTuple<T> convertExample(Example<T> example, ImmutableFeatureMap featureMap, Function<T,Float> responseExtractor) throws XGBoostError {
        // headers = array of start points for a row
        // indices = array of feature indices for all data
        // data = array of feature values for all data
        // SparseType = DMatrix.SparseType.CSR
        //public DMatrix(long[] headers, int[] indices, float[] data, SparseType st) throws XGBoostError
        //
        // then call
        //public void setLabel(float[] labels) throws XGBoostError

        boolean labelled = responseExtractor != null;
        ArrayList<Float> dataList = new ArrayList<>();
        ArrayList<Integer> indicesList = new ArrayList<>();
        ArrayList<Long> headersList = new ArrayList<>();
        headersList.add(0L);

        long header = convertSingleExample(example,featureMap,dataList,indicesList,headersList,0);

        float[] data = Util.toPrimitiveFloat(dataList);
        int[] indices = Util.toPrimitiveInt(indicesList);
        long[] headers = Util.toPrimitiveLong(headersList);

        DMatrix dataMatrix = new DMatrix(headers, indices, data, DMatrix.SparseType.CSR,featureMap.size());
        if (labelled) {
            float[] labels = new float[1];
            labels[0] = responseExtractor.apply(example.getOutput());
            dataMatrix.setLabel(labels);
            float[] weights = new float[1];
            weights[0] = example.getWeight();
            dataMatrix.setWeight(weights);
        }
        @SuppressWarnings("unchecked") // Generic array creation
        Example<T>[] exampleArray = (Example<T>[])new Example[]{example};
        return new DMatrixTuple<>(dataMatrix,new int[]{(int)header},exampleArray);
    }

    /**
     * Writes out the features from an example into the three supplied {@link ArrayList}s.
     * <p>
     * This is used to transform examples into the right format for an XGBoost call.
     * It's used in both the Classification and Regression XGBoost backends.
     * The ArrayLists must be non-null, and can contain existing values (as this
     * method is called multiple times to build up an arraylist containing all the
     * feature values for a dataset).
     * <p>
     * Features with colliding feature ids are summed together.
     * <p>
     * Can throw IllegalArgumentException if the {@link Example} contains no features.
     * @param example The example to inspect.
     * @param featureMap The feature map of the model/dataset (used to preserve hash information).
     * @param dataList The output feature values.
     * @param indicesList The output indices.
     * @param headersList The output header position (an integer saying how long each sparse example is).
     * @param header The current header position.
     * @param <T> The type of the example.
     * @return The updated header position.
     */
    protected static <T extends Output<T>> long convertSingleExample(Example<T> example, ImmutableFeatureMap featureMap, ArrayList<Float> dataList, ArrayList<Integer> indicesList, ArrayList<Long> headersList, long header) {
        int numActiveFeatures = 0;
        int prevIdx = -1;
        int indicesSize = indicesList.size();
        for (Feature f : example) {
            int id = featureMap.getID(f.getName());
            if (id > prevIdx){
                prevIdx = id;
                dataList.add((float) f.getValue());
                indicesList.add(id);
                numActiveFeatures++;
            } else if (id > -1) {
                //
                // Collision, deal with it.
                int collisionIdx = Util.binarySearch(indicesList,id,indicesSize,numActiveFeatures+indicesSize);
                if (collisionIdx < 0) {
                    //
                    // Collision but not present in tmpIndices
                    // move data and bump i
                    collisionIdx = - (collisionIdx + 1);
                    indicesList.add(collisionIdx,id);
                    dataList.add(collisionIdx,(float) f.getValue());
                    numActiveFeatures++;
                } else {
                    //
                    // Collision present in tmpIndices
                    // add the values.
                    dataList.set(collisionIdx, dataList.get(collisionIdx) + (float) f.getValue());
                }
            }
        }
        if (numActiveFeatures == 0) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }
        header += numActiveFeatures;
        headersList.add(header);
        return header;
    }

    /**
     * Writes out the features from a SparseVector into the three supplied {@link ArrayList}s.
     * <p>
     * This is used to transform examples into the right format for an XGBoost call.
     * It's used when predicting with an externally trained XGBoost model, as the
     * external training may not respect Tribuo's feature ordering constraints.
     * The ArrayLists must be non-null, and can contain existing values (as this
     * method is called multiple times to build up an arraylist containing all the
     * feature values for a dataset).
     * </p>
     * <p>
     * This is much simpler than {@link XGBoostTrainer#convertSingleExample} as the validation
     * of feature indices is done in the {@link org.tribuo.interop.ExternalModel} class.
     * </p>
     * @param vector The features to convert.
     * @param dataList The output feature values.
     * @param indicesList The output indices.
     * @param headersList The output header position (an integer saying how long each sparse example is).
     * @param header The current header position.
     * @return The updated header position.
     */
    static long convertSingleExample(SparseVector vector, ArrayList<Float> dataList, ArrayList<Integer> indicesList, ArrayList<Long> headersList, long header) {
        int numActiveFeatures = 0;
        for (VectorTuple v : vector) {
            dataList.add((float) v.value);
            indicesList.add(v.index);
            numActiveFeatures++;
        }
        header += numActiveFeatures;
        headersList.add(header);
        return header;
    }

    /**
     * Used when predicting with an externally trained XGBoost model.
     * @param vector The features to convert.
     * @return A DMatrix representing the features.
     * @throws XGBoostError If the native library returns an error state.
     */
    protected static DMatrix convertSparseVector(SparseVector vector) throws XGBoostError {
        // headers = array of start points for a row
        // indices = array of feature indices for all data
        // data = array of feature values for all data
        // SparseType = DMatrix.SparseType.CSR
        //public DMatrix(long[] headers, int[] indices, float[] data, SparseType st) throws XGBoostError
        ArrayList<Float> dataList = new ArrayList<>();
        ArrayList<Long> headersList = new ArrayList<>();
        ArrayList<Integer> indicesList = new ArrayList<>();

        long rowHeader = 0;
        headersList.add(rowHeader);
        convertSingleExample(vector,dataList,indicesList,headersList,rowHeader);

        float[] data = Util.toPrimitiveFloat(dataList);
        int[] indices = Util.toPrimitiveInt(indicesList);
        long[] headers = Util.toPrimitiveLong(headersList);

        return new DMatrix(headers, indices, data, DMatrix.SparseType.CSR,vector.size());
    }

    /**
     * Used when predicting with an externally trained XGBoost model.
     * <p>
     * It is assumed all vectors are the same size when passed into this function.
     * @param vectors The batch of features to convert.
     * @return A DMatrix representing the batch of features.
     * @throws XGBoostError If the native library returns an error state.
     */
    protected static DMatrix convertSparseVectors(List<SparseVector> vectors) throws XGBoostError {
        // headers = array of start points for a row
        // indices = array of feature indices for all data
        // data = array of feature values for all data
        // SparseType = DMatrix.SparseType.CSR
        //public DMatrix(long[] headers, int[] indices, float[] data, SparseType st) throws XGBoostError
        ArrayList<Float> dataList = new ArrayList<>();
        ArrayList<Long> headersList = new ArrayList<>();
        ArrayList<Integer> indicesList = new ArrayList<>();

        int numFeatures = 0;
        long rowHeader = 0;
        headersList.add(rowHeader);
        for (SparseVector e : vectors) {
            rowHeader = convertSingleExample(e,dataList,indicesList,headersList,rowHeader);
            numFeatures = e.size(); // All vectors are assumed to be the same size.
        }

        float[] data = Util.toPrimitiveFloat(dataList);
        int[] indices = Util.toPrimitiveInt(indicesList);
        long[] headers = Util.toPrimitiveLong(headersList);

        return new DMatrix(headers, indices, data, DMatrix.SparseType.CSR, numFeatures);
    }

    /**
     * Tuple of a DMatrix, the number of valid features in each example, and the examples themselves.
     * <p>
     * One day it'll be a record.
     * @param <T> The output type.
     */
    protected static class DMatrixTuple<T extends Output<T>> {
        /**
         * The data matrix.
         */
        public final DMatrix data;
        /**
         * The number of valid features in each example.
         */
        public final int[] numValidFeatures;
        /**
         * The examples.
         */
        public final Example<T>[] examples;

        /**
         * Constructs a tuple containing the data and some Tribuo metadata.
         * @param data The data matrix.
         * @param numValidFeatures The number of valid features in each example.
         * @param examples The examples.
         */
        protected DMatrixTuple(DMatrix data, int[] numValidFeatures, Example<T>[] examples) {
            this.data = data;
            this.numValidFeatures = numValidFeatures;
            this.examples = examples;
        }
    }

    /**
     * Provenance for {@link XGBoostTrainer}. No longer used.
     * @deprecated Unused.
     */
    @Deprecated
    protected static class XGBoostTrainerProvenance extends SkeletalTrainerProvenance {
        private static final long serialVersionUID = 1L;

        /**
         * Constructs an XGBoostTrainerProvenance
         * @param host The host object.
         * @param <T> The output type.
         */
        protected <T extends Output<T>> XGBoostTrainerProvenance(XGBoostTrainer<T> host) {
            super(host);
        }

        /**
         * Deserializes an XGBoostTrainerProvenance.
         * @param map The map to deserialize from.
         */
        protected XGBoostTrainerProvenance(Map<String,Provenance> map) {
            super(map);
        }
    }
}
