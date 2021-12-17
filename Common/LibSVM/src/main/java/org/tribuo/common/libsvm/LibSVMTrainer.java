/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.common.libsvm;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Trainer;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.util.Util;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.logging.Logger;

/**
 * A trainer that will train using libsvm's Java implementation.
 * <p>
 * Note the train method is synchronized on {@code LibSVMTrainer.class} due to a global RNG in LibSVM.
 * This is insufficient to ensure reproducibility if LibSVM is used directly in the same JVM as Tribuo, but
 * avoids locking on classes Tribuo does not control.
 * <p>
 * See:
 * <pre>
 * Chang CC, Lin CJ.
 * "LIBSVM: a library for Support Vector Machines"
 * ACM transactions on intelligent systems and technology (TIST), 2011.
 * </pre>
 * for the nu-svm algorithm:
 * <pre>
 * Schölkopf B, Smola A, Williamson R, Bartlett P L.
 * "New support vector algorithms"
 * Neural Computation, 2000, 1207-1245.
 * </pre>
 * and for the original algorithm:
 * <pre>
 * Cortes C, Vapnik V.
 * "Support-Vector Networks"
 * Machine Learning, 1995.
 * </pre>
 */
public abstract class LibSVMTrainer<T extends Output<T>> implements Trainer<T> {
    
    private static final Logger logger = Logger.getLogger(LibSVMTrainer.class.getName());

    /**
     * The SVM parameters suitable for use by LibSVM.
     */
    protected svm_parameter parameters;

    /**
     * The type of SVM algorithm.
     */
    @Config(mandatory=true,description="Type of SVM algorithm.")
    protected SVMType<T> svmType;

    @Config(description="Type of Kernel.")
    private KernelType kernelType = KernelType.LINEAR;

    @Config(description="Polynomial degree.")
    private int degree = 3;

    @Config(description="Width of the RBF kernel, or scalar on sigmoid kernel.")
    private double gamma = 0.0;

    @Config(description="Polynomial coefficient or shift in sigmoid kernel.")
    private double coef0 = 0.0;

    @Config(description="nu value in NU SVM.")
    private double nu = 0.5;

    @Config(description="Internal cache size, most of the time should be left at default.")
    private double cache_size = 500;

    @Config(description="Cost parameter for incorrect predictions.")
    private double cost = 1.0; // aka svm_parameters.C

    @Config(description="Tolerance of the termination criterion.")
    private double eps = 1e-3;

    @Config(description="Epsilon in EPSILON_SVR.")
    private double p = 1e-3;

    @Config(description="Regularise the weight parameters.")
    private boolean shrinking = true;

    @Config(description="Generate probability estimates.")
    private boolean probability = false;

    @Config(description="RNG seed.")
    private long seed = Trainer.DEFAULT_SEED;

    private SplittableRandom rng;

    private int trainInvocationCounter = 0;

    /**
     * For olcut.
     */
    protected LibSVMTrainer() {}

    /**
     * Constructs a LibSVMTrainer from the parameters.
     * @param parameters The SVM parameters.
     * @param seed The RNG seed.
     */
    protected LibSVMTrainer(SVMParameters<T> parameters, long seed) {
        this.parameters = parameters.getParameters();
        // Unpack the parameters for the provenance system.
        this.svmType = parameters.getSvmType();
        this.kernelType = parameters.getKernelType();
        this.degree = this.parameters.degree;
        this.gamma = parameters.getGamma();
        this.coef0 = this.parameters.coef0;
        this.nu = this.parameters.nu;
        this.cache_size = this.parameters.cache_size;
        this.cost = this.parameters.C;
        this.eps = this.parameters.eps;
        this.p = this.parameters.p;
        this.shrinking = this.parameters.shrinking == 1;
        this.probability = this.parameters.probability == 1;
        this.seed = seed;
        this.rng = new SplittableRandom(seed);
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        parameters = new svm_parameter();
        parameters.svm_type = svmType.getNativeType();
        parameters.kernel_type = kernelType.getNativeType();
        parameters.degree = degree;
        parameters.gamma = gamma;
        parameters.coef0 = coef0;
        parameters.nu = nu;
        parameters.cache_size = cache_size;
        parameters.C = cost;
        parameters.eps = eps;
        parameters.p = p;
        parameters.shrinking = shrinking ? 1 : 0;
        parameters.probability = probability ? 1 : 0;
        this.rng = new SplittableRandom(seed);
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder();

        buffer.append("LibSVMTrainer(");
        buffer.append("svm_params=");
        buffer.append(SVMParameters.svmParamsToString(parameters));
        buffer.append(",seed=");
        buffer.append(seed);
        buffer.append(")");

        return buffer.toString();
    }

    @Override
    public LibSVMModel<T> train(Dataset<T> examples) {
        return train(examples, Collections.emptyMap());
    }

    @Override
    public LibSVMModel<T> train(Dataset<T> examples, Map<String, Provenance> runProvenance) {
        return (train(examples, runProvenance, INCREMENT_INVOCATION_COUNT));
    }

    @Override
    public LibSVMModel<T> train(Dataset<T> examples, Map<String, Provenance> runProvenance, int invocationCount) {
        if (examples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }
        ImmutableFeatureMap featureIDMap = examples.getFeatureIDMap();
        ImmutableOutputInfo<T> outputIDInfo = examples.getOutputIDInfo();

        // Creates a new RNG, adds one to the invocation count, generates a local optimiser.
        TrainerProvenance trainerProvenance;
        SplittableRandom localRNG;
        synchronized(this) {
            if(invocationCount != INCREMENT_INVOCATION_COUNT) {
                setInvocationCount(invocationCount);
            }
            localRNG = rng.split();
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }

        svm_parameter curParams = setupParameters(outputIDInfo);

        Pair<svm_node[][],double[][]> data = extractData(examples,outputIDInfo,featureIDMap);

        List<svm_model> models;
        synchronized (LibSVMTrainer.class) {
            // localRNG is used to seed LibSVM's global RNG before each call to svm.svm_train
            models = trainModels(curParams, featureIDMap.size() + 1, data.getA(), data.getB(), localRNG);
        }

        ModelProvenance provenance = new ModelProvenance(LibSVMModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance);
        return createModel(provenance,featureIDMap,outputIDInfo,models);
    }

    /**
     * Construct the appropriate subtype of LibSVMModel for the prediction task.
     * @param provenance The model provenance.
     * @param featureIDMap The feature id map.
     * @param outputIDInfo The output id info.
     * @param models The svm models.
     * @return An implementation of LibSVMModel.
     */
    protected abstract LibSVMModel<T> createModel(ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo, List<svm_model> models);

    /**
     * Train all the LibSVM instances necessary for this dataset.
     * @param curParams The LibSVM parameters.
     * @param numFeatures The number of features in this dataset.
     * @param features The features themselves.
     * @param outputs The outputs.
     * @param localRNG The RNG to use for seeding LibSVM's RNG.
     * @return A list of LibSVM models.
     */
    protected abstract List<svm_model> trainModels(svm_parameter curParams, int numFeatures, svm_node[][] features, double[][] outputs, SplittableRandom localRNG);

    /**
     * Extracts the features and {@link Output}s in LibSVM's format.
     * @param data The input data.
     * @param outputInfo The output info.
     * @param featureMap The feature info.
     * @return The features and outputs.
     */
    protected abstract Pair<svm_node[][],double[][]> extractData(Dataset<T> data, ImmutableOutputInfo<T> outputInfo, ImmutableFeatureMap featureMap);

    /**
     * Constructs the svm_parameter. Most of the time this is a no-op, but
     * classification overrides it to incorporate label weights if they exist.
     * @param info The output info.
     * @return The svm_parameters to use for training.
     */
    protected svm_parameter setupParameters(ImmutableOutputInfo<T> info) {
        return SVMParameters.copyParameters(parameters);
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

        rng = new SplittableRandom(seed);

        for (trainInvocationCounter = 0; trainInvocationCounter < invocationCount; trainInvocationCounter++){
            SplittableRandom localRNG = rng.split();
        }
    }

    /**
     * Convert the example into an array of svm_node which represents a sparse feature vector.
     * <p>
     * If there are collisions in the feature ids then the values are summed.
     * @param example The example to convert.
     * @param featureIDMap The feature id map which holds the indices.
     * @param features A buffer to use.
     * @param <T> The type of the ouput.
     * @return A sparse feature vector.
     */
    public static <T extends Output<T>> svm_node[] exampleToNodes(Example<T> example, ImmutableFeatureMap featureIDMap, List<svm_node> features) {
        if (features == null) {
            features = new ArrayList<>();
        }
        features.clear();
        int prevIdx = -1;
        for (Feature f : example) {
            int id = featureIDMap.getID(f.getName());
            double value = f.getValue();
            if (id > prevIdx){
                prevIdx = id;
                svm_node n = new svm_node();
                n.index = id;
                n.value = value;
                features.add(n);
            } else if (id > -1) {
                //
                // Collision, deal with it.
                int collisionIdx = Util.binarySearch(features,id,(svm_node n) -> n.index);
                if (collisionIdx < 0) {
                    //
                    // Collision but not present in features
                    // move data and bump i
                    collisionIdx = - (collisionIdx + 1);
                    svm_node n = new svm_node();
                    n.index = id;
                    n.value = value;
                    features.add(collisionIdx,n);
                } else {
                    //
                    // Collision present in features
                    // add the values.
                    svm_node n = features.get(collisionIdx);
                    n.value += value;
                }
            }
        }
        return features.toArray(new svm_node[0]);
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}

