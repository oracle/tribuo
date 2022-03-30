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

package org.tribuo.common.liblinear;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
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
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Parameter;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.SplittableRandom;
import java.util.logging.Logger;

/**
 * A {@link Trainer} which wraps a liblinear-java trainer.
 * <p>
 * See:
 * <pre>
 * Fan RE, Chang KW, Hsieh CJ, Wang XR, Lin CJ.
 * "LIBLINEAR: A library for Large Linear Classification"
 * Journal of Machine Learning Research, 2008.
 * </pre>
 * and for the original algorithm:
 * <pre>
 * Cortes C, Vapnik V.
 * "Support-Vector Networks"
 * Machine Learning, 1995.
 * </pre>
 */
public abstract class LibLinearTrainer<T extends Output<T>> implements Trainer<T> {

    private static final Logger logger = Logger.getLogger(LibLinearTrainer.class.getName());

    protected Parameter libLinearParams;

    @Config(description="Algorithm to use.")
    protected LibLinearType<T> trainerType;

    @Config(description="Cost penalty for misclassifications.")
    protected double cost = 1;

    @Config(description="Maximum number of iterations before terminating.")
    protected int maxIterations = 1000;

    @Config(description="Stop iterating when the loss score decreases by less than this value.")
    protected double terminationCriterion = 0.1;

    @Config(description="Epsilon insensitivity in the regression cost function.")
    protected double epsilon = 0.1;

    @Config(description="RNG seed.")
    protected long seed = Trainer.DEFAULT_SEED;

    private SplittableRandom rng;

    private int trainInvocationCount = 0;

    /**
     * For OLCUT
     */
    protected LibLinearTrainer() {}

    /**
     * Creates a trainer for a LibLinear model
     * <p>
     * Uses {@link Trainer#DEFAULT_SEED} as the RNG seed, and 0.1 as epsilon.
     * @param trainerType Loss function and optimisation method combination.
     * @param cost Cost penalty for each incorrectly classified training point.
     * @param maxIterations The maximum number of dataset iterations.
     * @param terminationCriterion How close does the optimisation function need to be before terminating that subproblem (usually set to 0.1).
     */
    protected LibLinearTrainer(LibLinearType<T> trainerType, double cost, int maxIterations, double terminationCriterion) {
        this(trainerType,cost,maxIterations,terminationCriterion,0.1);
    }

    /**
     * Creates a trainer for a LibLinear model
     * @param trainerType Loss function and optimisation method combination.
     * @param cost Cost penalty for each incorrectly classified training point.
     * @param maxIterations The maximum number of dataset iterations.
     * @param terminationCriterion How close does the optimisation function need to be before terminating that subproblem (usually set to 0.1).
     */
    protected LibLinearTrainer(LibLinearType<T> trainerType, double cost, int maxIterations, double terminationCriterion, long seed) {
        this(trainerType,cost,maxIterations,terminationCriterion,0.1, seed);
    }

    /**
     * Creates a trainer for a LibLinear model
     * <p>
     * Uses {@link Trainer#DEFAULT_SEED} as the RNG seed.
     * @param trainerType Loss function and optimisation method combination.
     * @param cost Cost penalty for each incorrectly classified training point.
     * @param maxIterations The maximum number of dataset iterations.
     * @param terminationCriterion How close does the optimisation function need to be before terminating that subproblem (usually set to 0.1).
     * @param epsilon The insensitivity of the regression loss to small differences.
     */
    protected LibLinearTrainer(LibLinearType<T> trainerType, double cost, int maxIterations, double terminationCriterion, double epsilon) {
        this(trainerType,cost,maxIterations,terminationCriterion,epsilon,Trainer.DEFAULT_SEED);
    }

    /**
     * Creates a trainer for a LibLinear model
     * @param trainerType Loss function and optimisation method combination.
     * @param cost Cost penalty for each incorrectly classified training point.
     * @param maxIterations The maximum number of dataset iterations.
     * @param terminationCriterion How close does the optimisation function need to be before terminating that subproblem (usually set to 0.1).
     * @param epsilon The insensitivity of the regression loss to small differences.
     * @param seed The RNG seed.
     */
    protected LibLinearTrainer(LibLinearType<T> trainerType, double cost, int maxIterations, double terminationCriterion, double epsilon, long seed) {
        this.trainerType = trainerType;
        this.cost = cost;
        this.maxIterations = maxIterations;
        this.terminationCriterion = terminationCriterion;
        this.epsilon = epsilon;
        this.seed = seed;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        libLinearParams = new Parameter(trainerType.getSolverType(),cost,terminationCriterion,maxIterations,epsilon);
        rng = new SplittableRandom(seed);
        Linear.disableDebugOutput();
    }

    @Override
    public LibLinearModel<T> train(Dataset<T> examples) {
        return train(examples, Collections.emptyMap());
    }

    @Override
    public LibLinearModel<T> train(Dataset<T> examples, Map<String, Provenance> runProvenance) {
        return(train(examples, runProvenance, INCREMENT_INVOCATION_COUNT));
    }

    @Override
    public LibLinearModel<T> train(Dataset<T> examples, Map<String, Provenance> runProvenance, int invocationCount) {
        if (examples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }

        // Creates a new RNG, adds one to the invocation count.
        TrainerProvenance trainerProvenance;
        SplittableRandom localRNG;
        synchronized(this) {
            if(invocationCount != INCREMENT_INVOCATION_COUNT) {
                setInvocationCount(invocationCount);
            }
            localRNG = rng.split();
            trainerProvenance = getProvenance();
            trainInvocationCount++;
        }

        ImmutableFeatureMap featureIDMap = examples.getFeatureIDMap();
        ImmutableOutputInfo<T> outputIDInfo = examples.getOutputIDInfo();

        // Setup parameters and RNG
        Parameter curParams = setupParameters(outputIDInfo);
        curParams.setRandom(new Random(localRNG.nextLong()));

        ModelProvenance provenance = new ModelProvenance(LibLinearModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance);

        Pair<FeatureNode[][],double[][]> data = extractData(examples,outputIDInfo,featureIDMap);

        List<de.bwaldvogel.liblinear.Model> models = trainModels(curParams, featureIDMap.size() + 1, data.getA(), data.getB());

        return createModel(provenance,featureIDMap,outputIDInfo,models);
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCount;
    }

    @Override
    public synchronized void setInvocationCount(int invocationCount) {
        if(invocationCount < 0){
            throw new IllegalArgumentException("The supplied invocationCount is less than zero.");
        }

        rng = new SplittableRandom(seed);

        for (trainInvocationCount = 0; trainInvocationCount < invocationCount; trainInvocationCount++){
            SplittableRandom localRNG = rng.split();
        }
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder();

        buffer.append("LibLinearTrainer(");
        buffer.append("solver=");
        buffer.append(libLinearParams.getSolverType());
        buffer.append(",cost=");
        buffer.append(libLinearParams.getC());
        buffer.append(",terminationCriterion=");
        buffer.append(libLinearParams.getEps());
        buffer.append(",maxIterations=");
        buffer.append(libLinearParams.getMaxIters());
        buffer.append(",regression-epsilon=");
        buffer.append(libLinearParams.getP());
        buffer.append(",seed=");
        buffer.append(seed);
        buffer.append(')');

        return buffer.toString();
    }

    /**
     * Train all the liblinear instances necessary for this dataset.
     * @param curParams The LibLinear parameters.
     * @param numFeatures The number of features in this dataset.
     * @param features The features themselves.
     * @param outputs The outputs.
     * @return A list of liblinear models.
     */
    protected abstract List<de.bwaldvogel.liblinear.Model> trainModels(Parameter curParams, int numFeatures, FeatureNode[][] features, double[][] outputs);

    /**
     * Construct the appropriate subtype of LibLinearModel for the prediction task.
     * @param provenance The model provenance.
     * @param featureIDMap The feature id map.
     * @param outputIDInfo The output id info.
     * @param models The list of linear models.
     * @return An implementation of LibLinearModel.
     */
    protected abstract LibLinearModel<T> createModel(ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo, List<de.bwaldvogel.liblinear.Model> models);

    /**
     * Extracts the features and {@link Output}s in LibLinear's format.
     * @param data The input data.
     * @param outputInfo The output info.
     * @param featureMap The feature info.
     * @return The features and outputs.
     */
    protected abstract Pair<FeatureNode[][],double[][]> extractData(Dataset<T> data, ImmutableOutputInfo<T> outputInfo, ImmutableFeatureMap featureMap);

    /**
     * Constructs the parameters. Most of the time this just clones the existing ones, but
     * classification overrides it to incorporate label weights if they exist.
     * @param info The output info.
     * @return The Parameters to use for training.
     */
    protected Parameter setupParameters(ImmutableOutputInfo<T> info) {
        return libLinearParams.clone();
    }

    /**
     * Converts a Tribuo {@link Example} into a liblinear {@code FeatureNode} array, including a bias feature.
     * <p>
     * If there is a collision between feature ids (i.e., if there is feature hashing or some other mechanism changing
     * the feature ids) then the feature values are summed.
     * @param example The input example.
     * @param featureIDMap The feature id map which contains the example's indices.
     * @param features A buffer. If null then an array list is created and used internally.
     * @param <T> The output type.
     * @return The features suitable for use in liblinear.
     */
    public static <T extends Output<T>> FeatureNode[] exampleToNodes(Example<T> example, ImmutableFeatureMap featureIDMap, List<FeatureNode> features) {
        int biasIndex = featureIDMap.size()+1;

        if (features == null) {
            features = new ArrayList<>();
        }
        features.clear();

        int prevIdx = -1;
        for (Feature f : example) {
            int id = featureIDMap.getID(f.getName());
            if (id > prevIdx){
                prevIdx = id;
                features.add(new FeatureNode(id + 1, f.getValue()));
            } else if (id > -1) {
                //
                // Collision, deal with it.
                int collisionIdx = Util.binarySearch(features,id+1, FeatureNode::getIndex);
                if (collisionIdx < 0) {
                    //
                    // Collision but not present in features
                    // move data and bump i
                    collisionIdx = - (collisionIdx + 1);
                    features.add(collisionIdx,new FeatureNode(id + 1, f.getValue()));
                } else {
                    //
                    // Collision present in features
                    // add the values.
                    FeatureNode n = features.get(collisionIdx);
                    n.setValue(n.getValue() + f.getValue());
                }
            }
        }

        features.add(new FeatureNode(biasIndex,1.0));

        return features.toArray(new FeatureNode[0]);
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}
