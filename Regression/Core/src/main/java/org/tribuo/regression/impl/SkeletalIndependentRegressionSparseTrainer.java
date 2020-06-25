/*
 * Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.regression.impl;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.SparseTrainer;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.regression.Regressor;

import java.time.OffsetDateTime;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Base class for training n independent sparse models, one per dimension. Generates the SparseVectors
 * once to reduce allocation.
 * <p>
 * Then wraps them in an {@link SkeletalIndependentRegressionSparseModel} to provide a {@link Regressor}
 * prediction.
 * <p>
 * It trains each model sequentially, and could be optimised to train in parallel.
 */
public abstract class SkeletalIndependentRegressionSparseTrainer<T> implements SparseTrainer<Regressor> {

    @Config(description="Seed for the RNG, may be unused.")
    private long seed = 1L;

    private SplittableRandom rng;

    private int trainInvocationCounter = 0;

    /**
     * for olcut.
     */
    protected SkeletalIndependentRegressionSparseTrainer() {}

    @Override
    public synchronized void postConfig() {
        this.rng = new SplittableRandom(seed);
    }

    @Override
    public SkeletalIndependentRegressionSparseModel train(Dataset<Regressor> examples) {
        return train(examples, Collections.emptyMap());
    }

    @Override
    public SkeletalIndependentRegressionSparseModel train(Dataset<Regressor> examples, Map<String, Provenance> runProvenance) {
        if (examples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }
        SplittableRandom localRNG;
        TrainerProvenance trainerProvenance;
        synchronized(this) {
            localRNG = rng.split();
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }
        ImmutableOutputInfo<Regressor> outputInfo = examples.getOutputIDInfo();
        ImmutableFeatureMap featureMap = examples.getFeatureIDMap();
        Set<Regressor> domain = outputInfo.getDomain();
        LinkedHashMap<String, T> models = new LinkedHashMap<>();
        int numExamples = examples.size();
        boolean needBias = useBias();
        float[] weights = new float[numExamples];
        double[][] outputs = new double[outputInfo.size()][numExamples];
        SparseVector[] inputs = new SparseVector[numExamples];
        int i = 0;
        for (Example<Regressor> e : examples) {
            inputs[i] = SparseVector.createSparseVector(e,featureMap,needBias);
            weights[i] = e.getWeight();
            for (Regressor.DimensionTuple r : e.getOutput()) {
                int id = outputInfo.getID(r);
                outputs[id][i] = r.getValue();
            }
            i++;
        }
        for (Regressor r : domain) {
            int id = outputInfo.getID(r);
            T innerModel = trainDimension(outputs[id],inputs,weights,localRNG);
            models.put(r.getNames()[0],innerModel);
        }
        ModelProvenance provenance = new ModelProvenance(getModelClassName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance);
        return createModel(models,provenance,featureMap,outputInfo);
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    /**
     * Constructs the appropriate subclass of {@link SkeletalIndependentRegressionModel} for this trainer.
     * @param models The models to use.
     * @param provenance The model provenance
     * @param featureMap The feature map.
     * @param outputInfo The regression info.
     * @return A subclass of IndependentRegressionModel.
     */
    protected abstract SkeletalIndependentRegressionSparseModel createModel(Map<String,T> models, ModelProvenance provenance, ImmutableFeatureMap featureMap, ImmutableOutputInfo<Regressor> outputInfo);

    /**
     * Trains a single dimension of the possibly multiple dimensions.
     * @param outputs The regression targets for this dimension.
     * @param features The features.
     * @param weights The example weights.
     * @param rng The RNG to use.
     * @return An object representing the model.
     */
    protected abstract T trainDimension(double[] outputs, SparseVector[] features, float[] weights, SplittableRandom rng);

    /**
     * Returns true if the SparseVector should be constructed with a bias feature.
     * @return True if the trainer needs a bias.
     */
    protected abstract boolean useBias();

    /**
     * Returns the class name of the model that this class produces.
     * @return The class name of the model.
     */
    protected abstract String getModelClassName();
}

