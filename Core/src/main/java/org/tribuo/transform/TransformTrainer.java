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

package org.tribuo.transform;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Trainer;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;

import java.time.OffsetDateTime;
import java.util.Collections;
import java.util.Map;
import java.util.logging.Logger;

/**
 * A {@link Trainer} which encapsulates another trainer plus a {@link TransformationMap} object
 * to apply to each {@link Dataset} before training each {@link Model}.
 * <p>
 * By default transformations only operate on explicit feature values. To include implicit zeros
 * in transformation fitting set {@code includeImplicitZeroFeatures}. To convert implicit
 * zeros to explicit zeros before applying the transformations set {@code densify}.
 * See {@link org.tribuo.transform} for a more detailed discussion of these parameters.
 */
public final class TransformTrainer<T extends Output<T>> implements Trainer<T> {
    
    private static final Logger logger = Logger.getLogger(TransformTrainer.class.getName());

    @Config(mandatory = true,description="Trainer to use.")
    private Trainer<T> innerTrainer;

    @Config(mandatory = true,description="Transformations to apply.")
    private TransformationMap transformations;

    @Config(description="Densify all the features before applying transformations.")
    private boolean densify;

    @Config(description="Include the implicit zeros in the transformation statistics collection")
    private boolean includeImplicitZeroFeatures;

    /**
     * For OLCUT.
     */
    private TransformTrainer() {}

    /**
     * Creates a trainer which transforms the data before training, and stores
     * the transformers along with the trained model in a {@link TransformedModel}.
     * <p>
     * Sets {@code observeSparse} to false and so this constructor makes a trainer
     * which keeps the data sparse, and does not use the implicit zeros to construct
     * the transformations. Models produced by this trainer will not convert implicit
     * zeros in the feature space to explicit zeros (i.e., densify is false).
     * @param innerTrainer The trainer to use.
     * @param transformations The transformations to apply to the data first.
     */
    public TransformTrainer(Trainer<T> innerTrainer, TransformationMap transformations) {
        this(innerTrainer,transformations,false);
    }

    /**
     * Creates a trainer which transforms the data before training, and stores
     * the transformers along with the trained model in a {@link TransformedModel}.
     * <p>
     * Sets {@code observeSparse} to false and so this constructor makes a trainer
     * which keeps the data sparse, and does not use the implicit zeros to construct
     * the transformations.
     * @param innerTrainer The trainer to use.
     * @param transformations The transformations to apply to the data first.
     * @param densify Convert the implicit zeros in each training and prediction example
     *                to explicit zeros before training/prediction.
     */
    public TransformTrainer(Trainer<T> innerTrainer, TransformationMap transformations, boolean densify) {
        this(innerTrainer,transformations,densify,false);
    }

    /**
     * Creates a trainer which transforms the data before training, and stores
     * the transformers along with the trained model in a {@link TransformedModel}.
     *
     * @param innerTrainer The trainer to use.
     * @param transformations The transformations to apply to the data first.
     * @param densify Convert the implicit zeros in each training and prediction example
     *                to explicit zeros before training/prediction.
     * @param includeImplicitZeroFeatures Use the implicit zero feature values to construct the transformations.
     */
    public TransformTrainer(Trainer<T> innerTrainer, TransformationMap transformations, boolean densify, boolean includeImplicitZeroFeatures) {
        this.innerTrainer = innerTrainer;
        this.transformations = transformations;
        this.densify = densify;
        this.includeImplicitZeroFeatures = includeImplicitZeroFeatures;
    }

    @Override
    public TransformedModel<T> train(Dataset<T> examples, Map<String, Provenance> instanceProvenance) {
        return train(examples, instanceProvenance, INCREMENT_INVOCATION_COUNT);
    }

    @Override
    public TransformedModel<T> train(Dataset<T> examples, Map<String, Provenance> instanceProvenance, int invocationCount) {

        logger.fine("Creating transformers");

        TransformerMap transformerMap = examples.createTransformers(transformations, includeImplicitZeroFeatures);

        logger.fine("Transforming data set");

        Dataset<T> transformedDataset = transformerMap.transformDataset(examples,densify);

        logger.fine("Running inner trainer");

        TrainerProvenance provenance;
        Model<T> innerModel;
        synchronized (innerTrainer) {
            provenance = getProvenance();
            innerModel = innerTrainer.train(transformedDataset, Collections.emptyMap(), invocationCount);
        }

        ModelProvenance modelProvenance = new ModelProvenance(TransformedModel.class.getName(), OffsetDateTime.now(), transformedDataset.getProvenance(), provenance, instanceProvenance);

        return new TransformedModel<>(modelProvenance,innerModel,transformerMap,densify);
    }

    @Override
    public int getInvocationCount() {
        return innerTrainer.getInvocationCount();
    }

    @Override
    public synchronized void setInvocationCount(int invocationCount){
        innerTrainer.setInvocationCount(invocationCount);
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}
