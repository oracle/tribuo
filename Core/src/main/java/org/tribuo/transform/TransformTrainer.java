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

package org.tribuo.transform;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Output;
import org.tribuo.Trainer;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;

import java.time.OffsetDateTime;
import java.util.Map;

/**
 * A {@link Trainer} which encapsulates another trainer plus a {@link TransformationMap} object
 * to apply to each {@link Dataset} before training each {@link Model}.
 * <p>
 * Transformations only operate on observed values. To operate on implicit zeros then
 * first call {@link MutableDataset#densify} on the datasets.
 */
public final class TransformTrainer<T extends Output<T>> implements Trainer<T> {

    @Config(mandatory = true,description="Trainer to use.")
    private Trainer<T> innerTrainer;

    @Config(mandatory = true,description="Transformations to apply.")
    private TransformationMap transformations;

    @Config(description="Densify all the features before applying transformations.")
    private boolean densify;

    /**
     * For OLCUT.
     */
    private TransformTrainer() {}

    /**
     * Creates a trainer which transforms the data before training, and stores
     * the transformers along with the trained model in a {@link TransformedModel}.
     * <p>
     * This constructor makes a trainer which keeps the data sparse.
     * @param innerTrainer The trainer to use.
     * @param transformations The transformations to apply to the data first.
     */
    public TransformTrainer(Trainer<T> innerTrainer, TransformationMap transformations) {
        this(innerTrainer,transformations,false);
    }

    /**
     * Creates a trainer which transforms the data before training, and stores
     * the transformers along with the trained model in a {@link TransformedModel}.
     *
     * @param innerTrainer The trainer to use.
     * @param transformations The transformations to apply to the data first.
     * @param densify Densify the dataset (and any predict time data) before training/prediction.
     */
    public TransformTrainer(Trainer<T> innerTrainer, TransformationMap transformations, boolean densify) {
        this.innerTrainer = innerTrainer;
        this.transformations = transformations;
        this.densify = densify;
    }

    @Override
    public TransformedModel<T> train(Dataset<T> examples, Map<String, Provenance> instanceProvenance) {
        TransformerMap transformerMap = examples.createTransformers(transformations);

        Dataset<T> transformedDataset = transformerMap.transformDataset(examples,densify);

        Model<T> innerModel = innerTrainer.train(transformedDataset);

        ModelProvenance provenance = new ModelProvenance(TransformedModel.class.getName(), OffsetDateTime.now(), transformedDataset.getProvenance(), getProvenance(), instanceProvenance);

        return new TransformedModel<>(provenance,innerModel,transformerMap,densify);
    }

    @Override
    public int getInvocationCount() {
        return innerTrainer.getInvocationCount();
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}
