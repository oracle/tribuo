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

package org.tribuo.common.sgd;

import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.math.FeedForwardParameters;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;

/**
 * A model trained using SGD.
 * <p>
 * See:
 * <pre>
 * Bottou L.
 * "Large-Scale Machine Learning with Stochastic Gradient Descent"
 * Proceedings of COMPSTAT, 2010.
 * </pre>
 */
public abstract class AbstractSGDModel<T extends Output<T>> extends Model<T> {
    private static final long serialVersionUID = 1L;

    /**
     * The weights for this model.
     */
    // Note this is not final to allow backwards compatibility for 4.0 models which need to rewrite the field on load.
    protected FeedForwardParameters modelParameters;

    // Defaults to true for backwards compatibility with 4.0 models, not final due to this defaulting.
    protected boolean addBias = true;

    /**
     * Constructs a linear model trained via SGD.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     * @param weights The model weights.
     * @param generatesProbabilities Does this model generate probabilities?
     * @param addBias Should the model add a bias feature to the feature vector?
     */
    protected AbstractSGDModel(String name, ModelProvenance provenance,
                               ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
                               FeedForwardParameters weights, boolean generatesProbabilities, boolean addBias) {
        super(name, provenance, featureIDMap, outputIDInfo, generatesProbabilities);
        this.modelParameters = weights;
        this.addBias = addBias;
    }

    /**
     * Generates the dense vector prediction from the supplied example.
     * @param example The example to use for prediction.
     * @return The prediction and the number of features involved.
     */
    protected PredAndActive predictSingle(Example<T> example) {
        SGDVector features;
        if (example.size() == featureIDMap.size()) {
            features = DenseVector.createDenseVector(example, featureIDMap, addBias);
        } else {
            features = SparseVector.createSparseVector(example, featureIDMap, addBias);
        }
        int minNumFeatures = addBias ? 1 : 0;
        if (features.numActiveElements() == minNumFeatures) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }
        return new PredAndActive(modelParameters.predict(features),features.numActiveElements());
    }

    /**
     * Returns a copy of the model parameters.
     * @return A copy of the model parameters.
     */
    public FeedForwardParameters getModelParameters() {
        return modelParameters.copy();
    }

    /**
     * A nominal tuple used to capture the prediction and the number of active features used by the model.
     */
    protected static final class PredAndActive {
        /**
         * The vector prediction.
         */
        public final DenseVector prediction;
        /**
         * The number of active features used in the prediction.
         */
        public final int numActiveFeatures;

        PredAndActive(DenseVector prediction, int numActiveFeatures) {
            this.prediction = prediction;
            this.numActiveFeatures = numActiveFeatures;
        }
    }
}
