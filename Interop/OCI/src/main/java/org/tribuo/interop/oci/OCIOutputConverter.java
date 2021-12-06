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

package org.tribuo.interop.oci;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;

import java.io.Serializable;
import java.util.List;

/**
 * Converter for a {@link DenseMatrix} received from OCI Data Science Model Deployment.
 * @param <T> The output type.
 */
public interface OCIOutputConverter<T extends Output<T>> extends Configurable, Provenancable<ConfiguredObjectProvenance>,  Serializable {

    /**
     * Converts a dense vector into a single prediction of the appropriate type.
     * @param scores The score vector.
     * @param numValidFeature The number of valid features (stored in the prediction).
     * @param example The example (stored in the prediction).
     * @param outputIDInfo The output information.
     * @return A prediction representing the value of this score vector.
     */
    public Prediction<T> convertOutput(DenseVector scores, int numValidFeature, Example<T> example, ImmutableOutputInfo<T> outputIDInfo);

    /**
     * Converts a dense matrix into a list of predictions of the appropriate type.
     * @param scores The score matrix.
     * @param numValidFeatures The number of valid features in each example (stored in the prediction).
     * @param examples The examples (stored in the prediction).
     * @param outputIDInfo The output information.
     * @return A list of predictions representing the value of this score matrix.
     */
    public List<Prediction<T>> convertOutput(DenseMatrix scores, int[] numValidFeatures, List<Example<T>> examples, ImmutableOutputInfo<T> outputIDInfo);

    /**
     * Does this OCIOutputConverter generate probabilities?
     * @return True if it produces a probability distribution in the {@link Prediction}.
     */
    public boolean generatesProbabilities();
}
