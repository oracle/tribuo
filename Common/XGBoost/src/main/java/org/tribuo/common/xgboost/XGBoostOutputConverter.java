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

import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Prediction;

import java.io.Serializable;
import java.util.List;
import org.tribuo.common.xgboost.protos.XGBoostOutputConverterProto;
import org.tribuo.protos.ProtoSerializable;

/**
 * Converts the output of XGBoost into the appropriate prediction type.
 */
public interface XGBoostOutputConverter<T extends Output<T>> extends
    ProtoSerializable<XGBoostOutputConverterProto>, Serializable {

    /**
     * Does this converter produce probabilities?
     * @return True if it produces probabilities.
     */
    public boolean generatesProbabilities();

    /**
     * Converts a list of float arrays from XGBoost Boosters into a Tribuo {@link Prediction}.
     * @param info The output info.
     * @param probabilities The XGBoost output.
     * @param numValidFeatures The number of valid features used in the prediction.
     * @param example The example this prediction was generated from.
     * @return The prediction object.
     */
    public Prediction<T> convertOutput(ImmutableOutputInfo<T> info, List<float[]> probabilities, int numValidFeatures, Example<T> example);

    /**
     * Converts a list of float arrays from XGBoost Boosters into a Tribuo {@link Prediction}.
     * @param info The output info.
     * @param probabilities The XGBoost output, list dimension is across models, first array dimension is across examples, second array dimension is across outputs.
     * @param numValidFeatures The number of valid features used in each prediction.
     * @param examples The examples these predictions were generated from.
     * @return The prediction object.
     */
    public List<Prediction<T>> convertBatchOutput(ImmutableOutputInfo<T> info, List<float[][]> probabilities, int[] numValidFeatures, Example<T>[] examples);

    /**
     * Gets the type witness for the output this converter uses.
     * <p>
     * The default implementation throws {@link UnsupportedOperationException} for compatibility
     * with subclasses which don't support protobuf serialization.
     * @return The class of the output.
     */
    default public Class<T> getTypeWitness() {
        throw new UnsupportedOperationException("This class has not been updated to support protobuf serialization.");
    }

}
