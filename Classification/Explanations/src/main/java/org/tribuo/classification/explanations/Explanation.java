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

package org.tribuo.classification.explanations;

import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;

import java.io.Serializable;
import java.util.List;

/**
 * An explanation knows what features are used, what the explaining Model is and what the original Model's prediction is.
 */
public interface Explanation<T extends Output<T>> extends Serializable {

    /**
     * Returns the names of the active features in this explanation.
     * @return The active feature names.
     */
    public List<String> getActiveFeatures();

    /**
     * Returns the explanining model.
     * @return The explanation model.
     */
    public Model<T> getModel();

    /**
     * The original model's prediction which is being explained.
     * @return The original prediction.
     */
    public Prediction<Label> getPrediction();

}
