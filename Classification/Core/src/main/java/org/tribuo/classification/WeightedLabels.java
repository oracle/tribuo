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

package org.tribuo.classification;

import org.tribuo.Trainer;

import java.util.Map;

/**
 * Tag interface denoting the {@link Trainer} can use label weights.
 */
public interface WeightedLabels {

    /**
     * Sets the label weights used by this trainer.
     * <p>
     * Supply {@link java.util.Collections#emptyMap()} to turn off label weights.
     * @param map A map from Label instances to weight values.
     */
    public void setLabelWeights(Map<Label,Float> map);

}


