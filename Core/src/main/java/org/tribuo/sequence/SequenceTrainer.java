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

package org.tribuo.sequence;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Output;
import org.tribuo.provenance.TrainerProvenance;

import java.util.Collections;
import java.util.Map;

/**
 * An interface for things that can train sequence prediction models.
 */
public interface SequenceTrainer<T extends Output<T>> extends Configurable, Provenancable<TrainerProvenance> {
    
    /**
     * Trains a sequence prediction model using the examples in the given data set.
     * @param examples the data set containing the examples.
     * @return a prediction model that can be used to predict values for new examples.
     */
    default public SequenceModel<T> train(SequenceDataset<T> examples) {
        return train(examples, Collections.emptyMap());
    }

    /**
     * Trains a sequence prediction model using the examples in the given data set.
     * @param examples the data set containing the examples.
     * @param runProvenance Training run specific provenance (e.g., fold number).
     * @return a predictive model that can be used to generate predictions for new examples.
     */
    public SequenceModel<T> train(SequenceDataset<T> examples, Map<String, Provenance> runProvenance);

    /**
     * Returns the number of times the train method has been invoked.
     * @return The number of times train has been invoked.
     */
    public int getInvocationCount();
}
