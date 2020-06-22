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

package org.tribuo;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.provenance.TrainerProvenance;

import java.util.Collections;
import java.util.Map;

/**
 * An interface for things that can train predictive models.
 * @param <T> the type of the {@link Output} in the examples
 */
public interface Trainer<T extends Output<T>> extends Configurable, Provenancable<TrainerProvenance> {

    /**
     * Default seed used to initialise RNGs.
     */
    public static long DEFAULT_SEED = 12345L;
    
    /**
     * Trains a predictive model using the examples in the given data set.
     * @param examples the data set containing the examples.
     * @return a predictive model that can be used to generate predictions for new examples.
     */
    default public Model<T> train(Dataset<T> examples) {
        return train(examples, Collections.emptyMap());
    }

    /**
     * Trains a predictive model using the examples in the given data set.
     * @param examples the data set containing the examples.
     * @param runProvenance Training run specific provenance (e.g. fold number).
     * @return a predictive model that can be used to generate predictions for new examples.
     */
    public Model<T> train(Dataset<T> examples, Map<String, Provenance> runProvenance);

    /**
     * The number of times this trainer instance has had it's train method invoked.
     * <p>
     * This is used to determine how many times the trainer's RNG has been accessed
     * to ensure replicability in the random number stream.
     * @return The number of train invocations.
     */
    public int getInvocationCount();
}
