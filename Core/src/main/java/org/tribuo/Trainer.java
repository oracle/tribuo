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

package org.tribuo;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.provenance.TrainerProvenance;

import java.util.Collections;
import java.util.Map;
import java.util.logging.Logger;

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
     * When training a model, passing this value will inform the trainer to
     * simply increment the invocation count rather than set a new one
     */
    public static int INCREMENT_INVOCATION_COUNT = -1;
    
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
     * @param runProvenance Training run specific provenance (e.g., fold number).
     * @return a predictive model that can be used to generate predictions for new examples.
     */
    public Model<T> train(Dataset<T> examples, Map<String, Provenance> runProvenance);

    /**
     * Trains a predictive model using the examples in the given data set.
     *
     * @param examples        the data set containing the examples.
     * @param runProvenance   Training run specific provenance (e.g., fold number).
     * @param invocationCount The invocation counter that the trainer should be set to before training, which in most
     *                        cases alters the state of the RNG inside this trainer. If the value is set to
     *                        {@link #INCREMENT_INVOCATION_COUNT} then the invocation count is not changed.
     * @return a predictive model that can be used to generate predictions for new examples.
     */
    public default Model<T> train(Dataset<T> examples, Map<String, Provenance> runProvenance, int invocationCount) {
        synchronized (this) {
            if (invocationCount != INCREMENT_INVOCATION_COUNT) {
                setInvocationCount(invocationCount);
            }
            return train(examples, runProvenance);
        }
    }

    /**
     * The number of times this trainer instance has had it's train method invoked.
     * <p>
     * This is used to determine how many times the trainer's RNG has been accessed
     * to ensure replicability in the random number stream.
     * @return The number of train invocations.
     */
    public int getInvocationCount();

    /**
     * Set the internal state of the trainer to the provided number of invocations of the train method.
     * <p>
     * This is used when reproducing a Tribuo-trained model by setting the state of the RNG to
     * what it was at when Tribuo trained the original model by simulating invocations of the train method.
     * This method should ALWAYS be overridden, and the default method is purely for compatibility.
     * <p>
     * In a future major release this default implementation will be removed.
     * @param  invocationCount the number of invocations of the train method to simulate
     */
    default public void setInvocationCount(int invocationCount){
        Logger.getLogger(this.getClass().getName()).warning("This class is using the default implementation of " +
                "setInvocationCount and so might not behave as expected when reproduced. We highly recommend overriding " +
                "this method as per the documentation.");
    }
}
