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

package org.tribuo.interop.tensorflow;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tensorflow.Session;
import org.tensorflow.types.family.TType;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Output;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tensorflow.Tensor;

import java.io.Serializable;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * TensorFlow support is experimental, and may change without a major version bump.
 * <p>
 * Transforms an {@link Example}, extracting the features from it as a {@link FeedDict}.
 * <p>
 * This usually densifies the example, so can be a lot larger than the input example.
 */
public interface ExampleTransformer<T extends Output<T>> extends Configurable, Provenancable<ConfiguredObjectProvenance>, Serializable {

    /**
     * Converts an {@link Example} into a {@link FeedDict} suitable for supplying as an input to a graph.
     * <p>
     * It generates it as a single example minibatch.
     * @param example The example to convert.
     * @param featureIDMap The id map to convert feature names into id numbers.
     * @return A FeedDict representing the features in this example.
     */
    public FeedDict transform(Example<T> example, ImmutableFeatureMap featureIDMap);

    /**
     * Converts a batch of {@link Example}s into a single {@link FeedDict} suitable for supplying as
     * an input to a graph.
     * @param example The examples to convert.
     * @param featureIDMap THe id map to convert feature names into id numbers.
     * @return A FeedDict representing the features in this minibatch.
     */
    public FeedDict transform(List<Example<T>> example, ImmutableFeatureMap featureIDMap);

    /**
     * Converts a {@link SGDVector} representing the features into a {@link FeedDict}.
     * <p>
     * It generates it as a single example minibatch.
     * @param vector The features to convert.
     * @return A FeedDict representing this vector.
     */
    public FeedDict transform(SGDVector vector);

    /**
     * Converts a list of {@link SGDVector}s representing a batch of features into a {@link FeedDict}.
     * <p>
     * @param vectors The batch of features to convert.
     * @return A FeedDict representing this minibatch.
     */
    public FeedDict transform(List<? extends SGDVector> vectors);

    /**
     * Gets a view of the names of the inputs this transformer produces.
     * @return The input names.
     */
    public Set<String> inputNamesSet();

    /**
     * A map of names and tensors to feed into a session.
     */
    public static class FeedDict implements AutoCloseable {
        private final Map<String, TType> map;
        private boolean isClosed;

        public FeedDict(String inputName, TType value) {
            Map<String,TType> tmp = new HashMap<>(1);
            tmp.put(inputName,value);
            this.map = Collections.unmodifiableMap(tmp);
            this.isClosed = false;
        }

        public FeedDict(Map<String, TType> map) {
            this.map = Collections.unmodifiableMap(map);
            this.isClosed = false;
        }

        public Map<String, TType> getMap() {
            return map;
        }

        /**
         * Feeds the tensors in this FeedDict into the runner.
         * @param runner The session runner.
         * @return
         */
        public Session.Runner feedInto(Session.Runner runner) {
            if (isClosed) {
                throw new IllegalStateException("Can't feed closed Tensors into a Runner.");
            }
            for (Map.Entry<String,TType> e : map.entrySet()) {
                runner.feed(e.getKey(),e.getValue());
            }
            return runner;
        }

        public void close() {
            isClosed = true;
            for (TType t : map.values()) {
                t.close();
            }
        }
    }

}
