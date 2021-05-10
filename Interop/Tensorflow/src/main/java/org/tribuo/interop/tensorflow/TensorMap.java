/*
 * Copyright (c) 2021 Oracle and/or its affiliates. All rights reserved.
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

import org.tensorflow.Session;
import org.tensorflow.Tensor;
import java.util.Collections;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * A map of names and tensors to feed into a session.
 */
public final class TensorMap implements AutoCloseable {
    private final Map<String, Tensor> map;
    private boolean isClosed;

    /**
     *  Creates a TensorMap containing the supplied mapping.
     * @param inputName The input name.
     * @param value The tensor value.
     */
    public TensorMap(String inputName, Tensor value) {
        this.map = Collections.singletonMap(inputName,value);
        this.isClosed = false;
    }

    /**
     * Creates a new TensorMap wrapping the supplied map.
     * @param map A map from strings to tensors.
     */
    public TensorMap(Map<String, Tensor> map) {
        this.map = Collections.unmodifiableMap(map);
        this.isClosed = false;
    }

    /**
     * Returns the underlying immutable map.
     * @return The map.
     */
    public Map<String, Tensor> getMap() {
        return map;
    }

    /**
     * Returns the specified tensor if present.
     * @param key The key to lookup.
     * @return An optional containing the specified tensor if it's in the map, an empty optional otherwise.
     */
    public Optional<Tensor> getTensor(String key) {
        return Optional.ofNullable(map.get(key));
    }

    /**
     * Feeds the tensors in this FeedDict into the runner.
     * <p>
     * If the session does not have placeholders with the names used in this TensorMap
     * then the {@code Session.Runner} will throw {@link IllegalArgumentException}.
     *
     * @param runner The session runner.
     * @return The runner.
     */
    public Session.Runner feedInto(Session.Runner runner) {
        if (isClosed) {
            throw new IllegalStateException("Can't feed closed Tensors into a Runner.");
        }
        for (Map.Entry<String, Tensor> e : map.entrySet()) {
            runner.feed(e.getKey(), e.getValue());
        }
        return runner;
    }

    @Override
    public void close() {
        isClosed = true;
        for (Tensor t : map.values()) {
            t.close();
        }
    }

    @Override
    public String toString() {
        return "TensorMap(" +
                map.entrySet().stream().map(e -> e.getKey() + ":Shape" + e.getValue().shape().toString())
                        .collect(Collectors.joining(",")) +
                ")";
    }
}
