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

package org.tribuo.data.text;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.Example;
import org.tribuo.Output;

/**
 * An interface for things that take text and turn them into examples that we
 * can use to train or evaluate a classifier.
 * @param <T> The type of the features that will be produced by the text
 * processing.
 */
public interface TextFeatureExtractor<T extends Output<T>> extends Configurable, Provenancable<ConfiguredObjectProvenance> {

    /**
     * Extracts an example from the supplied input text and output object.
     * @param output The output object.
     * @param data The input text.
     * @return An example
     */
    public Example<T> extract(T output, String data);

}
