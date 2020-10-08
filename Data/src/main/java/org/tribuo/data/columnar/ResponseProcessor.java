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

package org.tribuo.data.columnar;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.Output;
import org.tribuo.OutputFactory;

import java.util.Optional;

/**
 * An interface that will take the response field and produce an {@link Output}.
 */
public interface ResponseProcessor<T extends Output<T>> extends Configurable, Provenancable<ConfiguredObjectProvenance> {

    /**
     * Gets the OutputFactory this ResponseProcessor uses.
     * @return The output factory.
     */
    public OutputFactory<T> getOutputFactory();

    /**
     * Gets the field name this ResponseProcessor uses.
     * @return The field name.
     */
    public String getFieldName();

    /**
     * @deprecated Response processors should be immutable; downstream objects assume that they are
     * Set the field name this ResponseProcessor uses.
     * @param fieldName The field name.
     *
     */
    @Deprecated()
    public void setFieldName(String fieldName);

    /**
     * Returns Optional.empty() if it failed to process out a response.
     * @param value The value to process.
     * @return The response value if found.
     */
    public Optional<T> process(String value);
}
