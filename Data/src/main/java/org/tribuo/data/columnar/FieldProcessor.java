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

import java.util.List;

/**
 * An interface for things that process the columns in a data set.
 */
public interface FieldProcessor extends Configurable, Provenancable<ConfiguredObjectProvenance> {

    /**
     * The namespacing separator.
     */
    public static final String NAMESPACE = "#";

    /**
     * The types of generated features.
     */
    public enum GeneratedFeatureType {
        /**
         * Categoricals binarised into separate features.
         */
        BINARISED_CATEGORICAL,
        /**
         * Unordered categorical features with the values converted into doubles.
         */
        CATEGORICAL,
        /**
         * Ordered integral feature values (e.g. day of month) stored as doubles.
         */
        INTEGER,
        /**
         * Real valued features.
         */
        REAL,
        /**
         * Text features.
         */
        TEXT
    }

    /**
     * Gets the field name this FieldProcessor uses.
     * @return The field name.
     */
    public String getFieldName();

    /**
     * Processes the field value and generates a (possibly empty) list of {@link ColumnarFeature}s.
     * @param value The field value to process.
     * @return A list of {@link ColumnarFeature}s.
     */
    public List<ColumnarFeature> process(String value);

    /**
     * Returns the feature type this FieldProcessor generates.
     * @return The feature type.
     */
    public GeneratedFeatureType getFeatureType();

    /**
     * Binarised categoricals can be namespaced, where the field name is appended with "#&lt;non-negative-int&gt;" to denote the
     * namespace. This allows one FieldProcessor to emit multiple binarised categoricals from the same field value,
     * provided each emitted feature is in a different namespace. Without this guarantee it's impossible to
     * recover the original categorical distribution before binarisation.
     *
     * If there is only a single namespace, it is omitted from the feature name.
     * @return The number of namespaces.
     */
    default public int getNumNamespaces() {
        return 1;
    }

    /**
     * Returns a copy of this FieldProcessor bound to the supplied newFieldName.
     * @param newFieldName The new field name for the copy.
     * @return A copy of this FieldProcessor.
     */
    public FieldProcessor copy(String newFieldName);
}
