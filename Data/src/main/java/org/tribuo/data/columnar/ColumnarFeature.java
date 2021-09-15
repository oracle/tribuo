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

import org.tribuo.Example;
import org.tribuo.Feature;

/**
 * A Feature with extra bookkeeping for use inside the columnar package.
 * <p>
 * {@link Example}s may destroy and recreate Feature instances so don't
 * expect ColumnarFeatures to still be ColumnarFeatures if you probe the
 * Example after construction.
 */
public class ColumnarFeature extends Feature {
    private static final long serialVersionUID = 1L;

    /**
     * The string used as the field name of conjunction features.
     */
    public static final String CONJUNCTION = "CONJ";

    /**
     * The joiner between the field name and feature name.
     */
    public static final String JOINER = "@";

    private final String fieldName;

    private final String firstFieldName;

    private final String secondFieldName;

    private final String columnEntry;

    /**
     * Constructs a {@code ColumnarFeature} from the field name. The column entry is blank.
     * <p>
     * This produces a ColumnarFeature which is identical to a Feature.
     * @param fieldName The field name.
     * @param value The feature value.
     */
    public ColumnarFeature(String fieldName, double value) {
        super(fieldName,value);
        this.fieldName = fieldName;
        this.columnEntry = "";
        this.firstFieldName = "";
        this.secondFieldName = "";
    }

    /**
     * Constructs a {@code ColumnarFeature} from the field name, column entry and value.
     * @param fieldName The field name.
     * @param columnEntry The name of the extracted value from the field.
     * @param value The feature value.
     */
    public ColumnarFeature(String fieldName, String columnEntry, double value) {
        super(generateFeatureName(fieldName,columnEntry), value);
        this.fieldName = fieldName;
        this.columnEntry = columnEntry;
        this.firstFieldName = "";
        this.secondFieldName = "";
    }

    /**
     * Constructs a {@code ColumnarFeature} which is the conjunction of features from two fields.
     * @param firstFieldName The first field name.
     * @param secondFieldName The second field name.
     * @param columnEntry The name of the extracted value from the field.
     * @param value The feature value.
     */
    public ColumnarFeature(String firstFieldName, String secondFieldName, String columnEntry, double value) {
        super(generateFeatureName(firstFieldName,secondFieldName,columnEntry),value);
        this.fieldName = CONJUNCTION;
        this.columnEntry = columnEntry;
        this.firstFieldName = firstFieldName;
        this.secondFieldName = secondFieldName;
    }

    /**
     * Generates a feature name based on the field name and the name.
     * <p>
     * Uses {@link ColumnarFeature#JOINER} to join the strings.
     * @param fieldName The field name.
     * @param name The name of the extracted feature.
     * @return The new feature name.
     */
    public static String generateFeatureName(String fieldName, String name) {
        return fieldName + JOINER + name;
    }

    /**
     * Generates a feature name used for conjunction features.
     * <p>
     * Uses {@link ColumnarFeature#JOINER} to join the strings and {@link ColumnarFeature#CONJUNCTION} to prepend the name.
     * @param firstFieldName The name of the first field.
     * @param secondFieldName The name of the second field.
     * @param name The name of the extracted feature.
     * @return The new feature name.
     */
    public static String generateFeatureName(String firstFieldName, String secondFieldName, String name) {
        return CONJUNCTION + "[" + firstFieldName + "," + secondFieldName + "]" + JOINER + name;
    }

    /**
     * Gets the field name. Returns {@link ColumnarFeature#CONJUNCTION} if it's a conjunction.
     * @return The field name.
     */
    public String getFieldName() {
        return fieldName;
    }

    /**
     * If it's a conjunction feature, return the first field name.
     * Otherwise return an empty String.
     * @return The first field name, or an empty string.
     */
    public String getFirstFieldName() {
        return firstFieldName;
    }

    /**
     * If it's a conjunction feature, return the second field name.
     * Otherwise return an empty String.
     * @return The second field name, or an empty string.
     */
    public String getSecondFieldName() {
        return secondFieldName;
    }

    /**
     * Gets the columnEntry (i.e., the feature name produced by the {@link FieldExtractor}
     * without the fieldName).
     * @return The feature's column entry.
     */
    public String getColumnEntry() {
        return columnEntry;
    }
}
