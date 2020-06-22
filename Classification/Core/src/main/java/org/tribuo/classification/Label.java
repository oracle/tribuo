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

/**
 * An immutable multi-class classification label.
 * <p>
 * The labels themselves are Strings. A Label also contains an
 * optional score which measures the confidence in that label,
 * though it is not required to be a probability.
 * <p>
 * Label equality and hashCode is defined solely on the String
 * label, it does not take into account the score.
 */
public final class Label implements Classifiable<Label> {
    private static final long serialVersionUID = 1L;

    public static final String UNKNOWN = "LABEL##UNKNOWN";

    protected final String label;

    protected final double score;

    /**
     * Builds a label with the supplied string and score.
     * @param label The label name.
     * @param score The label instance score.
     */
    public Label(String label, double score) {
        this.label = label;
        this.score = score;
    }

    /**
     * Builds a label with the sentinel score of Double.NaN.
     * @param label The name of this label.
     */
    public Label(String label) {
        this(label,Double.NaN);
    }

    /**
     * Get a real valued score for this label.
     * <p>
     * If the score is not set then it returns Double.NaN.
     * @return The predicted score for this label.
     */
    public double getScore() {
        return score;
    }

    /**
     * Gets the name of this label.
     * @return A String.
     */
    public String getLabel() {
        return label;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Label)) return false;

        Label that = (Label) o;

        return label != null ? label.equals(that.label) : that.label == null;
    }

    @Override
    public boolean fullEquals(Label o) {
        if (this == o) return true;
        if (o == null) return false;

        if ((!(Double.isNaN(o.score) && Double.isNaN(score))) && (Double.compare(o.score, score) != 0)) return false;
        return label != null ? label.equals(o.label) : o.label == null;
    }

    @Override
    public int hashCode() {
        int result;
        result = label.hashCode();
        return result;
    }

    @Override
    public String toString() {
        if (Double.isNaN(score)) {
            return label;
        } else {
            return "("+label+","+score+")";
        }
    }

    @Override
    public Label copy() {
        return new Label(label,score);
    }

    /**
     * Returns "labelName" or "labelName,score=labelScore".
     * @param includeConfidence Include whatever confidence score the label contains, if known.
     * @return A String form suitable for serialization.
     */
    @Override
    public String getSerializableForm(boolean includeConfidence) {
        if (includeConfidence && !Double.isNaN(score)) {
            return label + ",score=" + score;
        } else {
            return label;
        }
    }
}
