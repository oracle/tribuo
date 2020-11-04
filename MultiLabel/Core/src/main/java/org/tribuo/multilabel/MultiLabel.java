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

package org.tribuo.multilabel;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.classification.Classifiable;
import org.tribuo.classification.Label;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * A class for multi-label classification.
 * <p>
 * Multi-label classification is where a (possibly empty) set of labels
 * is predicted for each example. For example, predicting that a Reuters
 * article has both the Finance and Sports labels.
 */
public class MultiLabel implements Classifiable<MultiLabel> {
    private static final long serialVersionUID = 1L;

    public static final String NEGATIVE_LABEL_STRING = "ML##NEGATIVE";
    /**
     * A Label representing the binary negative label. Used in binary
     * approaches to multi-label classification to represent the absence
     * of a Label.
     */
    public static final Label NEGATIVE_LABEL = new Label(NEGATIVE_LABEL_STRING);

    private final String label;
    private final double score;
    private final Set<Label> labels;
    private final Set<String> labelStrings;

    /**
     * Builds a MultiLabel object from a Set of Labels.
     *
     * Sets the whole set score to {@link Double#NaN}.
     * @param labels A set of (possibly scored) labels.
     */
    public MultiLabel(Set<Label> labels) {
        this(labels,Double.NaN);
    }

    /**
     * Builds a MultiLabel object from a Set of Labels,
     * when the whole set has a score as well as (optionally)
     * the individual labels.
     * @param labels A set of (possibly scored) labels.
     * @param score An overall score for the set.
     */
    public MultiLabel(Set<Label> labels, double score) {
        this.label = MultiLabelFactory.generateLabelString(labels);
        this.score = score;
        this.labels = Collections.unmodifiableSet(new HashSet<>(labels));
        Set<String> temp = new HashSet<>();
        for (Label l : labels) {
            temp.add(l.getLabel());
        }
        this.labelStrings = Collections.unmodifiableSet(temp);
    }

    /**
     * Builds a MultiLabel with a single String label.
     *
     * The created {@link Label} is unscored and used by MultiLabelInfo.
     *
     * Sets the whole set score to {@link Double#NaN}.
     * @param label The label.
     */
    public MultiLabel(String label) {
        this(new Label(label));
    }

    /**
     * Builds a MultiLabel from a single Label.
     *
     * Sets the whole set score to {@link Double#NaN}.
     * @param label The label.
     */
    public MultiLabel(Label label) {
        this(Collections.singleton(label));
    }

    /**
     * Creates a binary label from this multilabel.
     * The returned Label is the input parameter if
     * this MultiLabel contains that Label, and
     * {@link MultiLabel#NEGATIVE_LABEL} otherwise.
     * @param otherLabel The input label.
     * @return A binarised form of this MultiLabel.
     */
    public Label createLabel(Label otherLabel) {
        if (labelStrings.contains(otherLabel.getLabel())) {
            return otherLabel;
        } else {
            return NEGATIVE_LABEL;
        }
    }

    /**
     * Returns a comma separated string representing
     * the labels in this multilabel instance.
     * @return A comma separated string of labels.
     */
    public String getLabelString() {
        return label;
    }

    /**
     * The overall score for this set of labels.
     * @return The score for this MultiLabel.
     */
    public double getScore() {
        return score;
    }

    /**
     * The set of labels contained in this multilabel.
     * @return The set of labels.
     */
    public Set<Label> getLabelSet() {
        return new HashSet<>(labels);
    }

    /**
     * The set of strings that represent the labels in this multilabel.
     * @return The set of strings.
     */
    public Set<String> getNameSet() {
        return new HashSet<>(labelStrings);
    }

    /**
     * Does this MultiLabel contain this string?
     * @param input A string representing a {@link Label}.
     * @return True if the label string is in this MultiLabel.
     */
    public boolean contains(String input) {
        return labelStrings.contains(input);
    }

    /**
     * Does this MultiLabel contain this Label?
     * @param input A {@link Label}.
     * @return True if the label is in this MultiLabel.
     */
    public boolean contains(Label input) {
        return labels.contains(input);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        MultiLabel that = (MultiLabel) o;

        return labelStrings != null ? labelStrings.equals(that.labelStrings) : that.labelStrings == null;
    }

    @Override
    public boolean fullEquals(MultiLabel o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        if (Double.compare(score, o.score) != 0) {
            return false;
        }
        Map<String,Double> thisMap = new HashMap<>();
        for (Label l : labels) {
            thisMap.put(l.getLabel(),l.getScore());
        }
        Map<String,Double> thatMap = new HashMap<>();
        for (Label l : o.labels) {
            thatMap.put(l.getLabel(),l.getScore());
        }
        if (thisMap.size() == thatMap.size()) {
            for (Map.Entry<String,Double> e : thisMap.entrySet()) {
                Double thisValue = e.getValue();
                Double thatValue = thatMap.get(e.getKey());
                if ((thatValue == null) || Double.compare(thisValue,thatValue) != 0) {
                    return false;
                }
            }
            return true;
        } else {
            return false;
        }
    }

    @Override
    public int hashCode() {
        return Objects.hash(labelStrings);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();

        builder.append("(LabelSet={");
        for (Label l : labels) {
            builder.append(l.toString());
            builder.append(',');
        }
        builder.deleteCharAt(builder.length()-1);
        builder.append('}');
        if (!Double.isNaN(score)) {
            builder.append(",OverallScore=");
            builder.append(score);
        }
        builder.append(")");

        return builder.toString();
    }

    @Override
    public MultiLabel copy() {
        return new MultiLabel(labels,score);
    }

    /**
     * For a MultiLabel with label set = {a, b, c}, outputs a string of the form:
     * <pre>
     * "a=true,b=true,c=true"
     * </pre>
     * If includeConfidence is set to true, outputs a string of the form:
     * <pre>
     * "a=true,b=true,c=true:0.5"
     * </pre>
     * where the last element after the colon is this label's score.
     *
     * @param includeConfidence Include whatever confidence score the label contains, if known.
     * @return a comma-separated, densified string representation of this MultiLabel
     */
    @Override
    public String getSerializableForm(boolean includeConfidence) {
        /*
         * Note: Due to the sparse implementation of MultiLabel, all 'key=value' pairs will have value=true. That is,
         * say 'all possible labels' for a dataset are {R1,R2} but this particular example has label set = {R1}. Then
         * this method will output only "R1=true",whereas one might expect "R1=true,R2=false". Nevertheless, we generate
         * the 'serializable form' of this MultiLabel in this way to be consistent with that of other multi-output types
         * such as MultipleRegressor.
         */
        String str = labels.stream()
                .map(label -> String.format("%s=%b", label, true))
                .collect(Collectors.joining(","));
        if (includeConfidence) {
            return str + ":" + score;
        }
        return str;
    }

    /**
     * Parses a string of the form:
     * dimension-name=output,...,dimension-name=output
     * where output must be readable by {@link Boolean#parseBoolean(String)}.
     * @param s The string form of a multi-label example.
     * @return A {@link MultiLabel} parsed from the input string.
     */
    public static MultiLabel parseString(String s) {
        return parseString(s,',');
    }

    /**
     * Parses a string of the form:
     * <pre>
     * dimension-name=output&lt;splitChar&gt;...&lt;splitChar&gt;dimension-name=output
     * </pre>
     * where output must be readable by {@link Boolean#parseBoolean}.
     * @param s The string form of a multilabel output.
     * @param splitChar The char to split on.
     * @return A {@link MultiLabel} output parsed from the input string.
     */
    public static MultiLabel parseString(String s, char splitChar) {
        if (splitChar == '=') {
            throw new IllegalArgumentException("Can't split on an equals symbol");
        }
        String[] tokens = s.split(""+splitChar);
        List<Pair<String,Boolean>> pairs = new ArrayList<>();
        for (String token : tokens) {
            pairs.add(parseElement(token));
        }
        return createFromPairList(pairs);
    }

    /**
     * Parses a string of the form:
     *
     * <pre>
     *     class1=true
     * </pre>
     *
     * OR of the form:
     *
     * <pre>
     *     class1
     * </pre>
     *
     * In the first case, the value in the "key=value" pair must be parseable by {@link Boolean#parseBoolean(String)}.
     *
     * TODO: Boolean.parseBoolean("1") returns false. We may want to think more carefully about this case.
     *
     * @param s The string form of a single dimension from a multilabel input.
     * @return A tuple representing the dimension name and the value.
     */
    public static Pair<String,Boolean> parseElement(String s) {
        if (s.isEmpty()) {
            return new Pair<>("", false);
        }
        String[] split = s.split("=");
        if (split.length == 2) {
            //
            // Case: "Class1=TRUE,Class2=FALSE"
            return new Pair<>(split[0],Boolean.parseBoolean(split[1]));
        } else if (split.length == 1) {
            //
            // Case: "Class1,Class2"
            return new Pair<>(split[0], true);
        } else {
            throw new IllegalArgumentException("Failed to parse element " + s);
        }
    }

    /**
     * Creates a MultiLabel from a list of dimensions.
     * @param dimensions The dimensions to use.
     * @return A MultiLabel representing these dimensions.
     */
    public static MultiLabel createFromPairList(List<Pair<String,Boolean>> dimensions) {
        Set<Label> labels = new HashSet<>();
        for (int i = 0; i < dimensions.size(); i++) {
            Pair<String,Boolean> p = dimensions.get(i);
            String name = p.getA();
            boolean value = p.getB();
            if (value) {
                labels.add(new Label(name));
            }
        }
        return new MultiLabel(labels);
    }
}
