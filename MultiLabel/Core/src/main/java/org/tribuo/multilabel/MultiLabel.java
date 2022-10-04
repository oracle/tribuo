/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.classification.Classifiable;
import org.tribuo.classification.Label;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.multilabel.protos.MultiLabelProto;
import org.tribuo.protos.core.OutputProto;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalDouble;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * A class for multi-label classification.
 * <p>
 * Multi-label classification is where a (possibly empty) set of labels
 * is predicted for each example. For example, predicting that a Reuters
 * article has both the Finance and Sports labels.
 * <p>
 * Both the labels in the set, and the MultiLabel itself may have optional
 * scores (which are not required to be probabilities). If the scores are
 * not present these are represented by {@link Double#NaN}. This is most
 * common with ground-truth labels which usually do not supply scores.
 */
public class MultiLabel implements Classifiable<MultiLabel> {
    private static final long serialVersionUID = 1L;

    /**
     * The string for the binary negative label.
     */
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
     * <p>
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
        Set<String> temp = new HashSet<>(labels.size());
        for (Label l : labels) {
            temp.add(l.getLabel());
        }
        this.labelStrings = Collections.unmodifiableSet(temp);
    }

    /**
     * Builds a MultiLabel with a single String label.
     * <p>
     * The created {@link Label} is unscored and used by MultiLabelInfo.
     * <p>
     * Sets the whole set score to {@link Double#NaN}.
     * @param label The label.
     */
    public MultiLabel(String label) {
        this(new Label(label));
    }

    /**
     * Builds a MultiLabel from a single Label.
     * <p>
     * Sets the whole set score to {@link Double#NaN}.
     * @param label The label.
     */
    public MultiLabel(Label label) {
        this.label = label.getLabel();
        this.score = Double.NaN;
        this.labels = Collections.singleton(label);
        this.labelStrings = Collections.singleton(label.getLabel());
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static MultiLabel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        MultiLabelProto proto = message.unpack(MultiLabelProto.class);
        if (proto.getLabelCount() != proto.getLblScoreCount()) {
            throw new IllegalArgumentException("Invalid protobuf, must have equal counts of labels and scores, labels " + proto.getLabelCount() + ", scores " + proto.getLblScoreCount());
        }
        Set<String> strings = new HashSet<>();
        Set<Label> lblSet = new HashSet<>();
        for (int i = 0; i < proto.getLabelCount(); i++) {
            String lbl = proto.getLabel(i);
            if (strings.contains(lbl)) {
                throw new IllegalArgumentException("Invalid protobuf, multiple entries for label '" + lbl + "'");
            } else {
                strings.add(lbl);
                double score = proto.getLblScore(i);
                lblSet.add(new Label(lbl,score));
            }
        }
        return new MultiLabel(lblSet,proto.getOverallScore());
    }

    @Override
    public OutputProto serialize() {
        OutputProto.Builder outputBuilder = OutputProto.newBuilder();

        outputBuilder.setClassName(MultiLabel.class.getName());
        outputBuilder.setVersion(0);

        MultiLabelProto.Builder data = MultiLabelProto.newBuilder();
        data.setOverallScore(score);
        for (Label l : labels) {
            data.addLabel(l.getLabel());
            data.addLblScore(l.getScore());
        }

        outputBuilder.setSerializedData(Any.pack(data.build()));

        return outputBuilder.build();
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
     * The score for the specified label if present, returns an empty optional otherwise.
     * @param label The label to check.
     * @return The score for the label if present.
     */
    public OptionalDouble getLabelScore(Label label) {
        Label scored = null;
        for (Label l : labels) {
            if (l.getLabel().equals(label.getLabel())) {
                scored = l;
            }
        }
        if (scored != null) {
            return OptionalDouble.of(scored.getScore());
        } else {
            return OptionalDouble.empty();
        }
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
        return fullEquals(o, 0.0);
    }

    @Override
    public boolean fullEquals(MultiLabel o, double tolerance) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        if (Math.abs(o.score - score) > tolerance) {
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
                if ((thatValue == null) || Math.abs(thisValue - thatValue) > tolerance) {
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
     * Converts this MultiLabel into a DenseVector using the indices from the output info.
     * The label score is used as the value for that index if it's non-NaN, and is 1.0 otherwise.
     * Labels which are not present are given the score 0.0.
     * @param info The info to use for the ids.
     * @return A DenseVector representing this MultiLabel.
     */
    public DenseVector convertToDenseVector(ImmutableOutputInfo<MultiLabel> info) {
        if (!(info instanceof ImmutableMultiLabelInfo)) {
            throw new IllegalStateException("Unexpected info type, found " + info.getClass().getName() + ", expected " + ImmutableMultiLabelInfo.class.getName());
        } else {
            ImmutableMultiLabelInfo imInfo = (ImmutableMultiLabelInfo) info;
            Set<Integer> seenIndices = new HashSet<>(labels.size());
            double[] values = new double[imInfo.size()];

            for (Label l : labels) {
                int i = imInfo.getID(l.getLabel());
                if (i != -1) {
                    if (!seenIndices.contains(i)) {
                        double score = l.getScore();
                        // A NaN score means this label was constructed without one, meaning it's a ground truth label.
                        if (Double.isNaN(score)) {
                            score = 1.0;
                        }
                        seenIndices.add(i);
                        values[i] = score;
                    } else {
                        throw new IllegalArgumentException("Duplicate label ids found for id " + i + ", mapping to Label '" + l.getLabel() + "'");
                    }
                } else {
                    throw new IllegalArgumentException("Unknown label '" + l.getLabel() + "' which was not recognised by the supplied info object, info = " + info.toString());
                }
            }

            return DenseVector.createDenseVector(values);
        }
    }

    /**
     * Converts this MultiLabel into a SparseVector using the indices from the output info.
     * The label score is used as the value for that index if it's non-NaN, and is 1.0 otherwise.
     * @param info The info to use for the ids.
     * @return A SparseVector representing this MultiLabel.
     */
    public SparseVector convertToSparseVector(ImmutableOutputInfo<MultiLabel> info) {
        if (!(info instanceof ImmutableMultiLabelInfo)) {
            throw new IllegalStateException("Unexpected info type, found " + info.getClass().getName() + ", expected " + ImmutableMultiLabelInfo.class.getName());
        } else {
            ImmutableMultiLabelInfo imInfo = (ImmutableMultiLabelInfo) info;
            Map<Integer, Double> values = new HashMap<>();

            for (Label l : labels) {
                int i = imInfo.getID(l.getLabel());
                if (i != -1) {
                    double score = l.getScore();
                    // A NaN score means this label was constructed without one, meaning it's a ground truth label.
                    if (Double.isNaN(score)) {
                        score = 1.0;
                    }
                    Double t = values.put(i,score);
                    if (t != null) {
                        throw new IllegalArgumentException("Duplicate label ids found for id " + i + ", mapping to Label '" + l.getLabel() + "'");
                    }
                } else {
                    throw new IllegalArgumentException("Unknown label '" + l.getLabel() + "' which was not recognised by the supplied info object, info = " + info.toString());
                }
            }

            return SparseVector.createSparseVector(info.size(), values);
        }
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

    /**
     * The number of labels present in both MultiLabels.
     * @param first The first MultiLabel.
     * @param second The second MultiLabel.
     * @return The set intersection size.
     */
    public static int intersectionSize(MultiLabel first, MultiLabel second) {
        HashSet<String> intersection = new HashSet<>();

        intersection.addAll(first.labelStrings);
        intersection.retainAll(second.labelStrings);

        return intersection.size();
    }

    /**
     * The number of unique labels across both MultiLabels.
     * @param first The first MultiLabel.
     * @param second The second MultiLabel.
     * @return The set union size.
     */
    public static int unionSize(MultiLabel first, MultiLabel second) {
        HashSet<String> union = new HashSet<>();

        union.addAll(first.labelStrings);
        union.addAll(second.labelStrings);

        return union.size();
    }

    /**
     * The Jaccard score/index between the two MultiLabels.
     * @param first The first MultiLabel.
     * @param second The second MultiLabel.
     * @return The Jaccard score.
     */
    public static double jaccardScore(MultiLabel first, MultiLabel second) {
        return ((double) intersectionSize(first,second)) / unionSize(first,second);
    }
}
