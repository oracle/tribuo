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

package org.tribuo.test;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tribuo.Output;
import org.tribuo.protos.core.OutputProto;
import org.tribuo.test.protos.MockMultiOutputProto;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class MockMultiOutput implements Output<MockMultiOutput> {

    private static final long serialVersionUID = 1L;

    public static final String NEGATIVE_LABEL_STRING = "ML##NEGATIVE";
    /**
     * A MockOutput representing the binary negative label. Used in binary
     * approaches to multi-label classification to represent the absence
     * of a Label.
     */
    public static final MockOutput NEGATIVE_LABEL = new MockOutput(NEGATIVE_LABEL_STRING);

    private final String label;
    private final double score;
    private final Set<MockOutput> labels;
    private final Set<String> labelStrings;

    /**
     * Builds a multilabel object from a Set of Labels.
     * <p>
     * Assumes the model does not provide a score for the whole
     * set.
     * @param labels A set of (possibly scored) labels.
     */
    public MockMultiOutput(Set<MockOutput> labels) {
        this(labels,Double.NaN);
    }

    /**
     * Builds a multilabel object from a Set of Labels,
     * when the whole set has a score as well as (optionally)
     * the individual labels.
     * @param labels A set of (possibly scored) labels.
     * @param score An overall score for the set.
     */
    public MockMultiOutput(Set<MockOutput> labels, double score) {
        this.label = generateLabelString(labels);
        this.score = score;
        this.labels = Collections.unmodifiableSet(new HashSet<>(labels));
        Set<String> temp = new HashSet<>();
        for (MockOutput l : labels) {
            temp.add(l.label);
        }
        this.labelStrings = Collections.unmodifiableSet(temp);
    }

    /**
     * Builds a multilabel with a single String label.
     * <p>
     * This is unscored and used by MockMultiOutputInfo.
     * @param label The label.
     */
    public MockMultiOutput(String label) {
        this(new MockOutput(label));
    }

    /**
     * Builds a multilabel from a single Label.
     * @param label The label.
     */
    public MockMultiOutput(MockOutput label) {
        this(Collections.singleton(label));
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static MockMultiOutput deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        MockMultiOutputProto proto = message.unpack(MockMultiOutputProto.class);
        Set<MockOutput> labels = new HashSet<>();
        for (String s : proto.getLabelList()) {
            labels.add(new MockOutput(s));
        }
        MockMultiOutput info = new MockMultiOutput(labels, proto.getScore());
        return info;
    }

    @Override
    public OutputProto serialize() {
        MockMultiOutputProto.Builder builder = MockMultiOutputProto.newBuilder();
        for (MockOutput m : labels) {
            builder.addLabel(m.label);
        }
        builder.setScore(score);
        return OutputProto.newBuilder().setVersion(0).setClassName(this.getClass().getName()).setSerializedData(Any.pack(builder.build())).build();
    }

    /**
     * Creates a binary label from this multilabel.
     * The returned MockOutput is the input parameter if
     * this MultiMockOutput contains that Label, and
     * {@link MockMultiOutput#NEGATIVE_LABEL} otherwise.
     * @param otherLabel The input label.
     * @return A binarised form of this MockMultiOutput.
     */
    public MockOutput createLabel(MockOutput otherLabel) {
        if (labelStrings.contains(otherLabel.label)) {
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
     * @return The score for this MockMultiOutput.
     */
    public double getScore() {
        return score;
    }

    /**
     * The set of labels contained in this multilabel.
     * @return The set of labels.
     */
    public Set<MockOutput> getLabelSet() {
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
     * Does this MultiMockOutput contain this string?
     * @param input A string representing a {@link MockOutput}.
     * @return True if the label string is in this MockMultiOutput.
     */
    public boolean contains(String input) {
        return labelStrings.contains(input);
    }

    /**
     * Does this MultiMockOutput contain this Label?
     * @param input A {@link MockOutput}.
     * @return True if the label is in this MockMultiOutput.
     */
    public boolean contains(MockOutput input) {
        return labels.contains(input);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        MockMultiOutput that = (MockMultiOutput) o;

        return labelStrings != null ? labelStrings.equals(that.labelStrings) : that.labelStrings == null;
    }

    @Override
    public boolean fullEquals(MockMultiOutput o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Set<String> thisLabels = new HashSet<>();
        for (MockOutput l : labels) {
            thisLabels.add(l.label);
        }

        Set<String> thatLabels = new HashSet<>();
        for (MockOutput l : o.labels) {
            thatLabels.add(l.label);
        }

        return thisLabels.equals(thatLabels);
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (labels != null ? labels.hashCode() : 0);
        return result;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();

        builder.append("(LabelSet={");
        for (MockOutput l : labels) {
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
    public MockMultiOutput copy() {
        return new MockMultiOutput(labels,score);
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
        // TODO this is not a correct dense output though. Say 'all possible labels' for a dataset
        // are {R1,R2} but this example has label set {R1}. Then this method will ouput only "R1=true",
        // whereas we might expect "R1=true,R2=false".
        String str = labels.stream()
                .map(label -> String.format("%s=%b", label, true))
                .collect(Collectors.joining(","));
        if (includeConfidence) {
            return str + ":" + score;
        }
        return str;
    }


    public static String generateLabelString(Set<MockOutput> input) {
        if (input.isEmpty()) {
            return "";
        }
        List<String> list = new ArrayList<>();
        for (MockOutput l : input) {
            list.add(l.label);
        }
        list.sort(String::compareTo);

        StringBuilder builder = new StringBuilder();
        for (String s : list) {
            if (s.contains(",")) {
                throw new IllegalStateException("MultiMockOutput cannot contain a label with a ',', found " + s + ".");
            }
            builder.append(s);
            builder.append(',');
        }
        builder.deleteCharAt(builder.length()-1);
        return builder.toString();
    }
}