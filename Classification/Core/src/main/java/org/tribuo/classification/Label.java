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

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.Objects;
import org.tribuo.classification.protos.LabelProto;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.OutputProto;

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
@ProtoSerializableClass(serializedDataClass=LabelProto.class, version=0)
public final class Label implements Classifiable<Label> {
    private static final long serialVersionUID = 1L;

    /**
     * The name of the unknown label (i.e., an unlabelled output).
     */
    public static final String UNKNOWN = "LABEL##UNKNOWN";

    /**
     * The name of the label.
     */
    @ProtoSerializableField
    protected final String label;

    /**
     * The score of the label.
     */
    @ProtoSerializableField
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
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static Label deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        LabelProto proto = message.unpack(LabelProto.class);
        Label lbl = new Label(proto.getLabel(),proto.getScore());
        return lbl;
    }

    @Override
    public OutputProto serialize() {
        return ProtoUtil.serialize(this);
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

        return Objects.equals(label, that.label);
    }

    @Override
    public boolean fullEquals(Label o) {
        return fullEquals(o, 0.0);
    }

    @Override
    public boolean fullEquals(Label o, double tolerance) {
        if (this == o) return true;
        if (o == null) return false;

        if (Double.isNaN(o.score) ^ Double.isNaN(score)) {
            return false;
        }
        if (Math.abs(o.score - score) > tolerance) {
            return false;
        }
        return Objects.equals(label, o.label);
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
