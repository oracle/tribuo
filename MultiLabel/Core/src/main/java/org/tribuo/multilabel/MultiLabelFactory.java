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
import com.google.protobuf.ByteString;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableOutputInfo;
import org.tribuo.OutputFactory;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.evaluation.Evaluator;
import org.tribuo.multilabel.evaluation.MultiLabelEvaluation;
import org.tribuo.multilabel.evaluation.MultiLabelEvaluator;
import org.tribuo.protos.core.OutputFactoryProto;
import org.tribuo.provenance.OutputFactoryProvenance;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A factory for generating MultiLabel objects and their associated OutputInfo and Evaluator objects.
 */
public final class MultiLabelFactory implements OutputFactory<MultiLabel> {
    private static final long serialVersionUID = 1L;

    /**
     * The sentinel unknown multi-label output used to signal there is no ground truth value.
     */
    public static final MultiLabel UNKNOWN_MULTILABEL = new MultiLabel(LabelFactory.UNKNOWN_LABEL);

    private static final MultiLabelFactoryProvenance provenance = new MultiLabelFactoryProvenance();

    private static final MultiLabelEvaluator evaluator = new MultiLabelEvaluator();

    /**
     * Construct a MultiLabelFactory.
     */
    public MultiLabelFactory() {}

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @return The deserialized object.
     */
    public static MultiLabelFactory deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new MultiLabelFactory();
    }

    @Override
    public OutputFactoryProto serialize() {
        return OutputFactoryProto.newBuilder().setVersion(0).setClassName(MultiLabelFactory.class.getName()).build();
    }

    /**
     * Parses the MultiLabel value either by toStringing the input and calling {@link MultiLabel#parseString}
     * or if it's a {@link Collection} iterating over the elements calling toString on each element in turn and using
     * {@link MultiLabel#parseElement}.
     * @param label An input value.
     * @param <V> The type of the input value.
     * @return A MultiLabel
     */
    @Override
    public <V> MultiLabel generateOutput(V label) {
        if (label instanceof Collection) {
            Collection<?> c = (Collection<?>) label;
            List<Pair<String,Boolean>> dimensions = new ArrayList<>();
            for (Object o : c) {
                dimensions.add(MultiLabel.parseElement(o.toString()));
            }
            return MultiLabel.createFromPairList(dimensions);
        }
        return MultiLabel.parseString(label.toString());
    }

    @Override
    public MultiLabel getUnknownOutput() {
        return UNKNOWN_MULTILABEL;
    }

    @Override
    public MutableOutputInfo<MultiLabel> generateInfo() {
        return new MutableMultiLabelInfo();
    }

    @Override
    public ImmutableOutputInfo<MultiLabel> constructInfoForExternalModel(Map<MultiLabel,Integer> mapping) {
        // Validate inputs are dense
        OutputFactory.validateMapping(mapping);

        MutableMultiLabelInfo info = new MutableMultiLabelInfo();

        for (Map.Entry<MultiLabel,Integer> e : mapping.entrySet()) {
            info.observe(e.getKey());
        }

        return new ImmutableMultiLabelInfo(info,mapping);
    }

    @Override
    public Evaluator<MultiLabel, MultiLabelEvaluation> getEvaluator() {
        return evaluator;
    }

    @Override
    public Class<MultiLabel> getTypeWitness() {
        return MultiLabel.class;
    }

    @Override
    public int hashCode() {
        return "MultiLabelFactory".hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof MultiLabelFactory;
    }

    @Override
    public OutputFactoryProvenance getProvenance() {
        return provenance;
    }

    /**
     * Generates a comma separated string of labels from a {@link Set} of {@link Label}.
     * @param input A Set of Label objects.
     * @return A (possibly empty) comma separated string.
     */
    public static String generateLabelString(Set<Label> input) {
        if (input.isEmpty()) {
            return "";
        }
        List<String> list = new ArrayList<>();
        for (Label l : input) {
            list.add(l.getLabel());
        }
        list.sort(String::compareTo);

        StringBuilder builder = new StringBuilder();
        for (String s : list) {
            if (s.contains(",")) {
                throw new IllegalStateException("MultiLabel cannot contain a label with a ',', found " + s + ".");
            }
            builder.append(s);
            builder.append(',');
        }
        builder.deleteCharAt(builder.length()-1);
        return builder.toString();
    }

    /**
     * Provenance for {@link MultiLabelFactory}.
     */
    public final static class MultiLabelFactoryProvenance implements OutputFactoryProvenance {
        private static final long serialVersionUID = 1L;

        /**
         * Constructs a multi-label factory provenance.
         */
        MultiLabelFactoryProvenance() {}

        /**
         * Constructs a multi-label factory provenance from the empty marshalled form.
         * @param map An empty map.
         */
        public MultiLabelFactoryProvenance(Map<String, Provenance> map) { }

        @Override
        public String getClassName() {
            return MultiLabelFactory.class.getName();
        }

        @Override
        public String toString() {
            return generateString("OutputFactory");
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof MultiLabelFactoryProvenance;
        }

        @Override
        public int hashCode() {
            return 31;
        }
    }
}
