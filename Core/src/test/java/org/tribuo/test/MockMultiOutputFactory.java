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
import com.google.protobuf.ByteString;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableOutputInfo;
import org.tribuo.OutputFactory;
import org.tribuo.evaluation.Evaluation;
import org.tribuo.evaluation.Evaluator;
import org.tribuo.protos.core.OutputFactoryProto;
import org.tribuo.provenance.OutputFactoryProvenance;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class MockMultiOutputFactory implements OutputFactory<MockMultiOutput> {

    private static final long serialVersionUID = 1L;

    public static final MockMultiOutput UNKNOWN_MULTILABEL = new MockMultiOutput("unk");

    public MockMultiOutputFactory() {}

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @return The deserialized object.
     */
    public static MockMultiOutputFactory deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new MockMultiOutputFactory();
    }

    @Override
    public OutputFactoryProto serialize() {
        return OutputFactoryProto.newBuilder().setVersion(0).setClassName(MockMultiOutputFactory.class.getName()).build();
    }

    @Override
    public <V> MockMultiOutput generateOutput(V label) {
        if (label instanceof Collection) {
            Collection<?> c = (Collection<?>) label;
            List<Pair<String,Boolean>> dimensions = new ArrayList<>();
            for (Object o : c) {
                dimensions.add(parseElement(o.toString()));
            }
            return createFromPairList(dimensions);
        }
        return parseString(label.toString());
    }

    @Override
    public MockMultiOutput getUnknownOutput() {
        return UNKNOWN_MULTILABEL;
    }

    @Override
    public MutableOutputInfo<MockMultiOutput> generateInfo() {
        return new MockMultiOutputInfo();
    }

    @Override
    public ImmutableOutputInfo<MockMultiOutput> constructInfoForExternalModel(Map<MockMultiOutput, Integer> mapping) {
        throw new UnsupportedOperationException("constructInfoForExternalModel not implemented");
    }

    @Override
    public Evaluator<MockMultiOutput, ? extends Evaluation<MockMultiOutput>> getEvaluator() {
        throw new UnsupportedOperationException("generateEvaluator not implemented");
    }

    @Override
    public Class<MockMultiOutput> getTypeWitness() {
        return MockMultiOutput.class;
    }

    @Override
    public int hashCode() {
        return this.getClass().hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof MockMultiOutputFactory;
    }

    @Override
    public OutputFactoryProvenance getProvenance() {
        return new MockMultiOutputFactoryProvenance();
    }

    /**
     * Parses a string of the form:
     * dimension-name=output,...,dimension-name=output
     * where output must be readable by {@link Double#parseDouble}.
     * @param s The string form of a multiple regressor.
     * @return A multiple regressor parsed from the input string.
     */
    public static MockMultiOutput parseString(String s) {
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
     * @return A multilabel output parsed from the input string.
     */
    public static MockMultiOutput parseString(String s, char splitChar) {
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
     * <pre>
     * dimension-name=output-boolean (e.g., "Class1=FALSE")
     * </pre>
     * where the output must be readable by {@link Double#parseDouble}.
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
     * Creates a MultipleRegressor from a list of dimension tuples.
     * @param dimensions The dimensions to use.
     * @return A MultipleRegressor representing these dimensions.
     */
    public static MockMultiOutput createFromPairList(List<Pair<String,Boolean>> dimensions) {
        Set<MockOutput> labels = new HashSet<>();
        for (Pair<String, Boolean> p : dimensions) {
            String name = p.getA();
            boolean value = p.getB();
            if (value) {
                labels.add(new MockOutput(name));
            }
        }
        return new MockMultiOutput(labels);
    }

    public static class MockMultiOutputFactoryProvenance implements OutputFactoryProvenance {
        private static final long serialVersionUID=1L;

        MockMultiOutputFactoryProvenance() {}

        public MockMultiOutputFactoryProvenance(Map<String, Provenance> map) {}

        @Override
        public String getClassName() {
            return MockMultiOutputFactory.class.getName();
        }

        @Override
        public String toString() {
            return generateString("MockMultiOutputFactory");
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof MockMultiOutputFactoryProvenance;
        }

        @Override
        public int hashCode() {
            return 32;
        }
    }
}