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
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableOutputInfo;
import org.tribuo.OutputFactory;
import org.tribuo.evaluation.Evaluation;
import org.tribuo.evaluation.Evaluator;
import org.tribuo.protos.core.OutputFactoryProto;
import org.tribuo.provenance.OutputFactoryProvenance;

import java.util.Map;

/**
 * An OutputFactory for use in tests, very similar to LabelFactory.
 */
public class MockOutputFactory implements OutputFactory<MockOutput> {

    public static final MockOutput UNKNOWN_TEST_OUTPUT = new MockOutput("UNKNOWN");

    public MockOutputFactory() {}

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @return The deserialized object.
     */
    public static MockOutputFactory deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new MockOutputFactory();
    }

    @Override
    public OutputFactoryProto serialize() {
        return OutputFactoryProto.newBuilder().setVersion(0).setClassName(MockOutputFactory.class.getName()).build();
    }

    @Override
    public <V> MockOutput generateOutput(V label) {
        return new MockOutput(label.toString());
    }

    @Override
    public MockOutput getUnknownOutput() {
        return UNKNOWN_TEST_OUTPUT;
    }

    @Override
    public MutableOutputInfo<MockOutput> generateInfo() {
        return new MockOutputInfo();
    }

    @Override
    public ImmutableOutputInfo<MockOutput> constructInfoForExternalModel(Map<MockOutput, Integer> mapping) {
        throw new UnsupportedOperationException("constructInfoForExternalModel not implemented");
    }

    @Override
    public Evaluator<MockOutput, ? extends Evaluation<MockOutput>> getEvaluator() {
        throw new UnsupportedOperationException("generateEvaluator not implemented");
    }

    @Override
    public Class<MockOutput> getTypeWitness() {
        return MockOutput.class;
    }

    @Override
    public int hashCode() {
        return "MockOutputFactory".hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof MockOutputFactory;
    }

    @Override
    public OutputFactoryProvenance getProvenance() {
        return new TestOutputFactoryProvenance();
    }

    public static class TestOutputFactoryProvenance implements OutputFactoryProvenance {
        private static final long serialVersionUID = 1L;

        TestOutputFactoryProvenance() {}

        public TestOutputFactoryProvenance(Map<String, Provenance> map) { }

        @Override
        public String getClassName() {
            return MockOutputFactory.class.getName();
        }
        @Override
        public String toString() {
            return generateString("MockOutputFactory");
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof TestOutputFactoryProvenance;
        }

        @Override
        public int hashCode() {
            return 31;
        }
    }
}
