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

import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableOutputInfo;
import org.tribuo.OutputFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.evaluation.Evaluator;
import org.tribuo.provenance.OutputFactoryProvenance;

import java.util.Map;

/**
 * A factory for making Label related classes.
 * <p>
 * Parses the Label by calling toString on the input.
 * <p>
 * Label factories have no state, and are all equal to each other.
 */
public final class LabelFactory implements OutputFactory<Label> {
    private static final long serialVersionUID = 1L;

    /**
     * The singleton unknown label, used for unlablled examples.
     */
    public static final Label UNKNOWN_LABEL = new Label(Label.UNKNOWN);

    private static final OutputFactoryProvenance provenance = new LabelFactoryProvenance();

    private static final LabelEvaluator evaluator = new LabelEvaluator();

    /**
     * Constructs a label factory.
     */
    public LabelFactory() {}

    /**
     * Generates the Label string by calling toString
     * on the input.
     * @param label An input value.
     * @param <V> The type of the input.
     * @return A Label object.
     */
    @Override
    public <V> Label generateOutput(V label) {
        return new Label(label.toString());
    }

    @Override
    public Label getUnknownOutput() {
        return UNKNOWN_LABEL;
    }

    /**
     * Generates an empty MutableLabelInfo.
     * @return An empty MutableLabelInfo.
     */
    @Override
    public MutableOutputInfo<Label> generateInfo() {
        return new MutableLabelInfo();
    }

    @Override
    public ImmutableOutputInfo<Label> constructInfoForExternalModel(Map<Label,Integer> mapping) {
        // Validate inputs are dense
        OutputFactory.validateMapping(mapping);

        MutableLabelInfo info = new MutableLabelInfo();

        for (Map.Entry<Label,Integer> e : mapping.entrySet()) {
            info.observe(e.getKey());
        }

        return new ImmutableLabelInfo(info,mapping);
    }

    @Override
    public Evaluator<Label,LabelEvaluation> getEvaluator() {
        return evaluator;
    }

    @Override
    public int hashCode() {
        return "LabelFactory".hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof LabelFactory;
    }

    @Override
    public OutputFactoryProvenance getProvenance() {
        return provenance;
    }

    /**
     * Provenance for {@link LabelFactory}.
     */
    public final static class LabelFactoryProvenance implements OutputFactoryProvenance {
        private static final long serialVersionUID = 1L;

        LabelFactoryProvenance() {}

        /**
         * Constructor used by the provenance serialization system.
         * <p>
         * As the label factory has no state, the argument is expected to be empty, and it's contents are ignored.
         * @param map The provenance map to use.
         */
        public LabelFactoryProvenance(Map<String, Provenance> map) { }

        @Override
        public String getClassName() {
            return LabelFactory.class.getName();
        }

        @Override
        public String toString() {
            return generateString("OutputFactory");
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof LabelFactoryProvenance;
        }

        @Override
        public int hashCode() {
            return 31;
        }
    }
}
