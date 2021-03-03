/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.sequence;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.provenance.ModelProvenance;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * A SequenceModel which independently predicts each element of the sequence.
 * @param <T> The output type.
 */
public class IndependentSequenceModel<T extends Output<T>> extends SequenceModel<T> {
    private static final Logger logger = Logger.getLogger(IndependentSequenceModel.class.getName());
    private static final long serialVersionUID = 1L;

    private final Model<T> model;

    IndependentSequenceModel(String name, ModelProvenance description, Model<T> model) {
        super(name, description, model.getFeatureIDMap(), model.getOutputIDInfo());
        this.model = model;
    }

    @Override
    public List<Prediction<T>> predict(SequenceExample<T> example) {
        List<Prediction<T>> output = new ArrayList<>();
        for (Example<T> e : example) {
            output.add(model.predict(e));
        }
        return output;
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return model.getTopFeatures(n);
    }
}
