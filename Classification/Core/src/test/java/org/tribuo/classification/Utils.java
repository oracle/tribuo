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

import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.impl.ListExample;

import java.util.List;

public class Utils {

    private static final LabelFactory factory = new LabelFactory();

    public static Label label(String v) {
        return factory.generateOutput(v);
    }

    public static Prediction<Label> mkPrediction(String trueVal, String predVal) {
        LabelFactory factory = new LabelFactory();
        Example<Label> example = new ListExample<>(factory.generateOutput(trueVal));
        example.add(new Feature("noop", 1d));
        Prediction<Label> prediction = new Prediction<>(factory.generateOutput(predVal), 0, example);
        return prediction;
    }

    public static ImmutableOutputInfo<Label> mkDomain(List<Prediction<Label>> predictions) {
        MutableLabelInfo info = new MutableLabelInfo();
        for (Prediction<Label> p : predictions) {
            info.observe(p.getExample().getOutput());
        }
        return info.generateImmutableOutputInfo();
    }

}