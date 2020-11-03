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

import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.impl.ListExample;

import java.util.Arrays;
import java.util.List;

public class Utils {

    private static final MultiLabelFactory outputFactory = new MultiLabelFactory();

    public static MultiLabel getUnknown() {
        return outputFactory.getUnknownOutput();
    }

    public static MultiLabel label(String... values) {
        String csv = String.join(",", Arrays.asList(values));
        MultiLabel output = outputFactory.generateOutput(csv);
        return output;
    }

    public static Prediction<MultiLabel> mkPrediction(MultiLabel trueVal, MultiLabel predVal) {
        Example<MultiLabel> example = new ListExample<>(trueVal);
        example.add(new Feature("noop", 1d));
        return new Prediction<>(predVal, 0, example);
    }

    public static ImmutableOutputInfo<MultiLabel> mkDomain(MultiLabel... values) {
        MutableMultiLabelInfo info = new MutableMultiLabelInfo();
        for (MultiLabel value : values) {
            info.observe(value);
        }
        return info.generateImmutableOutputInfo();
    }

    public static ImmutableOutputInfo<MultiLabel> mkDomain(List<Prediction<MultiLabel>> predictions) {
        MutableMultiLabelInfo info = new MutableMultiLabelInfo();
        for (Prediction<MultiLabel> p : predictions) {
            info.observe(p.getExample().getOutput());
        }
        return info.generateImmutableOutputInfo();
    }

}