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

package org.tribuo.classification.evaluation;

import org.tribuo.DataSource;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.MutableDataset;
import org.tribuo.OutputFactory;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.provenance.DataSourceProvenance;
import org.tribuo.provenance.SimpleDataSourceProvenance;

import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

public class TestUtils {

    static Dataset<Label> mkDataset(List<Prediction<Label>> predictions) {
        List<Example<Label>> examples = predictions
                .stream().map(Prediction::getExample)
                .collect(Collectors.toList());
        DataSource<Label> src = new DataSource<Label>() {
            @Override public OutputFactory<Label> getOutputFactory() { return new LabelFactory(); }
            @Override public DataSourceProvenance getProvenance() { return new SimpleDataSourceProvenance("", getOutputFactory()); }
            @Override public Iterator<Example<Label>> iterator() { return examples.iterator(); }
        };
        return new MutableDataset<>(src);
    }

}