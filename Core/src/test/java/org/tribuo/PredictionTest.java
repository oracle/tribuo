/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo;

import org.junit.jupiter.api.Test;
import org.tribuo.impl.ArrayExample;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.PredictionProto;
import org.tribuo.test.Helpers;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class PredictionTest {

    @Test
    public void load431Protobufs() throws URISyntaxException, IOException {
        // Comparison objects
        MockOutput output = new MockOutput("foo");
        HashMap<String, MockOutput> scores = new HashMap<>();
        scores.put("foo", output);
        scores.put("bar", new MockOutput("bar"));
        ArrayExample<MockOutput> example = new ArrayExample<>(MockOutputFactory.UNKNOWN_TEST_OUTPUT, new String[]{"a", "b", "c"}, new double[]{1,2,3,});
        Prediction<MockOutput> prediction = new Prediction<>(output, scores, 3, example, false);

        Path predictionPath = Paths.get(PredictionTest.class.getResource("prediction-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(predictionPath)) {
            PredictionProto proto = PredictionProto.parseFrom(fis);
            Prediction<MockOutput> newPred = ProtoUtil.deserialize(proto);
            assertTrue(prediction.distributionEquals(newPred));
        }
    }

    public void generateProtobufs() throws IOException {
        MockOutput output = new MockOutput("foo");
        HashMap<String, MockOutput> scores = new HashMap<>();
        scores.put("foo", output);
        scores.put("bar", new MockOutput("bar"));
        ArrayExample<MockOutput> example = new ArrayExample<>(MockOutputFactory.UNKNOWN_TEST_OUTPUT, new String[]{"a", "b", "c"}, new double[]{1,2,3,});
        Prediction<MockOutput> prediction = new Prediction<>(output, scores, 3, example, false);

        Helpers.writeProtobuf(prediction, Paths.get("src","test","resources","org","tribuo","prediction-431.tribuo"));
    }
}
