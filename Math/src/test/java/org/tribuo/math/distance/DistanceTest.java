/*
 * Copyright (c) 2023, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.math.distance;

import org.junit.jupiter.api.Test;
import org.tribuo.math.protos.DistanceProto;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.test.Helpers;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class DistanceTest {

    private void testProto(String name, Distance actualDistance) throws URISyntaxException, IOException {
        Path distancePath = Paths.get(DistanceTest.class.getResource(name).toURI());
        try (InputStream fis = Files.newInputStream(distancePath)) {
            DistanceProto proto = DistanceProto.parseFrom(fis);
            Distance distance = ProtoUtil.deserialize(proto);
            assertEquals(actualDistance, distance);
        }
    }

    @Test
    public void load431Protobufs() throws URISyntaxException, IOException {
        testProto("cosine-distance-431.tribuo", new CosineDistance());
        testProto("l1-distance-431.tribuo", new L1Distance());
        testProto("l2-distance-431.tribuo", new L2Distance());
    }

    public void generateProtobufs() throws IOException {
        Helpers.writeProtobuf(new CosineDistance(), Paths.get("src","test","resources","org","tribuo","math","distance","cosine-distance-431.tribuo"));
        Helpers.writeProtobuf(new L1Distance(), Paths.get("src","test","resources","org","tribuo","math","distance","l1-distance-431.tribuo"));
        Helpers.writeProtobuf(new L2Distance(), Paths.get("src","test","resources","org","tribuo","math","distance","l2-distance-431.tribuo"));
    }
}
