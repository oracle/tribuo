/*
 * Copyright (c) 2022, 2023, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.math.util;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.tribuo.math.protos.NormalizerProto;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.test.Helpers;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.tribuo.test.Helpers.testProtoSerialization;

public class NormalizerTest {

    @Test
    public void normalizerTest() {
        Normalizer n = new Normalizer();
        testProtoSerialization(n);
    }

    @Test
    public void expNormalizerTest() {
        ExpNormalizer n = new ExpNormalizer();
        testProtoSerialization(n);
    }

    @Test
    public void noopNormalizerTest() {
        NoopNormalizer n = new NoopNormalizer();
        testProtoSerialization(n);
    }

    @Test
    public void sigmoidNormalizerTest() {
        SigmoidNormalizer n = new SigmoidNormalizer();
        testProtoSerialization(n);
    }

    @ParameterizedTest
    @MethodSource("load431Protobufs")
    public void testProto(String name, VectorNormalizer actualNormalizer) throws URISyntaxException, IOException {
        Path normalizerPath = Paths.get(NormalizerTest.class.getResource(name).toURI());
        try (InputStream fis = Files.newInputStream(normalizerPath)) {
            NormalizerProto proto = NormalizerProto.parseFrom(fis);
            VectorNormalizer normalizer = ProtoUtil.deserialize(proto);
            assertEquals(actualNormalizer, normalizer);
        }
    }

    private static Stream<Arguments> load431Protobufs() throws URISyntaxException, IOException {
    	return Stream.of(
  		      Arguments.of("normalizer-431.tribuo", new Normalizer()),
  		      Arguments.of("noop-normalizer-431.tribuo", new NoopNormalizer()),
  		      Arguments.of("exp-normalizer-431.tribuo", new ExpNormalizer()),
  		      Arguments.of("sigmoid-normalizer-431.tribuo", new SigmoidNormalizer()));
    }

    public void generateProtobufs() throws IOException {
        Helpers.writeProtobuf(new Normalizer(), Paths.get("src","test","resources","org","tribuo","math","util","normalizer-431.tribuo"));
        Helpers.writeProtobuf(new NoopNormalizer(), Paths.get("src","test","resources","org","tribuo","math","util","noop-normalizer-431.tribuo"));
        Helpers.writeProtobuf(new ExpNormalizer(), Paths.get("src","test","resources","org","tribuo","math","util","exp-normalizer-431.tribuo"));
        Helpers.writeProtobuf(new SigmoidNormalizer(), Paths.get("src","test","resources","org","tribuo","math","util","sigmoid-normalizer-431.tribuo"));
    }
}
