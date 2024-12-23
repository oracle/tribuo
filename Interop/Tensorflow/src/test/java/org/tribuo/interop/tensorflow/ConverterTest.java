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

package org.tribuo.interop.tensorflow;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.tribuo.interop.tensorflow.protos.FeatureConverterProto;
import org.tribuo.interop.tensorflow.protos.OutputConverterProto;
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

public class ConverterTest {

    @ParameterizedTest
    @MethodSource("load431FeatureProtobufs")
    public void testFeatureProto(String name, FeatureConverter actualTransformer) throws URISyntaxException, IOException {
        Path distancePath = Paths.get(ConverterTest.class.getResource(name).toURI());
        try (InputStream fis = Files.newInputStream(distancePath)) {
            FeatureConverterProto proto = FeatureConverterProto.parseFrom(fis);
            FeatureConverter distance = ProtoUtil.deserialize(proto);
            assertEquals(actualTransformer, distance);
        }
    }

    private static Stream<Arguments> load431FeatureProtobufs() {
        return Stream.of(
                Arguments.of("dense-431.tribuo", new DenseFeatureConverter("foo")),
                Arguments.of("image-431.tribuo", new ImageConverter("foo",64,64,3))
        );
    }

    @ParameterizedTest
    @MethodSource("load431OutputProtobufs")
    public void testOutputProto(String name, OutputConverter<?> actualTransformer) throws URISyntaxException, IOException {
        Path distancePath = Paths.get(ConverterTest.class.getResource(name).toURI());
        try (InputStream fis = Files.newInputStream(distancePath)) {
            OutputConverterProto proto = OutputConverterProto.parseFrom(fis);
            OutputConverter<?> distance = ProtoUtil.deserialize(proto);
            assertEquals(actualTransformer, distance);
        }
    }

    private static Stream<Arguments> load431OutputProtobufs() {
        return Stream.of(
                Arguments.of("label-431.tribuo", new LabelConverter()),
                Arguments.of("multilabel-431.tribuo", new MultiLabelConverter()),
                Arguments.of("regressor-431.tribuo", new RegressorConverter())
        );
    }

    public void generateProtobufs() throws IOException {
        Helpers.writeProtobuf(new DenseFeatureConverter("foo"), Paths.get("src","test","resources","org","tribuo","interop","tensorflow","dense-431.tribuo"));
        Helpers.writeProtobuf(new ImageConverter("foo",64,64,3), Paths.get("src","test","resources","org","tribuo","interop","tensorflow","image-431.tribuo"));
        Helpers.writeProtobuf(new LabelConverter(), Paths.get("src","test","resources","org","tribuo","interop","tensorflow","label-431.tribuo"));
        Helpers.writeProtobuf(new MultiLabelConverter(), Paths.get("src","test","resources","org","tribuo","interop","tensorflow","multilabel-431.tribuo"));
        Helpers.writeProtobuf(new RegressorConverter(), Paths.get("src","test","resources","org","tribuo","interop","tensorflow","regressor-431.tribuo"));
    }
}
