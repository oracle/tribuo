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

package org.tribuo.interop.oci;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.tribuo.interop.oci.protos.OCIOutputConverterProto;
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

public class OCIConverterTest {

    @ParameterizedTest
    @MethodSource("load431Protobufs")
    public void testOutputProto(String name, OCIOutputConverter<?> actualTransformer) throws URISyntaxException, IOException {
        Path distancePath = Paths.get(OCIConverterTest.class.getResource(name).toURI());
        try (InputStream fis = Files.newInputStream(distancePath)) {
            OCIOutputConverterProto proto = OCIOutputConverterProto.parseFrom(fis);
            OCIOutputConverter<?> distance = ProtoUtil.deserialize(proto);
            assertEquals(actualTransformer, distance);
        }
    }

    private static Stream<Arguments> load431Protobufs() {
        return Stream.of(
                Arguments.of("label-431.tribuo", new OCILabelConverter(true)),
                Arguments.of("multilabel-431.tribuo", new OCIMultiLabelConverter(0.5,true)),
                Arguments.of("regressor-431.tribuo", new OCIRegressorConverter())
        );
    }

    public void generateProtobufs() throws IOException {
        Helpers.writeProtobuf(new OCILabelConverter(true), Paths.get("src","test","resources","org","tribuo","interop","oci","label-431.tribuo"));
        Helpers.writeProtobuf(new OCIMultiLabelConverter(0.5,true), Paths.get("src","test","resources","org","tribuo","interop","oci","multilabel-431.tribuo"));
        Helpers.writeProtobuf(new OCIRegressorConverter(), Paths.get("src","test","resources","org","tribuo","interop","oci","regressor-431.tribuo"));
    }
}
