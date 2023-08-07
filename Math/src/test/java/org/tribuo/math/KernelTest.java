/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.math;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.tribuo.test.Helpers.testProtoSerialization;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.tribuo.math.kernel.Kernel;
import org.tribuo.math.kernel.Linear;
import org.tribuo.math.kernel.Polynomial;
import org.tribuo.math.kernel.RBF;
import org.tribuo.math.kernel.Sigmoid;
import org.tribuo.math.protos.KernelProto;
import org.tribuo.test.Helpers;

public class KernelTest {

    @Test
    public void testLinear() {
        Linear lin = new Linear();
        testProtoSerialization(lin);
    }

    @Test
    public void testRBF() {
        RBF lin = new RBF(0.5);
        testProtoSerialization(lin);
    }

    @Test
    public void testPolynomial() {
        Polynomial lin = new Polynomial(0.5,1,3);
        testProtoSerialization(lin);
    }

    @Test
    public void testSigmoid() {
        Sigmoid lin = new Sigmoid(0.25,3);
        testProtoSerialization(lin);
    }

    @ParameterizedTest
    @MethodSource("load431Protobufs")
    public void testProto(String name, Kernel actualKernel) throws URISyntaxException, IOException {
        Path kernelPath = Paths.get(KernelTest.class.getResource(name).toURI());
        try (InputStream fis = Files.newInputStream(kernelPath)) {
            KernelProto proto = KernelProto.parseFrom(fis);
            Kernel kernel = Kernel.deserialize(proto);
            assertEquals(actualKernel, kernel);
        }
    }

   
    private static Stream<Arguments> load431Protobufs() throws URISyntaxException, IOException {
    	return Stream.of(
    		      Arguments.of("linear-kernel-431.tribuo", new Linear()),
    		      Arguments.of("poly-kernel-431.tribuo", new Polynomial(1,2,3)),
    		      Arguments.of("rbf-kernel-431.tribuo", new RBF(1.0)),
    		      Arguments.of("sigmoid-kernel-431.tribuo", new Sigmoid(1,2)));
    }

    public void generateProtobufs() throws IOException {
        Helpers.writeProtobuf(new Linear(), Paths.get("src","test","resources","org","tribuo","math","linear-kernel-431.tribuo"));
        Helpers.writeProtobuf(new Polynomial(1,2,3), Paths.get("src","test","resources","org","tribuo","math","poly-kernel-431.tribuo"));
        Helpers.writeProtobuf(new RBF(1.0), Paths.get("src","test","resources","org","tribuo","math","rbf-kernel-431.tribuo"));
        Helpers.writeProtobuf(new Sigmoid(1,2), Paths.get("src","test","resources","org","tribuo","math","sigmoid-kernel-431.tribuo"));
    }
}
