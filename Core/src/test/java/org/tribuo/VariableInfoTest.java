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
import org.tribuo.protos.core.VariableInfoProto;
import org.tribuo.test.Helpers;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class VariableInfoTest {

    @Test
    public void load431Protobufs() throws URISyntaxException, IOException {
        // Comparison objects
        RealInfo realInfo = new RealInfo("Foo", 5, 10, -3, 0, 2);
        RealIDInfo realIDInfo = new RealIDInfo("Bar", 5, 10, -3, 0, 2, 0);
        CategoricalInfo catInfo = new CategoricalInfo("Baz");
        catInfo.observe(1);
        catInfo.observe(1);
        catInfo.observe(2);
        catInfo.observe(2);
        catInfo.observe(5);
        CategoricalIDInfo catIDInfo = new CategoricalIDInfo(catInfo,1);
        catIDInfo.rename("Quux");

        Path realPath = Paths.get(VariableInfoTest.class.getResource("real-info-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(realPath)) {
            VariableInfoProto proto = VariableInfoProto.parseFrom(fis);
            VariableInfo info = VariableInfo.deserialize(proto);
            assertEquals(realInfo, info);
        }
        Path realIDPath = Paths.get(VariableInfoTest.class.getResource("real-id-info-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(realIDPath)) {
            VariableInfoProto proto = VariableInfoProto.parseFrom(fis);
            VariableInfo info = VariableInfo.deserialize(proto);
            assertEquals(realIDInfo, info);
        }
        Path catPath = Paths.get(VariableInfoTest.class.getResource("categorical-info-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(catPath)) {
            VariableInfoProto proto = VariableInfoProto.parseFrom(fis);
            VariableInfo info = VariableInfo.deserialize(proto);
            assertEquals(catInfo, info);
        }
        Path catIDPath = Paths.get(VariableInfoTest.class.getResource("categorical-id-info-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(catIDPath)) {
            VariableInfoProto proto = VariableInfoProto.parseFrom(fis);
            VariableInfo info = VariableInfo.deserialize(proto);
            assertEquals(catIDInfo, info);
        }
    }

    public void generateProtobufs() throws IOException {
        RealInfo realInfo = new RealInfo("Foo", 5, 10, -3, 0, 2);
        RealIDInfo realIDInfo = new RealIDInfo("Bar", 5, 10, -3, 0, 2, 0);
        CategoricalInfo catInfo = new CategoricalInfo("Baz");
        catInfo.observe(1);
        catInfo.observe(1);
        catInfo.observe(2);
        catInfo.observe(2);
        catInfo.observe(5);
        CategoricalIDInfo catIDInfo = new CategoricalIDInfo(catInfo,1);
        catIDInfo.rename("Quux");

        Helpers.writeProtobuf(realInfo, Paths.get("src","test","resources","org","tribuo","real-info-431.tribuo"));
        Helpers.writeProtobuf(realIDInfo, Paths.get("src","test","resources","org","tribuo","real-id-info-431.tribuo"));
        Helpers.writeProtobuf(catInfo, Paths.get("src","test","resources","org","tribuo","categorical-info-431.tribuo"));
        Helpers.writeProtobuf(catIDInfo, Paths.get("src","test","resources","org","tribuo","categorical-id-info-431.tribuo"));
    }
}
