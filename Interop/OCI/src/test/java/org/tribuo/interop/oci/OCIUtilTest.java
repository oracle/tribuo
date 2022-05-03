/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;

public class OCIUtilTest {

    @Test
    public void testFileCreation() throws URISyntaxException, IOException {
        OCIUtil.OCIDSConfig dsConfig = new OCIUtil.OCIDSConfig("some-compartment","some-project");
        OCIUtil.OCIModelArtifactConfig config = new OCIUtil.OCIModelArtifactConfig(dsConfig,
                "irises",
                "text",
                "com.example",
                0,
                "test_conda",
                "oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/Onnx for CPU Python 3.7/1.0/test_conda");
        Path zip = OCIUtil.createModelArtifact(new File(OCIUtilTest.class.getResource("iris-lr-model.onnx").toURI()).toPath(),config);
        zip.toFile().delete();
    }

    @Test
    public void testValidation() {
        // Valid conda env names
        Assertions.assertTrue(OCIUtil.validateCondaName("this_is_a_valid_name"));
        Assertions.assertTrue(OCIUtil.validateCondaName("this_is_a_valid_name0"));
        Assertions.assertTrue(OCIUtil.validateCondaName("0this_is_a_valid_name0"));

        // Invalid conda env names
        Assertions.assertFalse(OCIUtil.validateCondaName("this_isn't_a_valid_name"));
        Assertions.assertFalse(OCIUtil.validateCondaName("nor is this"));
        Assertions.assertFalse(OCIUtil.validateCondaName("nor@is@this"));
        Assertions.assertFalse(OCIUtil.validateCondaName("http://some-url.example.com"));

        // Valid conda paths
        Assertions.assertTrue(OCIUtil.validateCondaPath("oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/TensorFlow 2.7 for CPU Python 3.7/1.0/tensorflow27_p37_cpu_v1"));

        // Invalid conda paths
        Assertions.assertFalse(OCIUtil.validateCondaPath("https://example.com/some/file"));
        Assertions.assertFalse(OCIUtil.validateCondaPath("s3://some-bucket/test.log"));
        Assertions.assertFalse(OCIUtil.validateCondaPath("oci://some-bucket^^^^^/test.log"));
        Assertions.assertFalse(OCIUtil.validateCondaPath("file://path/on/disk/test.log"));
        Assertions.assertFalse(OCIUtil.validateCondaPath("/path/on/disk/test.log"));
        Assertions.assertFalse(OCIUtil.validateCondaPath("C:/path/on/disk/test.log"));
    }
}
