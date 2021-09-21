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

package org.tribuo.tests;

import com.oracle.labs.mlrg.olcut.util.IOUtil;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Utils for working with classpath resources at test time.
 */
public final class Resources {
    private Resources() {}

    /**
     * Copies a classpath resource to a temporary file.
     * @param resource The resource to copy.
     * @return The path of the temporary file.
     * @throws IOException If the resource could not be copied.
     */
    public static Path copyResourceToTmp(String resource) throws IOException {
        Path path = Files.createTempFile("test", ".csv");
        path.toFile().deleteOnExit();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(IOUtil.getInputStream(resource), StandardCharsets.UTF_8));
             BufferedWriter writer = Files.newBufferedWriter(path, Charset.defaultCharset())) {

            String ln;
            while ((ln = reader.readLine()) != null) {
                writer.write(ln);
                writer.newLine();
            }

            return path;
        }
    }
}