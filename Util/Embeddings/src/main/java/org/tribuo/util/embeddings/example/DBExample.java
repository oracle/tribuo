/*
 * Copyright (c) 2023, 2024, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.util.embeddings.example;

import ai.onnxruntime.OrtException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.oracle.labs.mlrg.olcut.config.ArgumentException;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import org.tribuo.util.embeddings.FloatTensorBuffer;
import org.tribuo.util.embeddings.processors.NoOpInputProcessor;
import org.tribuo.util.embeddings.processors.NoOpOutputProcessor;
import org.tribuo.util.embeddings.OnnxFeatureExtractor;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Example program for running an ONNX model which has been augmented by oml4py for use inside Oracle 23ai.
 */
public final class DBExample {
    private static final Logger logger = Logger.getLogger(DBExample.class.getName());

    /**
     * CLI options for running a DB augmented model.
     */
    public static class DBExampleOptions implements Options {
        /**
         * ONNX model path.
         */
        @Option(charName = 'm', longName = "model-path", usage = "ONNX model")
        public Path modelPath;
        /**
         * Embedding dimension.
         */
        @Option(charName = 'e', longName = "embedding-dim", usage = "Embedding dimension")
        public int embeddingDimension;
        /**
         * Path to the custom op library for tokenizers.
         */
        @Option(charName = 'l', longName = "custom-op-library-path", usage = "Path to the custom op library")
        public Path customOpPath;
        /**
         * Input file to read, one doc per line.
         */
        @Option(charName = 'i', longName = "input-file", usage = "Input file to read, one doc per line")
        public Path inputFile;
        /**
         * Output json file.
         */
        @Option(charName = 'o', longName = "output-file", usage = "Output json file.")
        public Path outputFile;
    }

    static float[] extractArray(Map<String, FloatTensorBuffer> map) {
        var buf = map.entrySet().stream().findFirst().get().getValue();
        return buf.getFlatArray();
    }

    /**
     * Entry point for the augmented model runner.
     * @param args The model arguments.
     * @throws IOException If the model could not be read or the output could not be written.
     * @throws OrtException If ORT threw an exception when running the model.
     */
    public static void main(String[] args) throws IOException, OrtException {
        DBExampleOptions opts = new DBExampleOptions();
        try (ConfigurationManager cm = new ConfigurationManager(args,opts)) {
            OnnxFeatureExtractor extractor = new OnnxFeatureExtractor(
                opts.modelPath,
                new NoOpInputProcessor(),
                new NoOpOutputProcessor(opts.embeddingDimension),
                opts.customOpPath
            );

            logger.info("Loading data from " + opts.inputFile);
            List<String> lines = Files.readAllLines(opts.inputFile, StandardCharsets.UTF_8);

            logger.info("Processing " + lines.size() + " lines with model");
            List<?> results = lines.stream().map(extractor::process).map(DBExample::extractArray).toList();

            logger.info("Saving out json to " + opts.outputFile);
            ObjectMapper mapper = new ObjectMapper();
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(opts.outputFile.toFile()))) {
                writer.write(mapper.writeValueAsString(results));
            }

            extractor.close();
        } catch (UsageException e) {
            System.out.println(e.getUsage());
        } catch (ArgumentException e) {
            System.err.println("Invalid arguments\n" + e.getMessage());
        }
    }

}
