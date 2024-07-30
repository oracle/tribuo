/*
 * Copyright (c) 2021, 2024, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.util.embeddings;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensorLike;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.oracle.labs.mlrg.olcut.config.ArgumentException;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Runs text through an ONNX model to extract features as tensors
 * <p>
 * Works with the provided {@link InputProcessor} and {@link OutputProcessor} to get data in and out of the model.
 * Model data is returned as a map from model output name to {@link FloatTensor}. For models, or more
 * specifically, OutputProcessors that return a single output, the name in the map can be ignored
 * and the value for the only entry in the map can be used.
 */
public class OnnxFeatureExtractor implements AutoCloseable, Configurable, Provenancable<ConfiguredObjectProvenance> {
    private static final Logger logger = Logger.getLogger(OnnxFeatureExtractor.class.getName());

    @Config(mandatory=true,description="Path to the model in ONNX format")
    private Path modelPath;

    @Config(mandatory = true, description = "Input processing including the tokenizer.")
    private InputProcessor inputProcessor;

    @Config(mandatory = true, description = "Output processing.")
    private OutputProcessor outputProcessor;

    @Config(description = "Use CUDA")
    private boolean useCUDA = false;

    @Config(description = "Inter-op thread count, can set to 0 to let ORT decide")
    private int interOpThreads = 1;

    @Config(description = "Intra-op thread count, can set to 0 to let ORT decide")
    private int intraOpThreads = 1;

    @Config(description = "Custom op library load path.")
    private Path customOpLibraryPath;

    // ONNX Runtime variables
    private OrtEnvironment env;
    private OrtSession session;
    private OrtSession.SessionOptions sessionOptions;
    private boolean closed = false;

    public record VerboseProcessResult(Map<String,FloatTensor> modelOutput, LongTensor tokenizationOutput) {};

    /**
     * For OLCUT
     */
    private OnnxFeatureExtractor() { }

    /**
     * Constructs a BERTFeatureExtractor.
     * @param modelPath The path to BERT in onnx format.
     * @param inputProcessor The input processor.
     * @param outputProcessor The output processor.
     */
    public OnnxFeatureExtractor(Path modelPath, InputProcessor inputProcessor, OutputProcessor outputProcessor) {
        this.modelPath = modelPath;
        this.inputProcessor = inputProcessor;
        this.outputProcessor = outputProcessor;
        postConfig();
    }

    /**
     * Constructs a BERTFeatureExtractor.
     * @param modelPath The path to BERT in onnx format.
     * @param inputProcessor The input processor.
     * @param outputProcessor The output processor.
     */
    public OnnxFeatureExtractor(Path modelPath, InputProcessor inputProcessor, OutputProcessor outputProcessor, Path customOpLibraryPath) {
        this.modelPath = modelPath;
        this.inputProcessor = inputProcessor;
        this.outputProcessor = outputProcessor;
        this.customOpLibraryPath = customOpLibraryPath;
        postConfig();
    }

    /**
     * Constructs a BERTFeatureExtractor.
     * @param modelPath The path to BERT in onnx format.
     * @param useCUDA Set to true to enable CUDA.
     */
    public OnnxFeatureExtractor(Path modelPath, InputProcessor inputProcessor, OutputProcessor outputProcessor,
                                boolean useCUDA, int interOpThreads, int intraOpThreads) {
        this.modelPath = modelPath;
        this.inputProcessor = inputProcessor;
        this.outputProcessor = outputProcessor;
        this.useCUDA = useCUDA;
        this.interOpThreads = interOpThreads;
        this.intraOpThreads = intraOpThreads;
        postConfig();
    }

    @Override
    public void postConfig() throws PropertyException {
        try {
            env = OrtEnvironment.getEnvironment();
            sessionOptions = new OrtSession.SessionOptions();
            if (customOpLibraryPath != null) {
                sessionOptions.registerCustomOpLibrary(customOpLibraryPath.toAbsolutePath().toString());
            }
            sessionOptions.setInterOpNumThreads(interOpThreads);
            sessionOptions.setIntraOpNumThreads(intraOpThreads);
            if (useCUDA) {
                sessionOptions.addCUDA();
            }
            session = env.createSession(modelPath.toString(),sessionOptions);
            // Validate model
            Map<String, NodeInfo> inputs = session.getInputInfo();
            if (!inputProcessor.validate(inputs)) {
                throw new PropertyException("", "modelPath", "Invalid model, could not validate against the input processor");
            }
            Map<String, NodeInfo> outputs = session.getOutputInfo();
            if (!outputProcessor.validate(outputs)) {
                throw new PropertyException("", "modelPath", "Invalid model, could not validate against the output processor");
            }
        } catch (OrtException e) {
            throw new PropertyException(e,"","modelPath","Failed to load model, ORT threw: ");
        }
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"FeatureExtractor");
    }

    /**
     * Reconstructs the OrtSession using the supplied options.
     * This allows the use of different computation backends and
     * configurations.
     * @param options The new session options.
     * @throws OrtException If the native runtime failed to rebuild itself.
     */
    public void reconfigureOrtSession(OrtSession.SessionOptions options) throws OrtException {
        session.close();
        sessionOptions.close();
        session = env.createSession(modelPath.toString(),options);
        sessionOptions = options;
    }

    public int getEmbeddingDimension() {
        return outputProcessor.getEmbeddingDimension();
    }

    /**
     * Returns the maximum length this BERT will accept.
     * @return The maximum number of tokens (including [CLS] and [SEP], so the maximum is effectively 2 less than this).
     */
    public int getMaxLength() {
        return inputProcessor.getMaxLength();
    }

    /**
     * Gets the configured InputProcessor
     * @return the InputProcessor instance used by this feature extractor
     */
    public InputProcessor getInputProcessor() {
        return inputProcessor;
    }

    /**
     * Returns the vocabulary that this BERTFeatureExtractor understands.
     * @return The vocabulary.
     */
    public Set<String> getVocab() {
        return inputProcessor.getVocab();
    }

    @Override
    public void close() throws OrtException {
        if (!closed) {
            session.close();
            env.close();
            closed = true;
        }
    }

    /**
     * Tokenizes the input using the loaded tokenizer, truncates the
     * token list if it's longer than {@code maxLength} - 2 (to account
     * for [CLS] and [SEP] tokens), and then passes the token
     * list to ONNX Runtime.
     * @param data The input text.
     * @return The BERT features for the supplied data.
     */
    public Map<String,FloatTensor> process(String data) {
        return process(List.of(data));
    }

    /**
     * Tokenizes the input using the loaded tokenizer, truncates the
     * token list if it's longer than {@code maxLength} - 2 (to account
     * for [CLS] and [SEP] tokens), and then passes the token
     * list to ONNX Runtime.
     * @param data The input text.
     * @return The BERT features for the supplied data.
     */
    public Map<String,FloatTensor> process(List<String> data) {
        Map<String, ? extends OnnxTensorLike> input = null;
        try {
            var pInput = inputProcessor.process(env, data);
            input = pInput.inputs();
            try (OrtSession.Result output = session.run(input)) {
                return outputProcessor.process(output, pInput.tokenLengths());
            }
        } catch (OrtException e) {
            throw new IllegalStateException("ORT failed to execute: ", e);
        } finally {
            if (input != null) {
                OnnxValue.close(input);
            }
        }
    }

    /**
     * Processes text as with the {@link #process(List)} method, but is
     * more verbose with its output. The current implementation provides
     * not only the named model outputs, but also a tensor representing
     * the tokenization output.
     *
     * @param data the data to encode
     * @return output from the process
     */
    public VerboseProcessResult processVerbose(List<String> data) {
        Map<String, ? extends OnnxTensorLike> input = null;
        try {
            var pInput = inputProcessor.process(env, data);
            input = pInput.inputs();
            try (OrtSession.Result output = session.run(input)) {
                var pOutput = outputProcessor.process(output, pInput.tokenLengths());
                return new VerboseProcessResult(pOutput, pInput.tokenIds());
            }
        } catch (OrtException e) {
            throw new IllegalStateException("ORT failed to execute: ", e);
        } finally {
            if (input != null) {
                OnnxValue.close(input);
            }
        }

    }

    static float[] extractArray(Map<String,FloatTensor> map) {
        var buf = map.entrySet().stream().findFirst().get().getValue();
        float[] output = new float[buf.buffer.capacity()];
        buf.buffer.get(output);
        return output;
    }

    /**
     * CLI options for running BERT.
     */
    public static class OnnxFeatureExtractorOptions implements Options {
        /**
         * BERTFeatureExtractor instance
         */
        @Option(charName = 'e', longName = "extractor", usage = "OnnxFeatureExtractor instance")
        public OnnxFeatureExtractor extractor;
        /**
         * Input file to read, one doc per line
         */
        @Option(charName = 'i', longName = "input-file", usage = "Input file to read, one doc per line")
        public Path inputFile;
        /**
         * Output json file.
         */
        @Option(charName = 'o', longName = "output-file", usage = "Output json file.")
        public Path outputFile;
    }

    /**
     * Test harness for running a BERT model and inspecting the output.
     * @param args The CLI arguments.
     * @throws IOException If the files couldn't be read or written to.
     * @throws OrtException If the BERT model failed to load, or threw an exception during computation.
     */
    public static void main(String[] args) throws IOException, OrtException {
        OnnxFeatureExtractorOptions opts = new OnnxFeatureExtractorOptions();
        try (ConfigurationManager cm = new ConfigurationManager(args,opts)) {

            logger.info("Loading data from " + opts.inputFile);
            List<String> lines = Files.readAllLines(opts.inputFile, StandardCharsets.UTF_8);

            logger.info("Processing " + lines.size() + " lines with model");
            List<?> results = lines.stream().map(opts.extractor::process).map(OnnxFeatureExtractor::extractArray).toList();

            logger.info("Saving out json to " + opts.outputFile);
            ObjectMapper mapper = new ObjectMapper();
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(opts.outputFile.toFile()))) {
                writer.write(mapper.writeValueAsString(results));
            }

            opts.extractor.close();
        } catch (UsageException e) {
            System.out.println(e.getUsage());
        } catch (ArgumentException e) {
            System.err.println("Invalid arguments\n" + e.getMessage());
        }
    }
}
