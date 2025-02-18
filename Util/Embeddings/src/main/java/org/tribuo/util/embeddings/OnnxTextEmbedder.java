/*
 * Copyright (c) 2021, 2025, Oracle and/or its affiliates. All rights reserved.
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
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtUtil;
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
import java.nio.Buffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Runs text through an ONNX model to extract features as tensors
 * <p>
 * Works with the provided {@link InputProcessor} and {@link OutputProcessor} to get data in and out of the model.
 * Model data is returned as a map from model output name to {@link FloatTensorBuffer}. For models, or more
 * specifically, OutputProcessors that return a single output, the name in the map can be ignored
 * and the value for the only entry in the map can be used.
 */
public final class OnnxTextEmbedder implements AutoCloseable, Configurable, Provenancable<ConfiguredObjectProvenance> {
    private static final Logger logger = Logger.getLogger(OnnxTextEmbedder.class.getName());

    /**
     * Path to the model in ONNX format.
     */
    @Config(mandatory = true, description = "Path to the model in ONNX format.")
    private Path modelPath;

    /**
     * The input processor object.
     */
    @Config(mandatory = true, description = "Input processing including the tokenizer.")
    private InputProcessor inputProcessor;

    /**
     * The output processor object.
     */
    @Config(mandatory = true, description = "Output processing.")
    private OutputProcessor outputProcessor;

    /**
     * Should the computation use CUDA GPUs?
     */
    @Config(description = "Use CUDA")
    private boolean useCUDA = false;

    /**
     * Inter-op thread count, can be set to 0 to let ORT decide.
     */
    @Config(description = "Inter-op thread count, can set to 0 to let ORT decide.")
    private int interOpThreads = 1;

    /**
     * Intra-op thread count, can be set to 0 to let ORT decide.
     */
    @Config(description = "Intra-op thread count, can set to 0 to let ORT decide.")
    private int intraOpThreads = 1;

    /**
     * Custom op library load path.
     */
    @Config(description = "Custom op library load path.")
    private Path customOpLibraryPath = null;

    // ONNX Runtime variables
    private OrtEnvironment env;
    private OrtSession session;
    private OrtSession.SessionOptions sessionOptions;
    private boolean closed = false;

    /**
     * The model output plus the input tokens.
     * @param modelOutput The model output.
     * @param tokenizationOutput The tokens.
     */
    public record VerboseProcessResult(Map<String, FloatTensorBuffer> modelOutput, LongTensorBuffer tokenizationOutput) {};

    /**
     * For OLCUT
     */
    private OnnxTextEmbedder() { }

    /**
     * Constructs a ONNXFeatureExtractor.
     * @param modelPath The path to an embedding model in ONNX format.
     * @param inputProcessor The input processor.
     * @param outputProcessor The output processor.
     */
    public OnnxTextEmbedder(Path modelPath, InputProcessor inputProcessor, OutputProcessor outputProcessor) {
        this(modelPath, inputProcessor, outputProcessor, null);
    }

    /**
     * Constructs a ONNXFeatureExtractor.
     * @param modelPath The path to an embedding model in ONNX format.
     * @param inputProcessor The input processor.
     * @param outputProcessor The output processor.
     * @param customOpLibraryPath The path to the custom op library required by this ONNX model.
     */
    public OnnxTextEmbedder(Path modelPath, InputProcessor inputProcessor, OutputProcessor outputProcessor, Path customOpLibraryPath) {
        this.modelPath = modelPath;
        this.inputProcessor = inputProcessor;
        this.outputProcessor = outputProcessor;
        this.customOpLibraryPath = customOpLibraryPath;
        postConfig();
    }

    /**
     * Constructs a ONNXFeatureExtractor.
     * @param modelPath The path to an embedding model in ONNX format.
     * @param inputProcessor The input processor.
     * @param outputProcessor The output processor.
     * @param useCUDA Set to true to enable CUDA.
     * @param interOpThreads The number of inter-op CPU threads to use.
     * @param intraOpThreads The number of intra-op CPU threads to use.
     * @param customOpLibraryPath The path to the custom op library required by this ONNX model.
     */
    public OnnxTextEmbedder(Path modelPath, InputProcessor inputProcessor, OutputProcessor outputProcessor,
                            Path customOpLibraryPath, boolean useCUDA, int interOpThreads, int intraOpThreads) {
        this.modelPath = modelPath;
        this.inputProcessor = inputProcessor;
        this.outputProcessor = outputProcessor;
        this.useCUDA = useCUDA;
        this.interOpThreads = interOpThreads;
        this.intraOpThreads = intraOpThreads;
        this.customOpLibraryPath = customOpLibraryPath;
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

    /**
     * Returns the embedding dimension of this model if known.
     * @return The embedding dimension.
     */
    public int getEmbeddingDimension() {
        return outputProcessor.getEmbeddingDimension();
    }

    /**
     * Returns the maximum length this model will accept.
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
     * Gets the configured OutputProcessor
     * @return the OutputProcessor instance used by this feature extractor
     */
    public OutputProcessor getOutputProcessor() {
        return outputProcessor;
    }

    /**
     * Returns the vocabulary that this ONNXFeatureExtractor understands.
     * @return The vocabulary.
     */
    public Set<String> getVocab() {
        return inputProcessor.getVocab();
    }

    @Override
    public void close() throws OrtException {
        if (!closed) {
            session.close();
            sessionOptions.close();
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
     * @return The embeddings for the supplied data.
     */
    public Map<String, FloatTensorBuffer> process(String data) {
        return process(List.of(data));
    }

    /**
     * Tokenizes the input using the loaded tokenizer, truncates the
     * token list if it's longer than {@code maxLength} - 2 (to account
     * for [CLS] and [SEP] tokens), and then passes the token
     * list to ONNX Runtime.
     * @param data The input text.
     * @param inputCache The input buffer cache, constructed by the {@link InputProcessor}. Each cache must be used by at most one thread simultaneously.
     * @param outputCache The output buffer cache, constructed by the {@link OutputProcessor}. Each cache must be used by at most one thread simultaneously.
     * @return The embeddings for the supplied data.
     */
    public Map<String, FloatTensorBuffer> process(String data, BufferCache inputCache, BufferCache outputCache) {
        return process(List.of(data), inputCache, outputCache);
    }

    /**
     * Tokenizes the input using the loaded tokenizer, truncates the
     * token list if it's longer than {@code maxLength} - 2 (to account
     * for [CLS] and [SEP] tokens), and then passes the token
     * list to ONNX Runtime.
     * @param data The input text.
     * @return The embeddings for the supplied data.
     */
    public Map<String, FloatTensorBuffer> process(List<String> data) {
        try (var pInput = inputProcessor.process(env, data)) {
            var input = pInput.inputs();
            try (OrtSession.Result output = session.run(input)) {
                return outputProcessor.process(output, pInput.tokenLengths());
            }
        } catch (OrtException e) {
            throw new IllegalStateException("ORT failed to execute: ", e);
        }
    }

    /**
     * Tokenizes the input using the loaded tokenizer, truncates the
     * token list if it's longer than {@code maxLength} - 2 (to account
     * for [CLS] and [SEP] tokens), and then passes the token
     * list to ONNX Runtime.
     * @param data The input text.
     * @param inputCache The input buffer cache, constructed by the {@link InputProcessor}. Each cache must be used by at most one thread simultaneously.
     * @param outputCache The output buffer cache, constructed by the {@link OutputProcessor}. Each cache must be used by at most one thread simultaneously.
     * @return The embeddings for the supplied data.
     */
    public Map<String, FloatTensorBuffer> process(List<String> data, BufferCache inputCache, BufferCache outputCache) {
        try (var pInput = inputProcessor.process(env, data, inputCache)) {
            var input = pInput.inputs();
            int batchSize = (int) pInput.tokenIds().shape()[0];
            int numTokens = (int) pInput.tokenIds().shape()[1];
            var outputBuffers = outputProcessor.createOutputTensors(env, outputCache, batchSize, numTokens);
            try (OrtSession.Result output = session.run(input, outputBuffers)) {
                return outputProcessor.process(output, pInput.tokenLengths());
            } finally {
                OnnxValue.close(outputBuffers);
            }
        } catch (OrtException e) {
            throw new IllegalStateException("ORT failed to execute: ", e);
        }
    }

    /**
     * Processes the supplied token list and generates embeddings.
     * <p>
     * Any tokens which are unknown to the tokenizer are replaced with the UNK token before embedding.
     * @param tokens The tokens.
     * @return The embeddings for the supplied data.
     */
    public Map<String, FloatTensorBuffer> processTokens(List<List<String>> tokens) {
        try (var pInput = inputProcessor.processTokensBatch(env, tokens)) {
            var input = pInput.inputs();
            try (OrtSession.Result output = session.run(input)) {
                return outputProcessor.process(output, pInput.tokenLengths());
            }
        } catch (OrtException e) {
            throw new IllegalStateException("ORT failed to execute: ", e);
        }
    }

    /**
     * Processes the supplied token list and generates embeddings.
     * <p>
     * Any tokens which are unknown to the tokenizer are replaced with the UNK token before embedding.
     * @param tokens The tokens.
     * @param inputCache The input buffer cache, constructed by the {@link InputProcessor}. Each cache must be used by at most one thread simultaneously.
     * @param outputCache The output buffer cache, constructed by the {@link OutputProcessor}. Each cache must be used by at most one thread simultaneously.
     * @return The embeddings for the supplied data.
     */
    public Map<String, FloatTensorBuffer> processTokens(List<List<String>> tokens, BufferCache inputCache, BufferCache outputCache) {
        try (var pInput = inputProcessor.processTokensBatch(env, tokens, inputCache)) {
            var input = pInput.inputs();
            int batchSize = (int) pInput.tokenIds().shape()[0];
            int numTokens = (int) pInput.tokenIds().shape()[1];
            var outputBuffers = outputProcessor.createOutputTensors(env, outputCache, batchSize, numTokens);
            try (OrtSession.Result output = session.run(input, outputBuffers)) {
                return outputProcessor.process(output, pInput.tokenLengths());
            } finally {
                OnnxValue.close(outputBuffers);
            }
        } catch (OrtException e) {
            throw new IllegalStateException("ORT failed to execute: ", e);
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
        try (var pInput = inputProcessor.process(env, data)) {
            var input = pInput.inputs();
            try (OrtSession.Result output = session.run(input)) {
                var pOutput = outputProcessor.process(output, pInput.tokenLengths());
                return new VerboseProcessResult(pOutput, pInput.tokenIds());
            }
        } catch (OrtException e) {
            throw new IllegalStateException("ORT failed to execute: ", e);
        }
    }

    /**
     * Processes text as with the {@link #process(List)} method, but is
     * more verbose with its output. The current implementation provides
     * not only the named model outputs, but also a tensor representing
     * the tokenization output.
     *
     * @param data the data to encode
     * @param inputCache The input buffer cache, constructed by the {@link InputProcessor}. Each cache must be used by at most one thread simultaneously.
     * @param outputCache The output buffer cache, constructed by the {@link OutputProcessor}. Each cache must be used by at most one thread simultaneously.
     * @return output from the process
     */
    public VerboseProcessResult processVerbose(List<String> data, BufferCache inputCache, BufferCache outputCache) {
        try (var pInput = inputProcessor.process(env, data, inputCache)) {
            var input = pInput.inputs();
            int batchSize = (int) pInput.tokenIds().shape()[0];
            int numTokens = (int) pInput.tokenIds().shape()[1];
            var outputBuffers = outputProcessor.createOutputTensors(env, outputCache, batchSize, numTokens);
            try (OrtSession.Result output = session.run(input, outputBuffers)) {
                var pOutput = outputProcessor.process(output, pInput.tokenLengths());
                return new VerboseProcessResult(pOutput, pInput.tokenIds());
            } finally {
                OnnxValue.close(outputBuffers);
            }
        } catch (OrtException e) {
            throw new IllegalStateException("ORT failed to execute: ", e);
        }
    }

    static float[] extractArray(Map<String, FloatTensorBuffer> map) {
        var buf = map.entrySet().stream().findFirst().get().getValue();
        float[] output = new float[buf.buffer.capacity()];
        buf.buffer.get(output);
        return output;
    }

    /**
     * CLI options for running an ONNX model.
     */
    public static class OnnxTextEmbedderOptions implements Options {
        /**
         * OnnxTextEmbedder instance.
         */
        @Option(charName = 'e', longName = "extractor", usage = "OnnxTextEmbedder instance.")
        public OnnxTextEmbedder extractor;
        /**
         * Input file to read, one doc per line.
         */
        @Option(charName = 'i', longName = "input-file", usage = "Input file to read, one doc per line.")
        public Path inputFile;
        /**
         * Output json file.
         */
        @Option(charName = 'o', longName = "output-file", usage = "Output json file.")
        public Path outputFile;
        /**
         * Batch size for processing.
         */
        @Option(charName = 'b', longName = "batch-size", usage = "The batch size for processing.")
        public int batchSize = 1;
        /**
         * Use caching?
         */
        @Option(charName = 'u', longName = "use-cache", usage = "Use the buffer cache.")
        public boolean useCache;
    }

    /**
     * Test harness for running an ONNX model and inspecting the output.
     * @param args The CLI arguments.
     * @throws IOException If the files couldn't be read or written to.
     * @throws OrtException If the ONNX model failed to load, or threw an exception during computation.
     */
    public static void main(String[] args) throws IOException, OrtException {
        OnnxTextEmbedderOptions opts = new OnnxTextEmbedderOptions();
        try (ConfigurationManager cm = new ConfigurationManager(args,opts)) {
            logger.info("Loading data from " + opts.inputFile);
            List<String> lines = Files.readAllLines(opts.inputFile, StandardCharsets.UTF_8);

            logger.info("Processing " + lines.size() + " lines with model");
            List<String> batch = new ArrayList<>(opts.batchSize);
            List<float[]> results = new ArrayList<>(lines.size());
            final Instant extractStart = Instant.now();
            if (opts.useCache) {
                var ip = opts.extractor.getInputProcessor();
                var op = opts.extractor.getOutputProcessor();
                BufferCache inputCache = ip.createInputCache(opts.batchSize, ip.getMaxLength());
                BufferCache outputCache = op.createOutputCache(opts.batchSize, ip.getMaxLength());
                for (var s : lines) {
                    batch.add(s);
                    if (batch.size() == opts.batchSize) {
                        var output = opts.extractor.process(batch, inputCache, outputCache);
                        float[] arr = extractArray(output);
                        float[][] shaped = (float[][]) OrtUtil.reshape(arr, new long[]{arr.length / opts.batchSize, arr.length % opts.batchSize});
                        results.addAll(Arrays.asList(shaped));
                        batch.clear();
                    }
                }
                if (!batch.isEmpty()) {
                    var output = opts.extractor.process(batch, inputCache, outputCache);
                    float[] arr = extractArray(output);
                    float[][] shaped = (float[][]) OrtUtil.reshape(arr, new long[]{arr.length / opts.batchSize, arr.length % opts.batchSize});
                    results.addAll(Arrays.asList(shaped));
                }
            } else if (opts.batchSize > 1) {
                for (var s : lines) {
                    batch.add(s);
                    if (batch.size() == opts.batchSize) {
                        var output = opts.extractor.process(batch);
                        float[] arr = extractArray(output);
                        float[][] shaped = (float[][]) OrtUtil.reshape(arr, new long[]{arr.length / opts.batchSize, arr.length % opts.batchSize});
                        results.addAll(Arrays.asList(shaped));
                        batch.clear();
                    }
                }
                if (!batch.isEmpty()) {
                    var output = opts.extractor.process(batch);
                    float[] arr = extractArray(output);
                    float[][] shaped = (float[][]) OrtUtil.reshape(arr, new long[]{arr.length / opts.batchSize, arr.length % opts.batchSize});
                    results.addAll(Arrays.asList(shaped));
                }
            } else {
                results = lines.stream().map(opts.extractor::process).map(OnnxTextEmbedder::extractArray).toList();
            }
            final Instant extractEnd = Instant.now();
            var duration = Duration.between(extractStart, extractEnd);
            logger.info("Feature extraction took " + duration);

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
