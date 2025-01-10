/*
 * Copyright (c) 2023, 2025, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.util.embeddings.processors;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.util.embeddings.BERTTokenizerConfig;
import org.tribuo.util.embeddings.InputProcessor;
import org.tribuo.util.embeddings.LongTensorBuffer;
import org.tribuo.util.embeddings.TokenizerConfig;
import org.tribuo.util.tokens.Tokenizer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

/**
 * An input processor which produces BERT style inputs, with input_ids, attention_mask and token_type_ids.
 */
public class BERTInputProcessor implements InputProcessor {
    private static final Logger logger = Logger.getLogger(BERTInputProcessor.class.getName());

    // BERT input names
    /**
     * Input name for the token ids.
     */
    public static final String INPUT_IDS = "input_ids";
    /**
     * Input name for the attention mask.
     */
    public static final String ATTENTION_MASK = "attention_mask";
    /**
     * Input name for the token type ids.
     */
    public static final String TOKEN_TYPE_IDS = "token_type_ids";

    // Token names
    /**
     * Default classification token name.
     */
    public static final String CLASSIFICATION_TOKEN = "[CLS]";
    /**
     * Default separator token name.
     */
    public static final String SEPARATOR_TOKEN = "[SEP]";
    /**
     * Default unknown token name.
     */
    public static final String UNKNOWN_TOKEN = "[UNK]";
    /**
     * Default pad token name.
     */
    public static final String PAD_TOKEN = "[PAD]";
    /**
     * Default BOS token.
     */
    public static final String BOS_TOKEN = "<s>";
    /**
     * Default EOS token.
     */
    public static final String EOS_TOKEN = "</s>";

    // Values expected by BERT inputs
    /**
     * Mask value.
     */
    public static final long MASK_VALUE = 1;
    /**
     * The value used in the attention mask to ignore a token
     */
    public static final long NEGATIVE_MASK_VALUE = 0;
    /**
     * Token type value for the first sentence.
     */
    public static final long TOKEN_TYPE_VALUE = 0;

    /**
     * Path to the tokenizer config.
     */
    @Config(mandatory=true,description="Path to the tokenizer config")
    protected Path tokenizerPath;
    /**
     * Maximum length of a token sequence.
     */
    @Config(mandatory=true,description="Maximum length of a token sequence")
    protected int maxLength;
    /**
     * Fully qualified tokenizer config class name.
     */
    @Config(description="Fully qualified tokenizer config class name")
    protected String configClass = BERTTokenizerConfig.class.getName();
    /**
     * Token id input name, defaults to {@link #INPUT_IDS}.
     */
    @Config(description="Token id input name")
    protected String inputIdsName = INPUT_IDS;
    /**
     * Is the attention mask input required?
     */
    @Config(description = "Is the attention_mask input required?")
    protected boolean useMask = true;
    /**
     * Attention mask input name, defaults to {@link #ATTENTION_MASK}.
     */
    @Config(description="Attention mask input name")
    protected String maskName = ATTENTION_MASK;
    /**
     * Value of the attention mask for tokens which should be attended to.
     */
    @Config(description = "Attention mask value")
    protected long maskValue = MASK_VALUE;
    /**
     * Value of the attention mask for tokens which should not be attended to.
     */
    @Config(description = "Negative attention maks value")
    protected long negativeMaskValue = NEGATIVE_MASK_VALUE;
    /**
     * Is the token type input required?
     */
    @Config(description = "Is the token_type input required?")
    protected boolean useTokenType = true;
    /**
     * Name of the token type input, defaults to {@link #TOKEN_TYPE_IDS}.
     */
    @Config(description="Token type input name")
    protected String tokenTypeName = TOKEN_TYPE_IDS;
    /**
     * Value of the token type ids for the first type.
     */
    @Config(description = "Token type value")
    protected long tokenTypeValue = TOKEN_TYPE_VALUE;

    // Vocab and special terms
    protected Map<String,Integer> tokenIDs;
    protected String bosToken = CLASSIFICATION_TOKEN;
    protected String eosToken = SEPARATOR_TOKEN;
    protected String unkToken = UNKNOWN_TOKEN;
    protected String padToken = PAD_TOKEN;
    protected int bosTokenId;
    protected int eosTokenId;
    protected int unkTokenId;
    protected int padTokenId;

    // Tokenizer
    protected Tokenizer tokenizer;

    /**
     * For OLCUT.
     */
    protected BERTInputProcessor() {}

    /**
     * Constructor which builds an input processor using the supplied tokenizer JSON, max length and tokenizer config class.
     * @param tokenizerPath The path to the tokenizer JSON.
     * @param maxLength The maximum number of tokens this input processor should produce.
     * @param configClass The tokenizer configuration class.
     */
    public BERTInputProcessor(Path tokenizerPath, int maxLength, Class<? extends TokenizerConfig> configClass) {
        this.tokenizerPath = tokenizerPath;
        this.maxLength = maxLength;
        this.configClass = configClass.getName();
        postConfig();
    }

    /**
     * Constructor which builds an input processor using the supplied tokenizer JSON, max length and tokenizer config class.
     * @param tokenizerPath The path to the tokenizer JSON.
     * @param maxLength The maximum number of tokens this input processor should produce.
     * @param configClass The tokenizer configuration class.
     * @param inputIdsName Name for the input ids input.
     * @param useMask Should this processor produce an attention mask?
     * @param maskName The name for the attention mask input.
     * @param maskValue Value for tokens which should be attended to.
     * @param useTokenType Should this processor produce a token type ids?
     * @param tokenTypeName The name for the token type ids input.
     * @param tokenTypeValue Value for the token type ids first input.
     */
    public BERTInputProcessor(Path tokenizerPath, int maxLength, Class<? extends TokenizerConfig> configClass,
                              String inputIdsName, boolean useMask, String maskName, long maskValue, boolean useTokenType,
                              String tokenTypeName, long tokenTypeValue) {
        this.tokenizerPath = tokenizerPath;
        this.maxLength = maxLength;
        this.configClass = configClass.getName();
        this.inputIdsName = inputIdsName;
        this.useMask = useMask;
        this.maskName = maskName;
        this.maskValue = maskValue;
        this.useTokenType = useTokenType;
        this.tokenTypeName = tokenTypeName;
        this.tokenTypeValue = tokenTypeValue;
        postConfig();
    }

    @Override
    public void postConfig() throws PropertyException {
        try {
            TokenizerConfig config = TokenizerConfig.loadTokenizer(Class.forName(configClass),
                tokenizerPath);
            tokenIDs = config.tokenIDs();
            unkToken = config.unknownToken();
            bosToken = config.classificationToken();
            eosToken = config.separatorToken();
            padToken = config.padToken();
            bosTokenId = tokenIDs.get(bosToken);
            eosTokenId = tokenIDs.get(eosToken);
            padTokenId = tokenIDs.get(padToken);
            unkTokenId = tokenIDs.get(unkToken);
            tokenizer = config.get();
        } catch (ClassNotFoundException e) {
            throw new PropertyException("","configClass",e.getMessage());
        }
    }

    @Override
    public boolean validate(Map<String, NodeInfo> inputs) {
        if (inputs.size() > 3) {
            logger.warning(
                "Invalid model for this processor, expected 1, 2 or 3 inputs, found " + inputs.size());
            return false;
        } else if (!inputs.containsKey(inputIdsName)) {
            logger.warning("Invalid model, expected to find an input called '" + inputIdsName + "'");
            return false;
        } else if (!inputs.containsKey(maskName) && useMask) {
            logger.warning("Invalid model, expected to find an input called '" + maskName + "'");
            return false;
        } else if (!inputs.containsKey(tokenTypeName) && useTokenType) {
            logger.warning("Invalid model, expected to find an input called '" + tokenTypeName + "'");
            return false;
        }
        return true;
    }

    /**
     * Returns the vocabulary that this BERTInputProcessor understands.
     * @return The vocabulary.
     */
    @Override
    public Set<String> getVocab() {
        return Collections.unmodifiableSet(tokenIDs.keySet());
    }

    @Override
    public long getTokenId(String token) {
        return tokenIDs.getOrDefault(token,unkTokenId);
    }

    @Override
    public int getMaxLength() {
        return maxLength;
    }

    @Override
    public ProcessedInput process(OrtEnvironment env, List<String> input) throws OrtException {
        if (input.size() == 1) {
            // no padding
            var tokens = tokenize(input.get(0));
            return processTokens(env, tokens);
        } else if (!useMask) {
            // If we're not masking, then we can't pad the input without affecting the embedding
            // so it goes bang.
            throw new IllegalStateException("Model must support the attention mask argument to use batches");
        } else {
            // May need padding
            int length = -1;
            var tokens = new ArrayList<List<String>>(input.size());
            for (var s : input) {
                var t = tokenize(s);
                tokens.add(t);
                length = Math.max(length, t.size());
            }
            return innerProcessTokensBatch(env, tokens, length);
        }
    }

    /**
     * Produces the processed input record by converting the tokens into ids, padding & truncating as necessary and producing OnnxTensors.
     * @param env The ORT environment.
     * @param tokens The input tokens.
     * @param length The maximum length of the token lists.
     * @return The processed input.
     * @throws OrtException If the native library throws an exception.
     */
    private ProcessedInput innerProcessTokensBatch(OrtEnvironment env, List<List<String>> tokens, int length) throws OrtException {
        length += 2; // Adjust for special characters
        long[] shape = new long[] {tokens.size(), length};
        long[] lengths = new long[tokens.size()];
        // create and fill the ids.
        LongTensorBuffer tokenIdsBuf = new LongTensorBuffer(shape);
        LongTensorBuffer maskBuf = new LongTensorBuffer(shape);
        int idx = 0;
        for (var list : tokens) {
            tokenIdsBuf.buffer().put(bosTokenId);
            maskBuf.buffer().put(maskValue);
            int i = 1;
            for (var token : list) {
                int id = tokenIDs.getOrDefault(token, unkTokenId);
                tokenIdsBuf.buffer().put(id);
                maskBuf.buffer().put(maskValue);
                i++;
            }
            tokenIdsBuf.buffer().put(eosTokenId);
            i++;

            lengths[idx] = i;
            maskBuf.buffer().put(maskValue);
            for (; i < length; i++) {
                tokenIdsBuf.buffer().put(padTokenId);
                maskBuf.buffer().put(negativeMaskValue);
            }
            idx++;
        }
        tokenIdsBuf.buffer().rewind();
        maskBuf.buffer().rewind();

        OnnxTensor tokenIds = tokenIdsBuf.wrapForORT(env);
        OnnxTensor mask = maskBuf.wrapForORT(env);

        tokenIdsBuf.buffer().rewind();
        if (useTokenType) {
            OnnxTensor tokenTypes = new LongTensorBuffer(shape, tokenTypeValue).wrapForORT(env);
            return new ProcessedInput(Map.of(inputIdsName, tokenIds, maskName, mask, tokenTypeName, tokenTypes), lengths, tokenIdsBuf);
        } else {
            return new ProcessedInput(Map.of(inputIdsName, tokenIds, maskName, mask), lengths, tokenIdsBuf);
        }
    }

    @Override
    public ProcessedInput processTokensBatch(OrtEnvironment env, List<List<String>> tokens) throws OrtException {
        if (!useMask) {
            // If we're not masking, then we can't pad the input without affecting the embedding
            // so it goes bang.
            throw new IllegalStateException("Model must support the attention mask argument to use batches");
        }
        // May need padding
        int length = -1;
        for (var s : tokens) {
            length = Math.max(length, s.size());
        }
        if (length > maxLength-2) {
            throw new IllegalArgumentException("Tokens must be truncated to maxLength-2 to use this method, length = " + length + ", maxLength = " + maxLength);
        }
        return innerProcessTokensBatch(env, tokens, length);
    }

    /**
     * Processes the input tokens into an input record.
     * @param env The ONNX Runtime environment so it can construct the appropriate tensors.
     * @param tokens The pre-tokenized input to embed.
     * @return A processed input record.
     * @throws OrtException If tensor creation failed.
     */
    public ProcessedInput processTokens(OrtEnvironment env, List<String> tokens) throws OrtException {
        var shape = new long[]{1, tokens.size()+2};
        Map<String, OnnxTensor> inputs = new HashMap<>(4);
        LongBuffer buf = convertTokens(env, tokens);
        OnnxTensor tokenIds = OnnxTensor.createTensor(env,buf,shape);
        inputs.put(inputIdsName, tokenIds);
        if (useMask) {
            OnnxTensor mask = new LongTensorBuffer(shape, maskValue).wrapForORT(env);
            // TODO: we should be setting the rest of this tensor to use the negativeMaskValue rather than accepting the zero default
            inputs.put(maskName, mask);
        }
        if (useTokenType) {
            OnnxTensor tokenTypes = new LongTensorBuffer(shape, tokenTypeValue).wrapForORT(env);
            inputs.put(tokenTypeName, tokenTypes);
        }
        return new ProcessedInput(inputs, new long[]{tokens.size()+2}, new LongTensorBuffer(buf, shape));
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "InputProcessor");
    }

    /**
     * Tokenizes the input using the loaded tokenizer, truncates the
     * token list if it's longer than {@code maxLength} - 2 (to account
     * for [CLS] and [SEP] tokens).
     * @param data The input text.
     * @return The wordpiece tokens for the supplied data.
     */
    public List<String> tokenize(String data) {
        List<String> tokens = tokenizer.split(data);
        if (tokens.size() > (maxLength - 2)) {
            logger.fine("Truncating sentence to " + (maxLength + 2) + " from " + tokens.size());
            tokens = tokens.subList(0,maxLength-2);
        }
        return tokens;
    }

    /**
     * Converts a token list into the appropriate tensor for ORT.
     * @param tokens The tokens to convert.
     * @return An OnnxTensor representing the input, with the appropriate start and end tokens.
     * @throws OrtException if it failed to create the tensor.
     */
    protected LongBuffer convertTokens(OrtEnvironment env, List<String> tokens) throws OrtException {
        int size = tokens.size() + 2; // for [CLS] and [SEP]
        LongBuffer buf = ByteBuffer.allocateDirect(size*8).order(ByteOrder.nativeOrder()).asLongBuffer();

        buf.put(bosTokenId);
        for (String token : tokens) {
            int id = tokenIDs.getOrDefault(token,unkTokenId);
            buf.put(id);
        }
        buf.put(eosTokenId);
        buf.rewind();

        return buf;
    }
}
