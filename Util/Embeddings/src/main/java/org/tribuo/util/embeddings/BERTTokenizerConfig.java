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

package org.tribuo.util.embeddings;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import com.fasterxml.jackson.databind.node.ArrayNode;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Locale;
import java.util.Map;

import org.tribuo.util.embeddings.processors.BERTInputProcessor;
import org.tribuo.util.tokens.Tokenizer;
import org.tribuo.util.tokens.impl.wordpiece.Wordpiece;
import org.tribuo.util.tokens.impl.wordpiece.WordpieceBasicTokenizer;
import org.tribuo.util.tokens.impl.wordpiece.WordpieceTokenizer;

/**
 * A Huggingface BERT style tokenizer configuration.
 */
public record BERTTokenizerConfig(Map<String, Integer> tokenIDs, String unknownToken, String classificationToken,
                           String separatorToken, String padToken, boolean lowercase, boolean stripAccents, int maxInputCharsPerWord) implements
    TokenizerConfig {

    public Tokenizer get() {
        Wordpiece wordpiece = new Wordpiece(tokenIDs().keySet(),unknownToken(),maxInputCharsPerWord());
        return new WordpieceTokenizer(wordpiece,new WordpieceBasicTokenizer(),lowercase(),stripAccents(),
            Collections.emptySet());
    }

    /**
     * Loads the tokenizer configuration out of the huggingface tokenizer json file.
     *
     * @param tokenizerPath The path to the json file.
     * @return The tokenizer configuration.
     * @throws IOException If the path could not be read.
     * @throws IllegalStateException If the JSON could not be parsed as a valid HF BERT tokenizer config.
     */
    public static BERTTokenizerConfig loadTokenizer(Path tokenizerPath) throws IOException {
        ObjectMapper mapper = new ObjectMapper(new JsonFactory());
        JsonNode rootNode = mapper.readTree(tokenizerPath.toFile());
        // The tokenizer file is a JSON object with the following schema
        /*
         * {
         *   "version": "1.0",
         *   "truncation": null,
         *   "padding": null,
         *   "added_tokens": [
         *     {
         *       "id": 0,
         *       "special": true,
         *       "content": "[PAD]",
         *       "single_word": false,
         *       "lstrip": false,
         *       "rstrip": false,
         *       "normalized": false
         *     }
         *   ],
         *   "normalizer": {
         *     "type": "BertNormalizer",
         *     "clean_text": true,
         *     "handle_chinese_chars": true,
         *     "strip_accents": null,
         *     "lowercase": false
         *   },
         *   "pre_tokenizer": {
         *     "type": "BertPreTokenizer"
         *   },
         *   "post_processor": {
         *     "type": "TemplateProcessing",
         *     "single": [
         *       {
         *         "SpecialToken": {
         *           "id": "[CLS]",
         *           "type_id": 0
         *         }
         *       },
         *       {
         *         "Sequence": {
         *           "id": "A",
         *           "type_id": 0
         *         }
         *       },
         *       {
         *         "SpecialToken": {
         *           "id": "[SEP]",
         *           "type_id": 0
         *         }
         *       }
         *     ],
         *     "pair": [
         *       {
         *         "SpecialToken": {
         *           "id": "[CLS]",
         *           "type_id": 0
         *         }
         *       },
         *       {
         *         "Sequence": {
         *           "id": "A",
         *           "type_id": 0
         *         }
         *       },
         *       {
         *         "SpecialToken": {
         *           "id": "[SEP]",
         *           "type_id": 0
         *         }
         *       },
         *       {
         *         "Sequence": {
         *           "id": "B",
         *           "type_id": 1
         *         }
         *       },
         *       {
         *         "SpecialToken": {
         *           "id": "[SEP]",
         *           "type_id": 1
         *         }
         *       }
         *     ],
         *     "special_tokens": {
         *       "[SEP]": {
         *         "id": "[SEP]",
         *         "ids": [
         *           102
         *         ],
         *         "tokens": [
         *           "[SEP]"
         *         ]
         *       },
         *       "[CLS]": {
         *         "id": "[CLS]",
         *         "ids": [
         *           101
         *         ],
         *         "tokens": [
         *           "[CLS]"
         *         ]
         *       }
         *     }
         *   },
         *   "decoder": {
         *     "type": "WordPiece",
         *     "prefix": "##",
         *     "cleanup": true
         *   },
         *   "model": {
         *     "unk_token": "[UNK]",
         *     "continuing_subword_prefix": "##",
         *     "max_input_chars_per_word": 100,
         *     "vocab": {
         *       "[PAD]": 0,
         *       ...
         *       }
         *   }
         * }
         */

        Map<String, Integer> vocabMap = new HashMap<>();
        String unknownToken;
        String classificationToken;
        String separatorToken;
        String padToken = null;
        boolean lowercase = false;
        boolean stripAccents = false;
        int maxInputCharsPerWord = 100;

        // Parse out token normalization settings
        JsonNode normalizer = rootNode.get("normalizer");
        if (normalizer != null) {
            lowercase = normalizer.get("lowercase").asBoolean();
            stripAccents = normalizer.get("strip_accents").asBoolean();
        } else {
            throw new IllegalStateException("Failed to parse tokenizer json, did not find the normalizer");
        }

        // Parse out pad token, we assume it's an "added_token" and it in some way contains the string "pad"

        JsonNode addedTokens = rootNode.get("added_tokens");
        if (addedTokens instanceof ArrayNode tokensArr) {
            for (JsonNode node : tokensArr) {
                if (node.get("content").asText().toLowerCase(Locale.US).contains("pad")) {
                    // Found it
                    padToken = node.get("content").asText();
                }
            }
            if (padToken == null) {
                throw new IllegalStateException("Failed to find pad token in tokenizer json.");
            }
        }

        // Parse out classification and separator tokens
        JsonNode postProcessor = rootNode.get("post_processor");
        if (postProcessor != null) {
            String processorType = postProcessor.get("type").asText();
            if (processorType != null && processorType.equals("TemplateProcessing")) {
                JsonNode specialTokens = postProcessor.get("special_tokens");
                if (specialTokens != null) {
                    JsonNode sepNode = specialTokens.get(BERTInputProcessor.SEPARATOR_TOKEN);
                    if (sepNode != null) {
                        separatorToken = sepNode.get("tokens").get(0).asText();
                    } else {
                        throw new IllegalStateException("Failed to parse tokenizer json, did not find separator token.");
                    }
                    JsonNode classificationNode = specialTokens.get(BERTInputProcessor.CLASSIFICATION_TOKEN);
                    if (classificationNode != null) {
                        classificationToken = classificationNode.get("tokens").get(0).asText();
                    } else {
                        throw new IllegalStateException("Failed to parse tokenizer json, did not find classification token.");
                    }
                } else {
                    throw new IllegalStateException("Failed to parse tokenizer json, did not find the special tokens.");
                }
            } else if (processorType != null && processorType.equals("BertProcessing")) {
                JsonNode sepNode = postProcessor.get("sep");
                if (sepNode != null) {
                    separatorToken = sepNode.get(0).asText();
                } else {
                    throw new IllegalStateException("Failed to parse tokenizer json, did not find separator token.");
                }
                JsonNode clsNode = postProcessor.get("cls");
                if (clsNode != null) {
                    classificationToken = clsNode.get(0).asText();
                } else {
                    throw new IllegalStateException("Failed to parse tokenizer json, did not find classification token.");
                }
            } else {
                throw new IllegalStateException("Failed to parse tokenizer json, did not recognise post_processor:type " + processorType);
            }
        } else {
            throw new IllegalStateException("Failed to parse tokenizer json, did not find the post processor");
        }

        // Parse out tokens and ids
        JsonNode model = rootNode.get("model");
        if (model != null) {
            unknownToken = model.get("unk_token").asText();
            if (unknownToken == null || unknownToken.isEmpty()) {
                throw new IllegalStateException("Failed to parse tokenizer json, did not extract unknown token");
            }
            maxInputCharsPerWord = model.get("max_input_chars_per_word").asInt();
            if (maxInputCharsPerWord == 0) {
                throw new IllegalStateException("Failed to parse tokenizer json, did not extract max_input_chars_per_word");
            }
            JsonNode vocab = model.get("vocab");
            if (vocab != null) {
                for (Iterator<Map.Entry<String, JsonNode>> termItr = vocab.fields(); termItr.hasNext(); ) {
                    Map.Entry<String, JsonNode> term = termItr.next();
                    int value = term.getValue().asInt(-1);

                    if (value == -1) {
                        throw new IllegalStateException("Failed to parse tokenizer json, could not extract vocab item '" + term.getKey() + "'");
                    } else {
                        vocabMap.put(term.getKey(), value);
                    }
                }
            } else {
                throw new IllegalStateException("Failed to parse tokenizer json, did not extract vocab");
            }
        } else {
            throw new IllegalStateException("Failed to parse tokenizer json, did not find the model");
        }
        return new BERTTokenizerConfig(vocabMap, unknownToken, classificationToken, separatorToken, padToken, lowercase, stripAccents, maxInputCharsPerWord);
    }
}
