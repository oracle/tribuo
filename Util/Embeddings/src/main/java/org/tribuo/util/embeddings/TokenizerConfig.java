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

import java.lang.reflect.InvocationTargetException;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.Supplier;
import org.tribuo.util.tokens.Tokenizer;

/**
 * An interface for tokenizer configuration objects, typically loaded from JSON.
 */
public interface TokenizerConfig extends Supplier<Tokenizer> {
    /**
     * The token vocab mapping.
     * @return The token vocab.
     */
    Map<String, Integer> tokenIDs();

    /**
     * The unknown token string.
     * @return The unknown token.
     */
    String unknownToken();

    /**
     * The equivalent of the {@code [CLS]} token, may be {@code [BOS]} in some models.
     * @return The classification token.
     */
    String classificationToken();

    /**
     * The equivalent of the {@code [SEP]} token, may be {@code [EOS]} in some models.
     * @return The separator token.
     */
    String separatorToken();

    /**
     * The pad token string.
     * @return The pad token.
     */
    String padToken();

    /**
     * Should this tokenizer lowercase all the inputs before executing the tokenization algorithm?
     * @return Lowercase inputs.
     */
    boolean lowercase();

    /**
     * Should this tokenizer strip off accent markers from characters?
     * @return Strip accent markers.
     */
    boolean stripAccents();

    /**
     * The maximum size of a token in characters.
     * @return The maximum token length.
     */
    int maxInputCharsPerWord();

    /**
     * Loads a tokenizer config instance from the supplied path.
     * @param configClass The tokenizer configuration class.
     * @param path The path to the config JSON.
     * @return The loaded tokenizer config.
     */
    public static TokenizerConfig loadTokenizer(Class<?> configClass, Path path) {
        try {
            var method = configClass.getDeclaredMethod("loadTokenizer", Path.class);
            return (TokenizerConfig) method.invoke(null, path);
        } catch (NoSuchMethodException |  IllegalAccessException e) {
            throw new IllegalArgumentException("Class '" + configClass + "' did not expose a 'loadTokenizer(Path)' static method", e);
        } catch (InvocationTargetException e) {
            throw new IllegalArgumentException("Failed to loadTokenizer(Path) for Path " + path,e);
        }
    }
}
