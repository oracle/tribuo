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
 *
 */
public interface TokenizerConfig extends Supplier<Tokenizer> {
    Map<String, Integer> tokenIDs();
    String unknownToken();
    String classificationToken();
    String separatorToken();
    String padToken();
    boolean lowercase();
    boolean stripAccents();
    int maxInputCharsPerWord();

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
