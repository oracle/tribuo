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

package org.tribuo.interop.onnx.extractors;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class BERTFeatureExtractorTest {

    @Test
    public void testTokenizerLoading() throws URISyntaxException, IOException {
        Path vocabPath = Paths.get(BERTFeatureExtractorTest.class.getResource("bert-base-cased-vocab.txt").toURI());
        Path tokenizerPath = Paths.get(BERTFeatureExtractorTest.class.getResource("bert-base-cased-tokenizer.json").toURI());
        BERTFeatureExtractor.TokenizerConfig config = BERTFeatureExtractor.loadTokenizer(tokenizerPath);

        List<String> vocabList = Files.readAllLines(vocabPath, StandardCharsets.UTF_8);

        Assertions.assertEquals(config.tokenIDs.size(),vocabList.size());

        for (String vocabElement : vocabList) {
            Assertions.assertTrue(config.tokenIDs.containsKey(vocabElement));
        }

        Assertions.assertEquals(100, config.maxInputCharsPerWord);
        Assertions.assertEquals(false, config.lowercase);
        Assertions.assertEquals(false, config.stripAccents);
        Assertions.assertEquals("[UNK]",config.unknownToken);
        Assertions.assertEquals("[CLS]",config.classificationToken);
        Assertions.assertEquals("[SEP]",config.separatorToken);
    }

}
