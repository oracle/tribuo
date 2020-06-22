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

package org.tribuo.data.text.impl;

import org.tribuo.data.text.TextFeatureExtractor;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;
import org.junit.jupiter.api.Test;
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.assertThrows;


public class SimpleStringDataSourceTest {

    @Test
    public void testEmptyRawLines() {
        List<String> rawLines = new ArrayList<>();
        SimpleStringDataSource<MockOutput> src = new SimpleStringDataSource<>(rawLines, new MockOutputFactory(), getFeatureExtractor());
        assertThrows(IllegalStateException.class, src::iterator);
    }

    public static TextFeatureExtractor<MockOutput> getFeatureExtractor() {
        TokenPipeline pl = new TokenPipeline(new BreakIteratorTokenizer(Locale.US), 1, false);
        TextFeatureExtractor<MockOutput> ex = new TextFeatureExtractorImpl<>(pl);
        return ex;
    }

}