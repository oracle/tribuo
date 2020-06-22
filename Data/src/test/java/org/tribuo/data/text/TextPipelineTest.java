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

package org.tribuo.data.text;

import org.tribuo.Feature;
import org.tribuo.data.text.impl.BasicPipeline;
import org.tribuo.data.text.impl.TokenPipeline;
import org.junit.jupiter.api.Test;
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;

import java.util.List;
import java.util.Locale;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 *
 */
public class TextPipelineTest {

    private static final Logger logger = Logger.getLogger(TextPipelineTest.class.getName());

    @Test
    public void testBasicPipeline() {
        String input = "This is some input text.";

        BasicPipeline pipeline = new BasicPipeline(new BreakIteratorTokenizer(Locale.US),2);

        List<Feature> featureList = pipeline.process("",input);

        //logger.log(Level.INFO,featureList.toString());

        assertTrue(featureList.contains(new Feature("1-N=This",1.0)));
        assertTrue(featureList.contains(new Feature("1-N=is",1.0)));
        assertTrue(featureList.contains(new Feature("1-N=some",1.0)));
        assertTrue(featureList.contains(new Feature("1-N=input",1.0)));
        assertTrue(featureList.contains(new Feature("1-N=text",1.0)));

        assertTrue(featureList.contains(new Feature("2-N=This/is",1.0)));
        assertTrue(featureList.contains(new Feature("2-N=is/some",1.0)));
        assertTrue(featureList.contains(new Feature("2-N=some/input",1.0)));
        assertTrue(featureList.contains(new Feature("2-N=input/text",1.0)));
    }

    @Test
    public void testBasicPipelineTagging() {
        String input = "This is some input text.";

        BasicPipeline pipeline = new BasicPipeline(new BreakIteratorTokenizer(Locale.US),2);

        List<Feature> featureList = pipeline.process("Monkeys",input);

        //logger.log(Level.INFO,featureList.toString());

        assertTrue(featureList.contains(new Feature("Monkeys-1-N=This",1.0)));
        assertTrue(featureList.contains(new Feature("Monkeys-1-N=is",1.0)));
        assertTrue(featureList.contains(new Feature("Monkeys-1-N=some",1.0)));
        assertTrue(featureList.contains(new Feature("Monkeys-1-N=input",1.0)));
        assertTrue(featureList.contains(new Feature("Monkeys-1-N=text",1.0)));

        assertTrue(featureList.contains(new Feature("Monkeys-2-N=This/is",1.0)));
        assertTrue(featureList.contains(new Feature("Monkeys-2-N=is/some",1.0)));
        assertTrue(featureList.contains(new Feature("Monkeys-2-N=some/input",1.0)));
        assertTrue(featureList.contains(new Feature("Monkeys-2-N=input/text",1.0)));
    }

    @Test
    public void testTokenPipeline() {
        String input = "This is some input text.";

        TokenPipeline pipeline = new TokenPipeline(new BreakIteratorTokenizer(Locale.US),2,true);

        List<Feature> featureList = pipeline.process("",input);

        //logger.log(Level.INFO,featureList.toString());

        assertTrue(featureList.contains(new Feature("1-N=This",1.0)));
        assertTrue(featureList.contains(new Feature("1-N=is",1.0)));
        assertTrue(featureList.contains(new Feature("1-N=some",1.0)));
        assertTrue(featureList.contains(new Feature("1-N=input",1.0)));
        assertTrue(featureList.contains(new Feature("1-N=text",1.0)));

        assertTrue(featureList.contains(new Feature("2-N=This/is",1.0)));
        assertTrue(featureList.contains(new Feature("2-N=is/some",1.0)));
        assertTrue(featureList.contains(new Feature("2-N=some/input",1.0)));
        assertTrue(featureList.contains(new Feature("2-N=input/text",1.0)));
    }

    @Test
    public void testTokenPipelineTagging() {
        String input = "This is some input text.";

        TokenPipeline pipeline = new TokenPipeline(new BreakIteratorTokenizer(Locale.US),2,true);

        List<Feature> featureList = pipeline.process("Monkeys",input);

        //logger.log(Level.INFO,featureList.toString());

        assertTrue(featureList.contains(new Feature("Monkeys-1-N=This",1.0)));
        assertTrue(featureList.contains(new Feature("Monkeys-1-N=is",1.0)));
        assertTrue(featureList.contains(new Feature("Monkeys-1-N=some",1.0)));
        assertTrue(featureList.contains(new Feature("Monkeys-1-N=input",1.0)));
        assertTrue(featureList.contains(new Feature("Monkeys-1-N=text",1.0)));

        assertTrue(featureList.contains(new Feature("Monkeys-2-N=This/is",1.0)));
        assertTrue(featureList.contains(new Feature("Monkeys-2-N=is/some",1.0)));
        assertTrue(featureList.contains(new Feature("Monkeys-2-N=some/input",1.0)));
        assertTrue(featureList.contains(new Feature("Monkeys-2-N=input/text",1.0)));
    }
}
