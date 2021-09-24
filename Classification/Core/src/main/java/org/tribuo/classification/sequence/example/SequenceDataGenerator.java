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

package org.tribuo.classification.sequence.example;

import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.impl.ListExample;
import org.tribuo.provenance.SimpleDataSourceProvenance;
import org.tribuo.sequence.MutableSequenceDataset;
import org.tribuo.sequence.SequenceExample;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * A data generator for smoke testing sequence label models.
 */
public final class SequenceDataGenerator {

    private static final LabelFactory labelFactory = new LabelFactory();

    private SequenceDataGenerator() { }

    /**
     * Generates a simple dataset consisting of numCopies repeats of two sequences.
     * @param numCopies The number of times to repeat the two sequence examples.
     * @return The dataset.
     */
    public static MutableSequenceDataset<Label> generateGorillaDataset(int numCopies) {
        List<SequenceExample<Label>> examples = new ArrayList<>();

        for (int i = 0; i < numCopies; i++) {
            examples.add(generateGorillaA());
            examples.add(generateGorillaB());
        }

        return new MutableSequenceDataset<>(examples, new SimpleDataSourceProvenance("ExampleSequenceDataset", OffsetDateTime.now(),labelFactory),labelFactory);
    }

    /**
     * Generates a sequence example with a mixture of features and three labels "O", "Status" and "Monkey".
     * @return A sequence example.
     */
    public static SequenceExample<Label> generateGorillaA() {
        //"The silverback gorilla is angry"
        List<Example<Label>> examples = new ArrayList<>();

        Example<Label> the = new ListExample<>(new Label("O"));
        the.add(new Feature("A",1.0));
        the.add(new Feature("B",1.0));
        the.add(new Feature("W=the",1.0));
        examples.add(the);

        Example<Label> silverback = new ListExample<>(new Label("Monkey"));
        silverback.add(new Feature("C",1.0));
        silverback.add(new Feature("D",1.0));
        silverback.add(new Feature("W=silverback",1.0));
        examples.add(silverback);

        Example<Label> gorilla = new ListExample<>(new Label("Monkey"));
        gorilla.add(new Feature("D",1.0));
        gorilla.add(new Feature("E",1.0));
        gorilla.add(new Feature("W=gorilla",1.0));
        examples.add(gorilla);

        Example<Label> is = new ListExample<>(new Label("O"));
        is.add(new Feature("B",1.0));
        is.add(new Feature("W=is",1.0));
        examples.add(is);

        Example<Label> angry = new ListExample<>(new Label("Status"));
        angry.add(new Feature("F",1.0));
        angry.add(new Feature("G",1.0));
        angry.add(new Feature("W=angry",1.0));
        examples.add(angry);

        return new SequenceExample<>(examples);
    }

    /**
     * Generates a sequence example with a mixture of features and three labels "O", "Status" and "Monkey".
     * @return A sequence example.
     */
    public static SequenceExample<Label> generateGorillaB() {
        //"That is one angry looking gorilla"
        List<Example<Label>> examples = new ArrayList<>();

        Example<Label> that = new ListExample<>(new Label("O"));
        that.add(new Feature("A",1.0));
        that.add(new Feature("B",1.0));
        that.add(new Feature("W=that",1.0));
        examples.add(that);

        Example<Label> is = new ListExample<>(new Label("O"));
        is.add(new Feature("B",1.0));
        is.add(new Feature("W=is",1.0));
        examples.add(is);

        Example<Label> one = new ListExample<>(new Label("O"));
        one.add(new Feature("B",1.0));
        one.add(new Feature("H",1.0));
        one.add(new Feature("W=one",1.0));
        examples.add(one);

        Example<Label> angry = new ListExample<>(new Label("Status"));
        angry.add(new Feature("F",1.0));
        angry.add(new Feature("G",1.0));
        angry.add(new Feature("W=angry",1.0));
        examples.add(angry);

        Example<Label> looking = new ListExample<>(new Label("O"));
        looking.add(new Feature("I",1.0));
        looking.add(new Feature("J",1.0));
        looking.add(new Feature("W=looking",1.0));
        examples.add(looking);

        Example<Label> gorilla = new ListExample<>(new Label("Monkey"));
        gorilla.add(new Feature("D",1.0));
        gorilla.add(new Feature("E",1.0));
        gorilla.add(new Feature("W=gorilla",1.0));
        examples.add(gorilla);

        return new SequenceExample<>(examples);
    }

    /**
     * This generates a sequence example with features that are unused by the training data.
     * @return A {@link SequenceExample} which is invalid in the context of the Gorilla example data.
     */
    public static SequenceExample<Label> generateInvalidExample() {
        //"invalid example"
        List<Example<Label>> examples = new ArrayList<>();

        Example<Label> invalid = new ListExample<>(new Label("O"));
        invalid.add(new Feature("1",1.0));
        invalid.add(new Feature("2",1.0));
        invalid.add(new Feature("W=invalid",1.0));
        examples.add(invalid);

        Example<Label> example = new ListExample<>(new Label("O"));
        example.add(new Feature("3",1.0));
        example.add(new Feature("2",1.0));
        example.add(new Feature("W=example",1.0));
        examples.add(example);

        return new SequenceExample<>(examples);
    }

    /**
     * This generates a sequence example where the first example has no features.
     * @return A {@link SequenceExample} which is invalid as one example contains no features.
     */
    public static SequenceExample<Label> generateOtherInvalidExample() {
        //"invalid example"
        List<Example<Label>> examples = new ArrayList<>();

        Example<Label> invalid = new ListExample<>(new Label("O"));
        examples.add(invalid);

        Example<Label> example = new ListExample<>(new Label("O"));
        example.add(new Feature("3",1.0));
        example.add(new Feature("2",1.0));
        example.add(new Feature("W=example",1.0));
        examples.add(example);

        return new SequenceExample<>(examples);
    }

    /**
     * This generates a sequence example with no examples.
     * @return A {@link SequenceExample} which is invalid as it contains no examples.
     */
    public static SequenceExample<Label> generateEmptyExample() {
        List<Example<Label>> examples = new ArrayList<>();
        return new SequenceExample<>(examples);
    }
}
