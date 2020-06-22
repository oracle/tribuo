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

package org.tribuo.classification.explanations.lime;

import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.SparseModel;
import org.tribuo.SparseTrainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.explanations.TextExplainer;
import org.tribuo.data.text.TextFeatureExtractor;
import org.tribuo.impl.ArrayExample;
import org.tribuo.provenance.SimpleDataSourceProvenance;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.util.tokens.Token;
import org.tribuo.util.tokens.Tokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.SplittableRandom;
import java.util.logging.Logger;

/**
 * Uses a Tribuo {@link TextFeatureExtractor} to explain the prediction for a given piece of text.
 * <p>
 * LIME uses a naive sampling procedure which blanks out words and trains the linear model on
 * a set of binary features, each of which is the presence of a word+position combination. Words
 * are not permuted, and new words are not added (so it's only explaining when the absence of a
 * word would change the prediction).
 * <p>
 * See:
 * <pre>
 * Ribeiro MT, Singh S, Guestrin C.
 * "Why should I trust you?: Explaining the predictions of any classifier"
 * Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining 2016.
 * </pre>
 */
public class LIMEText extends LIMEBase implements TextExplainer<Regressor> {

    private static final Logger logger = Logger.getLogger(LIMEText.class.getName());

    private final TextFeatureExtractor<Label> extractor;

    private final Tokenizer tokenizer;

    private final ThreadLocal<Tokenizer> tokenizerThreadLocal;

    /**
     * Constructs a LIME explainer for a model which uses text data.
     * @param rng The rng to use for sampling.
     * @param innerModel The model to explain.
     * @param explanationTrainer The sparse trainer to use to generate explanations.
     * @param numSamples The number of samples to generate for each explanation.
     * @param extractor The {@link TextFeatureExtractor} used to generate text features from a string.
     * @param tokenizer The tokenizer used to tokenize the examples.
     */
    public LIMEText(SplittableRandom rng, Model<Label> innerModel, SparseTrainer<Regressor> explanationTrainer, int numSamples, TextFeatureExtractor<Label> extractor, Tokenizer tokenizer) {
        super(rng, innerModel, explanationTrainer, numSamples);
        this.extractor = extractor;
        this.tokenizer = tokenizer;
        this.tokenizerThreadLocal = ThreadLocal.withInitial(() -> {try { return this.tokenizer.clone(); } catch (CloneNotSupportedException e) { throw new IllegalArgumentException("Tokenizer not cloneable",e); }});
    }

    @Override
    public LIMEExplanation explain(String inputText) {
        Example<Label> trueExample = extractor.extract(LabelFactory.UNKNOWN_LABEL, inputText);
        Prediction<Label> prediction = innerModel.predict(trueExample);

        ArrayExample<Regressor> bowExample = new ArrayExample<>(transformOutput(prediction));
        List<Token> tokens = tokenizerThreadLocal.get().tokenize(inputText);
        for (int i = 0; i < tokens.size(); i++) {
            bowExample.add(nameFeature(tokens.get(i).text,i),1.0);
        }

        // Sample a dataset.
        List<Example<Regressor>> sample = sampleData(inputText,tokens);

        // Generate a sparse model on the sampled data.
        SparseModel<Regressor> model = trainExplainer(bowExample, sample);

        // Test the sparse model against the predictions of the real model.
        List<Prediction<Regressor>> predictions = new ArrayList<>(model.predict(sample));
        predictions.add(model.predict(bowExample));
        RegressionEvaluation evaluation = evaluator.evaluate(model,predictions,new SimpleDataSourceProvenance("LIMEText sampled data",regressionFactory));

        return new LIMEExplanation(model, prediction, evaluation);
    }

    /**
     * Generate the feature name by combining the word and index.
     * @param name The word.
     * @param idx The index.
     * @return A string representing both of the inputs.
     */
    protected String nameFeature(String name, int idx) {
        return name+"@idx"+idx;
    }

    /**
     * Samples a new dataset from the input text. Uses the tokenized representation,
     * removes words by blanking them out. Only removes words to generate a new sentence,
     * and does not generate the empty sentence.
     * @param inputText The input text.
     * @param tokens The tokenized representation of the input text.
     * @return A list of samples from the input text.
     */
    protected List<Example<Regressor>> sampleData(String inputText, List<Token> tokens) {
        List<Example<Regressor>> output = new ArrayList<>();

        Random innerRNG = new Random(rng.nextLong());
        for (int i = 0; i < numSamples; i++) {
            // Sample a new Example.
            double distance = 0.0;
            int[] activeFeatures = new int[tokens.size()];
            char[] sampledText = inputText.toCharArray();
            for (int j = 0; j < activeFeatures.length; j++) {
                activeFeatures[j] = innerRNG.nextInt(2);
                if (activeFeatures[j] == 0) {
                    distance++;
                    Token curToken = tokens.get(j);
                    Arrays.fill(sampledText,curToken.start,curToken.end,'\0');
                }
            }
            String sampledString = new String(sampledText);
            sampledString = sampledString.replace("\0","");

            Example<Label> sample = extractor.extract(LabelFactory.UNKNOWN_LABEL,sampledString);

            // If the sample has features.
            if (sample.size() > 0) {
                // Label it using the full model.
                Prediction<Label> samplePrediction = innerModel.predict(sample);

                // Transform distance into a weight.
                double weight = 1.0 - (distance / tokens.size());

                // Generate the new sample with the appropriate label and weight.
                ArrayExample<Regressor> labelledSample = new ArrayExample<>(transformOutput(samplePrediction), (float) weight);
                for (int j = 0; j < activeFeatures.length; j++) {
                    labelledSample.add(nameFeature(tokens.get(j).text, j), activeFeatures[j]);
                }
                output.add(labelledSample);
            }
        }

        return output;
    }
}
