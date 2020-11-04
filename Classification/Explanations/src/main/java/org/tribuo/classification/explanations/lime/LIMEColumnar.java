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

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.CategoricalInfo;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.RealInfo;
import org.tribuo.SparseModel;
import org.tribuo.SparseTrainer;
import org.tribuo.VariableIDInfo;
import org.tribuo.VariableInfo;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.explanations.ColumnarExplainer;
import org.tribuo.data.columnar.ColumnarFeature;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.ResponseProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.impl.ArrayExample;
import org.tribuo.impl.ListExample;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.SimpleDataSourceProvenance;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.util.Util;
import org.tribuo.util.tokens.Token;
import org.tribuo.util.tokens.Tokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.SplittableRandom;

/**
 * Uses the columnar data processing infrastructure to mix text and tabular data.
 * <p>
 * If the supplied {@link RowProcessor} doesn't reference any text or binarised fields
 * then it delegates to {@link LIMEBase#explain}, though it's still more expensive at
 * construction time.
 * <p>
 * See:
 * <pre>
 * Ribeiro MT, Singh S, Guestrin C.
 * "Why should I trust you?: Explaining the predictions of any classifier"
 * Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining 2016.
 * </pre>
 */
public class LIMEColumnar extends LIMEBase implements ColumnarExplainer<Regressor> {

    private final RowProcessor<Label> generator;

    private final Map<String,FieldProcessor> binarisedFields = new HashMap<>();

    private final Map<String,FieldProcessor> tabularFields = new HashMap<>();

    private final Map<String,FieldProcessor> textFields = new HashMap<>();

    private final ResponseProcessor<Label> responseProcessor;

    private final Map<String,List<VariableInfo>> binarisedInfos;

    private final Map<String,double[]> binarisedCDFs;

    private final ImmutableFeatureMap binarisedDomain;

    private final ImmutableFeatureMap tabularDomain;

    private final ImmutableFeatureMap textDomain;

    private final Tokenizer tokenizer;

    private final ThreadLocal<Tokenizer> tokenizerThreadLocal;

    /**
     * Constructs a LIME explainer for a model which uses the columnar data processing system.
     * @param rng The rng to use for sampling.
     * @param innerModel The model to explain.
     * @param explanationTrainer The trainer for the sparse model used to explain.
     * @param numSamples The number of samples to generate in each explanation.
     * @param exampleGenerator The {@link RowProcessor} which converts columnar data into an {@link Example}.
     * @param tokenizer The tokenizer to use on any text fields.
     */
    public LIMEColumnar(SplittableRandom rng, Model<Label> innerModel, SparseTrainer<Regressor> explanationTrainer,
                        int numSamples, RowProcessor<Label> exampleGenerator, Tokenizer tokenizer) {
        super(rng, innerModel, explanationTrainer, numSamples);
        this.generator = exampleGenerator.copy();
        this.responseProcessor = generator.getResponseProcessor();
        this.tokenizer = tokenizer;
        this.tokenizerThreadLocal = ThreadLocal.withInitial(() -> {try { return this.tokenizer.clone(); } catch (CloneNotSupportedException e) { throw new IllegalArgumentException("Tokenizer not cloneable",e); }});
        if (!this.generator.isConfigured()) {
            this.generator.expandRegexMapping(innerModel);
        }
        this.binarisedInfos = new HashMap<>();
        ArrayList<VariableInfo> infos = new ArrayList<>();
        for (VariableInfo i : innerModel.getFeatureIDMap()) {
            infos.add(i);
        }
        ArrayList<VariableInfo> allBinarisedInfos = new ArrayList<>();
        ArrayList<VariableInfo> tabularInfos = new ArrayList<>();
        ArrayList<VariableInfo> textInfos = new ArrayList<>();
        for (Map.Entry<String,FieldProcessor> p : generator.getFieldProcessors().entrySet()) {
            String searchName = p.getKey() + ColumnarFeature.JOINER;
            switch (p.getValue().getFeatureType()) {
                case BINARISED_CATEGORICAL: {
                    int numNamespaces = p.getValue().getNumNamespaces();
                    if (numNamespaces > 1) {
                        for (int i = 0; i < numNamespaces; i++) {
                            String namespace = p.getKey() + FieldProcessor.NAMESPACE + i;
                            String namespaceSearchName = namespace + ColumnarFeature.JOINER;
                            binarisedFields.put(namespace, p.getValue());
                            List<VariableInfo> binarisedInfoList = this.binarisedInfos.computeIfAbsent(namespace, (k) -> new ArrayList<>());
                            ListIterator<VariableInfo> li = infos.listIterator();
                            while (li.hasNext()) {
                                VariableInfo info = li.next();
                                if (info.getName().startsWith(namespaceSearchName)) {
                                    if (((CategoricalInfo) info).getUniqueObservations() != 1) {
                                        throw new IllegalStateException("Processor " + p.getKey() + ", should have been binary, but had " + ((CategoricalInfo) info).getUniqueObservations() + " unique values");
                                    }
                                    binarisedInfoList.add(info);
                                    allBinarisedInfos.add(info);
                                    li.remove();
                                }
                            }
                        }
                    } else {
                        binarisedFields.put(p.getKey(), p.getValue());
                        List<VariableInfo> binarisedInfoList = this.binarisedInfos.computeIfAbsent(p.getKey(), (k) -> new ArrayList<>());
                        ListIterator<VariableInfo> li = infos.listIterator();
                        while (li.hasNext()) {
                            VariableInfo i = li.next();
                            if (i.getName().startsWith(searchName)) {
                                if (((CategoricalInfo) i).getUniqueObservations() != 1) {
                                    throw new IllegalStateException("Processor " + p.getKey() + ", should have been binary, but had " + ((CategoricalInfo) i).getUniqueObservations() + " unique values");
                                }
                                binarisedInfoList.add(i);
                                allBinarisedInfos.add(i);
                                li.remove();
                            }
                        }
                    }
                    break;
                }
                case CATEGORICAL:
                case REAL: {
                    tabularFields.put(p.getKey(), p.getValue());
                    ListIterator<VariableInfo> li = infos.listIterator();
                    while (li.hasNext()) {
                        VariableInfo i = li.next();
                        if (i.getName().startsWith(searchName)) {
                            tabularInfos.add(i);
                            li.remove();
                        }
                    }
                    break;
                }
                case TEXT: {
                    textFields.put(p.getKey(), p.getValue());
                    ListIterator<VariableInfo> li = infos.listIterator();
                    while (li.hasNext()) {
                        VariableInfo i = li.next();
                        if (i.getName().startsWith(searchName)) {
                            textInfos.add(i);
                            li.remove();
                        }
                    }
                    break;
                }
                default:
                    throw new IllegalArgumentException("Unsupported feature type " + p.getValue().getFeatureType());
            }
        }
        if (infos.size() != 0) {
            throw new IllegalArgumentException("Found " + infos.size() + " unsupported features.");
        }
        if (generator.getFeatureProcessors().size() != 0) {
            throw new IllegalArgumentException("LIMEColumnar does not support FeatureProcessors.");
        }
        this.tabularDomain = new ImmutableFeatureMap(tabularInfos);
        this.textDomain = new ImmutableFeatureMap(textInfos);
        this.binarisedDomain = new ImmutableFeatureMap(allBinarisedInfos);
        this.binarisedCDFs = new HashMap<>();
        for (Map.Entry<String,List<VariableInfo>> e : binarisedInfos.entrySet()) {
            long totalCount = 0;
            long[] counts = new long[e.getValue().size()+1];
            int i = 0;
            for (VariableInfo info : e.getValue()) {
                long curCount = info.getCount();
                counts[i] = curCount;
                totalCount += curCount;
                i++;
            }
            long zeroCount = numTrainingExamples - totalCount;
            if (zeroCount < 0) {
                throw new IllegalStateException("Processor " + e.getKey() + " purports to be a BINARISED_CATEGORICAL, but had overlap in it's elements");
            }
            counts[i] = zeroCount;
            double[] cdf = Util.generateCDF(counts,numTrainingExamples);
            binarisedCDFs.put(e.getKey(),cdf);
        }
    }

    @Override
    public LIMEExplanation explain(Map<String, String> input) {
        return explainWithSamples(input).getA();
    }

    protected Pair<LIMEExplanation, List<Example<Regressor>>> explainWithSamples(Map<String, String> input) {
        Optional<Example<Label>> optExample = generator.generateExample(input,false);
        if (optExample.isPresent()) {
            Example<Label> example = optExample.get();
            if ((textDomain.size() == 0) && (binarisedCDFs.size() == 0)) {
                // Short circuit if there are no text or binarised fields.
                return explainWithSamples(example);
            } else {
                Prediction<Label> prediction = innerModel.predict(example);

                // Build the input example with simplified text features
                ArrayExample<Regressor> labelledExample = new ArrayExample<>(transformOutput(prediction));

                // Add the tabular features
                for (Feature f : example) {
                    if (tabularDomain.getID(f.getName()) != -1) {
                        labelledExample.add(f);
                    }
                }
                // Extract the tabular features into a SparseVector for later
                SparseVector tabularVector = SparseVector.createSparseVector(labelledExample,tabularDomain,false);

                // Tokenize the text fields, and generate the perturbed text representation
                Map<String, String> exampleTextValues = new HashMap<>();
                Map<String, List<Token>> exampleTextTokens = new HashMap<>();
                for (Map.Entry<String,FieldProcessor> e : textFields.entrySet()) {
                    String value = input.get(e.getKey());
                    if (value != null) {
                        List<Token> tokens = tokenizerThreadLocal.get().tokenize(value);
                        for (int i = 0; i < tokens.size(); i++) {
                            labelledExample.add(nameFeature(e.getKey(),tokens.get(i).text,i),1.0);
                        }
                        exampleTextValues.put(e.getKey(),value);
                        exampleTextTokens.put(e.getKey(),tokens);
                    }
                }

                // Sample a dataset.
                List<Example<Regressor>> sample = sampleData(tabularVector,exampleTextValues,exampleTextTokens);

                // Generate a sparse model on the sampled data.
                SparseModel<Regressor> model = trainExplainer(labelledExample, sample);

                // Test the sparse model against the predictions of the real model.
                List<Prediction<Regressor>> predictions = new ArrayList<>(model.predict(sample));
                predictions.add(model.predict(labelledExample));
                RegressionEvaluation evaluation = evaluator.evaluate(model,predictions,new SimpleDataSourceProvenance("LIMEColumnar sampled data",regressionFactory));

                return new Pair<>(new LIMEExplanation(model, prediction, evaluation),sample);
            }
        } else {
            throw new IllegalArgumentException("Label not found in input " + input.toString());
        }
    }

    /**
     * Generate the feature name by combining the word and index.
     * @param fieldName The name of the column this text feature came from.
     * @param name The word.
     * @param idx The index.
     * @return A string representing both of the inputs.
     */
    protected String nameFeature(String fieldName, String name, int idx) {
        return fieldName + "@" + name+"@idx"+idx;
    }

    /**
     * Samples a dataset based on the provided text, tokens and tabular features.
     *
     * The text features are sampled using the {@link LIMEText} sampling approach,
     * and the tabular features are sampled using the {@link LIMEBase} approach.
     *
     * The weight for each example is based on the distance for the tabular features,
     * combined with the distance for the text features (which is a hamming distance).
     * These distances are averaged using a weight function representing how many tokens
     * there are in the text fields, and how many tabular features there are.
     *
     * This weight calculation is subject to change, as it's not necessarily optimal.
     * @param tabularVector The tabular (i.e., non-text) features.
     * @param text A map from the field names to the field values for the text fields.
     * @param textTokens A map from the field names to lists of tokens for those fields.
     * @return A sampled dataset.
     */
    private List<Example<Regressor>> sampleData(SparseVector tabularVector, Map<String,String> text, Map<String,List<Token>> textTokens) {
        List<Example<Regressor>> output = new ArrayList<>();

        Random innerRNG = new Random(rng.nextLong());
        for (int i = 0; i < numSamples; i++) {
            // Create the full example
            ListExample<Label> sampledExample = new ListExample<>(LabelFactory.UNKNOWN_LABEL);

            // Tabular features.
            List<Feature> tabularFeatures = new ArrayList<>();
            // Sample the categorical and real features
            for (VariableInfo info : tabularDomain) {
                int id = ((VariableIDInfo) info).getID();
                double inputValue = tabularVector.get(id);

                if (info instanceof CategoricalInfo) {
                    // This one is tricksy as categorical info essentially implicitly includes a zero.
                    CategoricalInfo catInfo = (CategoricalInfo) info;
                    double sample = catInfo.frequencyBasedSample(innerRNG,numTrainingExamples);
                    // If we didn't sample zero.
                    if (Math.abs(sample) > 1e-10) {
                        Feature newFeature = new Feature(info.getName(),sample);
                        tabularFeatures.add(newFeature);
                    }
                } else if (info instanceof RealInfo) {
                    RealInfo realInfo = (RealInfo) info;
                    // As realInfo is sparse we sample from the mixture distribution,
                    // either 0 or N(inputValue,variance).
                    // This assumes realInfo never observed a zero, which is enforced from v2.1
                    // TODO check this makes sense. If the input value is zero do we still want to sample spike and slab?
                    // If it's not zero do we want to?
                    int count = realInfo.getCount();
                    double threshold = count / ((double)numTrainingExamples);
                    if (innerRNG.nextDouble() < threshold) {
                        double variance = realInfo.getVariance();
                        double sample = (innerRNG.nextGaussian() * Math.sqrt(variance)) + inputValue;
                        Feature newFeature = new Feature(info.getName(),sample);
                        tabularFeatures.add(newFeature);
                    }
                } else {
                    throw new IllegalStateException("Unsupported info type, expected CategoricalInfo or RealInfo, found " + info.getClass().getName());
                }
            }
            // Sample the binarised categorical features
            for (Map.Entry<String,double[]> e : binarisedCDFs.entrySet()) {
                // Sample from the CDF
                int sample = Util.sampleFromCDF(e.getValue(),innerRNG);
                // If the sample isn't zero (which is defined to be the last value to make the indices work)
                if (sample != (e.getValue().length-1)) {
                    VariableInfo info = binarisedInfos.get(e.getKey()).get(sample);
                    Feature newFeature = new Feature(info.getName(),1);
                    tabularFeatures.add(newFeature);
                }
            }
            // Add the tabular features to the current example
            sampledExample.addAll(tabularFeatures);
            // Calculate tabular distance
            double tabularDistance = measureDistance(tabularDomain,numTrainingExamples,tabularVector, SparseVector.createSparseVector(sampledExample,tabularDomain,false));

            // features are the full text features
            List<Feature> textFeatures = new ArrayList<>();
            // Perturbed features are the binarised tokens
            List<Feature> perturbedFeatures = new ArrayList<>();

            // Sample the text features
            double textDistance = 0.0;
            long numTokens = 0;
            for (Map.Entry<String, String> e : text.entrySet()) {
                String curText = e.getValue();
                List<Token> tokens = textTokens.get(e.getKey());
                numTokens += tokens.size();

                // Sample a new Example.
                int[] activeFeatures = new int[tokens.size()];
                char[] sampledText = curText.toCharArray();
                for (int j = 0; j < activeFeatures.length; j++) {
                    activeFeatures[j] = innerRNG.nextInt(2);
                    if (activeFeatures[j] == 0) {
                        textDistance++;
                        Token curToken = tokens.get(j);
                        Arrays.fill(sampledText, curToken.start, curToken.end, '\0');
                    }
                }
                String sampledString = new String(sampledText);
                sampledString = sampledString.replace("\0", "");

                textFeatures.addAll(textFields.get(e.getKey()).process(sampledString));

                for (int j = 0; j < activeFeatures.length; j++) {
                    perturbedFeatures.add(new Feature(nameFeature(e.getKey(), tokens.get(j).text, j), activeFeatures[j]));
                }
            }
            // Add the text features to the current example
            sampledExample.addAll(textFeatures);
            // Calculate text distance
            double totalTextDistance = textDistance / numTokens;

            // Label it using the full model.
            Prediction<Label> samplePrediction = innerModel.predict(sampledExample);

            double totalLength = tabularFeatures.size() + perturbedFeatures.size();
            // Combine the distances and transform into a weight
            // Currently this averages the two values based on their relative sizes.
            double weight = 1.0 - ((tabularFeatures.size()*(kernelDist(tabularDistance,kernelWidth) + perturbedFeatures.size()*totalTextDistance) / totalLength));

            // Generate the new sample with the appropriate label and weight.
            ArrayExample<Regressor> labelledSample = new ArrayExample<>(transformOutput(samplePrediction), (float) weight);
            labelledSample.addAll(tabularFeatures);
            labelledSample.addAll(perturbedFeatures);
            output.add(labelledSample);
        }

        return output;
    }
}
