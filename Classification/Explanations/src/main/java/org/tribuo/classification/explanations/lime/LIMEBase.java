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
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.OutputFactory;
import org.tribuo.Prediction;
import org.tribuo.RealInfo;
import org.tribuo.SparseModel;
import org.tribuo.SparseTrainer;
import org.tribuo.VariableIDInfo;
import org.tribuo.VariableInfo;
import org.tribuo.WeightedExamples;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.explanations.TabularExplainer;
import org.tribuo.impl.ArrayExample;
import org.tribuo.interop.ExternalModel;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.provenance.SimpleDataSourceProvenance;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.util.Util;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.SplittableRandom;
import java.util.logging.Logger;

/**
 * LIMEBase merges the lime_base.py and lime_tabular.py implementations, and deals with simple
 * matrices of numerical or categorical data. If you want a mixture of text, numerical
 * and categorical data try {@link LIMEColumnar}. For plain text data use {@link LIMEText}.
 * <p>
 * See:
 * <pre>
 * Ribeiro MT, Singh S, Guestrin C.
 * "Why should I trust you?: Explaining the predictions of any classifier"
 * Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining 2016.
 * </pre>
 */
public class LIMEBase implements TabularExplainer<Regressor> {
    private static final Logger logger = Logger.getLogger(LIMEBase.class.getName());

    /**
     * Width of the noise gaussian.
     */
    public static final double WIDTH_CONSTANT = 0.75;
    /**
     * Delta to consider two distances equal.
     */
    public static final double DISTANCE_DELTA = 1e-12;

    protected static final OutputFactory<Regressor> regressionFactory = new RegressionFactory();
    protected static final RegressionEvaluator evaluator = new RegressionEvaluator(true);

    protected final SplittableRandom rng;

    protected final Model<Label> innerModel;

    protected final SparseTrainer<Regressor> explanationTrainer;

    protected final int numSamples;

    protected final long numTrainingExamples;

    protected final double kernelWidth;

    private final ImmutableFeatureMap fMap;

    /**
     * Constructs a LIME explainer for a model which uses tabular data (i.e., no special treatment for text features).
     * @param rng The rng to use for sampling.
     * @param innerModel The model to explain.
     * @param explanationTrainer The sparse trainer used to explain predictions.
     * @param numSamples The number of samples to generate for an explanation.
     */
    public LIMEBase(SplittableRandom rng, Model<Label> innerModel, SparseTrainer<Regressor> explanationTrainer, int numSamples) {
        if (!(explanationTrainer instanceof WeightedExamples)) {
            throw new IllegalArgumentException("SparseTrainer must implement WeightedExamples, found " + explanationTrainer.toString());
        }
        if (!innerModel.generatesProbabilities()) {
            throw new IllegalArgumentException("LIME requires the model generate probabilities.");
        }
        if (innerModel instanceof ExternalModel) {
            throw new IllegalArgumentException("LIME requires the model to have been trained in Tribuo. Found " + innerModel.getClass() + " which is an external model.");
        }
        this.rng = rng;
        this.innerModel = innerModel;
        this.explanationTrainer = explanationTrainer;
        this.numSamples = numSamples;
        this.numTrainingExamples = innerModel.getOutputIDInfo().getTotalObservations();
        this.kernelWidth = Math.pow(innerModel.getFeatureIDMap().size() * WIDTH_CONSTANT, 2);
        this.fMap = innerModel.getFeatureIDMap();
    }

    @Override
    public LIMEExplanation explain(Example<Label> example) {
        return explainWithSamples(example).getA();
    }

    protected Pair<LIMEExplanation,List<Example<Regressor>>> explainWithSamples(Example<Label> example) {
        // Predict using the full model, and generate a new example containing that prediction.
        Prediction<Label> prediction = innerModel.predict(example);
        Example<Regressor> labelledExample = new ArrayExample<>(transformOutput(prediction),example,1.0f);

        // Sample a dataset.
        List<Example<Regressor>> sample = sampleData(example);

        // Generate a sparse model on the sampled data.
        SparseModel<Regressor> model = trainExplainer(labelledExample,sample);

        // Test the sparse model against the predictions of the real model.
        List<Prediction<Regressor>> predictions = new ArrayList<>(model.predict(sample));
        predictions.add(model.predict(labelledExample));
        RegressionEvaluation evaluation = evaluator.evaluate(model,predictions,new SimpleDataSourceProvenance("LIMEColumnar sampled data",regressionFactory));

        return new Pair<>(new LIMEExplanation(model,prediction,evaluation),sample);
    }

    /**
     * Sample a dataset based on the input example.
     * <p>
     * The sampled dataset uses the feature dimensions from the {@link Model}.
     * <p>
     * The outputs are the probability values of each class from the underlying Model,
     * rather than ground truth outputs. The distance is measured using the
     * {@link LIMEBase#measureDistance} function, transformed through a kernel and used
     * as the sampled Example's weight.
     * @param example The example to sample from.
     * @return A sampled dataset.
     */
    private List<Example<Regressor>> sampleData(Example<Label> example) {
        List<Example<Regressor>> output = new ArrayList<>();

        SparseVector exampleVector = SparseVector.createSparseVector(example,fMap,false);

        Random innerRNG = new Random(rng.nextLong());
        for (int i = 0; i < numSamples; i++) {
            // Sample a new Example.
            Example<Label> sample = samplePoint(innerRNG,fMap,numTrainingExamples,exampleVector);

            //logger.fine("Itr " + i + " sampled " + sample.toString());

            // Label it using the full model.
            Prediction<Label> samplePrediction = innerModel.predict(sample);

            // Measure the distance between this point and the input, to be used as a weight.
            double distance = measureDistance(fMap,numTrainingExamples,exampleVector, SparseVector.createSparseVector(sample,fMap,false));

            // Transform distance through the kernel function.
            distance = kernelDist(distance,kernelWidth);

            // Generate the new sample with the appropriate label and weight.
            Example<Regressor> labelledSample = new ArrayExample<>(transformOutput(samplePrediction),sample,(float)distance);
            output.add(labelledSample);
        }

        return output;
    }

    /**
     * Samples a single example from the supplied feature map and input vector.
     * @param rng The rng to use.
     * @param fMap The feature map describing the domain of the features.
     * @param numTrainingExamples The number of training examples the fMap has seen.
     * @param input The input sparse vector to use.
     * @return An Example sampled from the supplied feature map and input vector.
     */
    public static Example<Label> samplePoint(Random rng, ImmutableFeatureMap fMap, long numTrainingExamples, SparseVector input) {
        ArrayList<String> names = new ArrayList<>();
        ArrayList<Double> values = new ArrayList<>();

        for (VariableInfo info : fMap) {
            int id = ((VariableIDInfo)info).getID();
            double inputValue = input.get(id);

            if (info instanceof CategoricalInfo) {
                // This one is tricksy as categorical info essentially implicitly includes a zero.
                CategoricalInfo catInfo = (CategoricalInfo) info;
                double sample = catInfo.frequencyBasedSample(rng,numTrainingExamples);
                // If we didn't sample zero.
                if (Math.abs(sample) > 1e-10) {
                    names.add(info.getName());
                    values.add(sample);
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
                if (rng.nextDouble() < threshold) {
                    double variance = realInfo.getVariance();
                    double sample = (rng.nextGaussian() * Math.sqrt(variance)) + inputValue;
                    names.add(info.getName());
                    values.add(sample);
                }
            } else {
                throw new IllegalStateException("Unsupported info type, expected CategoricalInfo or RealInfo, found " + info.getClass().getName());
            }
        }

        return new ArrayExample<>(LabelFactory.UNKNOWN_LABEL,names.toArray(new String[0]),Util.toPrimitiveDouble(values));
    }

    /**
     * Trains the explanation model using the supplied sampled data and the input example.
     * <p>
     * The labels are usually the predicted probabilities from the real model.
     * @param target The input example to explain.
     * @param samples The sampled data around the input.
     * @return An explanation model.
     */
    protected SparseModel<Regressor> trainExplainer(Example<Regressor> target, List<Example<Regressor>> samples) {
        MutableDataset<Regressor> explanationDataset = new MutableDataset<>(new SimpleDataSourceProvenance("explanationDataset", OffsetDateTime.now(), regressionFactory), regressionFactory);
        explanationDataset.add(target);
        explanationDataset.addAll(samples);

        SparseModel<Regressor> explainer = explanationTrainer.train(explanationDataset);

        return explainer;
    }

    /**
     * Calculates an RBF kernel of a specific width.
     * @param input The input value.
     * @param width The width of the kernel.
     * @return sqrt ( exp ( - input*input / width))
     */
    public static double kernelDist(double input, double width) {
        return Math.sqrt(Math.exp(-(input*input) / width));
    }

    /**
     * Measures the distance between an input point and a sampled point.
     * <p>
     * This distance function takes into account categorical and real values. It uses
     * the hamming distance for categoricals and the euclidean distance for real values.
     * @param fMap The feature map used to determine if a feature is categorical or real.
     * @param numTrainingExamples The number of training examples the fMap has seen.
     * @param input The input point.
     * @param sample The sampled point.
     * @return The distance between the two points.
     */
    public static double measureDistance(ImmutableFeatureMap fMap, long numTrainingExamples, SparseVector input, SparseVector sample) {
        double score = 0.0;

        Iterator<VectorTuple> itr = input.iterator();
        Iterator<VectorTuple> otherItr = sample.iterator();
        VectorTuple tuple;
        VectorTuple otherTuple;
        while (itr.hasNext() && otherItr.hasNext()) {
            tuple = itr.next();
            otherTuple = otherItr.next();
            //after this loop, either itr is out or tuple.index >= otherTuple.index
            while (itr.hasNext() && (tuple.index < otherTuple.index)) {
                score += calculateSingleDistance(fMap,numTrainingExamples,tuple.index,tuple.value);
                tuple = itr.next();
            }
            //after this loop, either otherItr is out or tuple.index <= otherTuple.index
            while (otherItr.hasNext() && (tuple.index > otherTuple.index)) {
                score += calculateSingleDistance(fMap,numTrainingExamples,otherTuple.index,otherTuple.value);
                otherTuple = otherItr.next();
            }
            if (tuple.index == otherTuple.index) {
                //the indices line up, do the calculation.
                score += calculateSingleDistance(fMap,numTrainingExamples,tuple.index,tuple.value,otherTuple.value);
            } else {
                // Now consume both the values as they'll be gone next iteration.
                // Consume the value in tuple.
                score += calculateSingleDistance(fMap,numTrainingExamples,tuple.index,tuple.value);
                // Consume the value in otherTuple.
                score += calculateSingleDistance(fMap,numTrainingExamples,otherTuple.index,otherTuple.value);
            }
        }
        while (itr.hasNext()) {
            tuple = itr.next();
            score += calculateSingleDistance(fMap,numTrainingExamples,tuple.index,tuple.value);
        }
        while (otherItr.hasNext()) {
            otherTuple = otherItr.next();
            score += calculateSingleDistance(fMap,numTrainingExamples,otherTuple.index,otherTuple.value);
        }

        return Math.sqrt(score);
    }

    /**
     * Calculates the distance between two values for a single feature.
     * <p>
     * Assumes the other value is zero as the example is sparse.
     * @param fMap The feature map which knows if a feature is categorical or real.
     * @param numTrainingExamples The number of training examples this feature map observed.
     * @param index The id number for this feature.
     * @param value One feature value.
     * @return The distance from zero to the supplied value.
     */
    private static double calculateSingleDistance(ImmutableFeatureMap fMap, long numTrainingExamples, int index, double value) {
        VariableInfo info = fMap.get(index);
        if (info instanceof CategoricalInfo) {
            return 1.0;
        } else if (info instanceof RealInfo) {
            RealInfo rInfo = (RealInfo) info;
            // Fudge the distance calculation so it doesn't overpower the categoricals.
            double curScore = value * value;
            double range;
            // This further fudge is because the RealInfo may have observed a zero if it's sparse, but it might not.
            if (numTrainingExamples != info.getCount()) {
                range = Math.max(rInfo.getMax(),0.0) - Math.min(rInfo.getMin(),0.0);
            } else {
                range = rInfo.getMax() - rInfo.getMin();
            }
            return curScore / (range*range);
        } else {
            throw new IllegalStateException("Unsupported info type, expected CategoricalInfo or RealInfo, found " + info.getClass().getName());
        }
    }

    /**
     * Calculates the distance between two values for a single feature.
     *
     * @param fMap The feature map which knows if a feature is categorical or real.
     * @param numTrainingExamples The number of training examples this feature map observed.
     * @param index The id number for this feature.
     * @param firstValue The first feature value.
     * @param secondValue The second feature value.
     * @return The distance between the two values.
     */
    private static double calculateSingleDistance(ImmutableFeatureMap fMap, long numTrainingExamples, int index, double firstValue, double secondValue) {
        VariableInfo info = fMap.get(index);
        if (info instanceof CategoricalInfo) {
            if (Math.abs(firstValue - secondValue) > DISTANCE_DELTA) {
                return 1.0;
            } else {
                // else the values are the same so the hamming distance is zero.
                return 0.0;
            }
        } else if (info instanceof RealInfo) {
            RealInfo rInfo = (RealInfo) info;
            // Fudge the distance calculation so it doesn't overpower the categoricals.
            double tmp = firstValue - secondValue;
            double range;
            // This further fudge is because the RealInfo may have observed a zero if it's sparse, but it might not.
            if (numTrainingExamples != info.getCount()) {
                range = Math.max(rInfo.getMax(),0.0) - Math.min(rInfo.getMin(),0.0);
            } else {
                range = rInfo.getMax() - rInfo.getMin();
            }
            return (tmp*tmp) / (range*range);
        } else {
            throw new IllegalStateException("Unsupported info type, expected CategoricalInfo or RealInfo, found " + info.getClass().getName());
        }
    }

    /**
     * Transforms a {@link Prediction} for a multiclass problem into a {@link Regressor}
     * output which represents the probability for each class.
     * <p>
     * Used as the target for LIME Models.
     * @param prediction A multiclass prediction object. Must contain probabilities.
     * @return The n dimensional probability output.
     */
    public static Regressor transformOutput(Prediction<Label> prediction) {
        Map<String,Label> outputs = prediction.getOutputScores();

        String[] names = new String[outputs.size()];
        double[] values = new double[outputs.size()];

        int i = 0;
        for (Map.Entry<String,Label> e : outputs.entrySet()) {
            names[i] = e.getKey();
            values[i] = e.getValue().getScore();
            i++;
        }

        return new Regressor(names,values);
    }

}
