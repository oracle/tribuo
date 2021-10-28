/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.classification.ensemble;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableDataset;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.Trainer;
import org.tribuo.WeightedExamples;
import org.tribuo.classification.Label;
import org.tribuo.dataset.DatasetView;
import org.tribuo.ensemble.WeightedEnsembleModel;
import org.tribuo.provenance.EnsembleModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.util.Util;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Implements Adaboost.SAMME one of the more popular algorithms for multiclass boosting.
 * Based on  <a href="https://web.stanford.edu/~hastie/Papers/samme.pdf">this paper</a>.
 * <p>
 * If the trainer implements {@link WeightedExamples} then it performs boosting by weighting,
 * otherwise it uses a weighted bootstrap sample.
 * <p>
 * See:
 * <pre>
 * J. Zhu, S. Rosset, H. Zou, T. Hastie.
 * "Multi-class Adaboost"
 * 2006.
 * </pre>
 */
public class AdaBoostTrainer implements Trainer<Label> {

    private static final Logger logger = Logger.getLogger(AdaBoostTrainer.class.getName());
    
    @Config(mandatory=true, description="The trainer to use to build each weak learner.")
    protected Trainer<Label> innerTrainer;

    @Config(mandatory=true, description="The number of ensemble members to train.")
    protected int numMembers;

    @Config(mandatory=true, description="The seed for the RNG.")
    protected long seed;

    protected SplittableRandom rng;

    protected int trainInvocationCounter;

    /**
     * For the OLCUT configuration system.
     */
    private AdaBoostTrainer() { }

    /**
     * Constructs an adaboost trainer using the supplied weak learner trainer and the specified number of
     * boosting rounds. Uses the default seed.
     * @param trainer The weak learner trainer.
     * @param numMembers The maximum number of boosting rounds.
     */
    public AdaBoostTrainer(Trainer<Label> trainer, int numMembers) {
        this(trainer, numMembers, Trainer.DEFAULT_SEED);
    }

    /**
     * Constructs an adaboost trainer using the supplied weak learner trainer, the specified number of
     * boosting rounds and the supplied seed.
     * @param trainer The weak learner trainer.
     * @param numMembers The maximum number of boosting rounds.
     * @param seed The RNG seed.
     */
    public AdaBoostTrainer(Trainer<Label> trainer, int numMembers, long seed) {
        this.innerTrainer = trainer;
        this.numMembers = numMembers;
        this.seed = seed;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public synchronized void postConfig() {
        this.rng = new SplittableRandom(seed);
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder();

        buffer.append("AdaBoostTrainer(");
        buffer.append("innerTrainer=");
        buffer.append(innerTrainer.toString());
        buffer.append(",numMembers=");
        buffer.append(numMembers);
        buffer.append(",seed=");
        buffer.append(seed);
        buffer.append(")");

        return buffer.toString();
    }

    /**
     * If the trainer implements {@link WeightedExamples} then do boosting by weighting,
     * otherwise do boosting by sampling.
     * @param examples the data set containing the examples.
     * @return A {@link WeightedEnsembleModel}.
     */
    @Override
    public Model<Label> train(Dataset<Label> examples, Map<String, Provenance> runProvenance) {
        return(train(examples, runProvenance, INCREMENT_INVOCATION_COUNT));
    }

    @Override
    public Model<Label> train(Dataset<Label> examples, Map<String, Provenance> runProvenance, int invocationCount) {
        if (examples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }
        // Creates a new RNG, adds one to the invocation count.
        SplittableRandom localRNG;
        TrainerProvenance trainerProvenance;
        synchronized(this) {
            if(invocationCount != INCREMENT_INVOCATION_COUNT) {
                setInvocationCount(invocationCount);
            }
            localRNG = rng.split();
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }
        boolean weighted = innerTrainer instanceof WeightedExamples;
        ImmutableFeatureMap featureIDs = examples.getFeatureIDMap();
        ImmutableOutputInfo<Label> labelIDs = examples.getOutputIDInfo();
        int numClasses = labelIDs.size();
        logger.log(Level.INFO,"NumClasses = " + numClasses);
        ArrayList<Model<Label>> models = new ArrayList<>();
        float[] modelWeights = new float[numMembers];
        float[] exampleWeights = Util.generateUniformFloatVector(examples.size(), 1.0f/examples.size());
        if (weighted) {
            logger.info("Using weighted Adaboost.");
            examples = ImmutableDataset.copyDataset(examples);
            for (int i = 0; i < examples.size(); i++) {
                Example<Label> e = examples.getExample(i);
                e.setWeight(exampleWeights[i]);
            }
        } else {
            logger.info("Using sampling Adaboost.");
        }
        for (int i = 0; i < numMembers; i++) {
            logger.info("Building model " + i);
            Model<Label> newModel;
            if (weighted) {
                newModel = innerTrainer.train(examples);
            } else {
                DatasetView<Label> bag = DatasetView.createWeightedBootstrapView(examples,examples.size(),localRNG.nextLong(),exampleWeights,featureIDs,labelIDs);
                newModel = innerTrainer.train(bag);
            }

            //
            // Score this model
            List<Prediction<Label>> predictions = newModel.predict(examples);
            float accuracy = accuracy(predictions,examples,exampleWeights);
            float error = 1.0f - accuracy;
            float alpha = (float) (Math.log(accuracy/error) + Math.log(numClasses - 1));
            models.add(newModel);
            modelWeights[i] = alpha;
            if ((accuracy + 1e-10) > 1.0) {
                //
                // Perfect accuracy, can no longer boost.
                float[] newModelWeights = Arrays.copyOf(modelWeights, models.size());
                newModelWeights[models.size()-1] = 1.0f; //Set the last weight to 1, as it's infinity.
                logger.log(Level.FINE, "Perfect accuracy reached on iteration " + i + ", returning current model.");
                logger.log(Level.FINE, "Model weights:");
                Util.logVector(logger, Level.FINE, newModelWeights);
                EnsembleModelProvenance provenance = new EnsembleModelProvenance(WeightedEnsembleModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance, ListProvenance.createListProvenance(models));
                return new WeightedEnsembleModel<>("boosted-ensemble",provenance,featureIDs,labelIDs,models,new VotingCombiner(),newModelWeights);
            }

            //
            // Update the weights
            for (int j = 0; j < predictions.size(); j++) {
                if (!predictions.get(j).getOutput().equals(examples.getExample(j).getOutput())) {
                    exampleWeights[j] *= Math.exp(alpha);
                }
            }
            Util.inplaceNormalizeToDistribution(exampleWeights);
            if (weighted) {
                for (int j = 0; j < examples.size(); j++) {
                    examples.getExample(j).setWeight(exampleWeights[j]);
                }
            }
        }
        logger.log(Level.FINE, "Model weights:");
        Util.logVector(logger, Level.FINE, modelWeights);
        EnsembleModelProvenance provenance = new EnsembleModelProvenance(WeightedEnsembleModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance, ListProvenance.createListProvenance(models));
        return new WeightedEnsembleModel<>("boosted-ensemble",provenance,featureIDs,labelIDs,models,new VotingCombiner(),modelWeights);
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public synchronized void setInvocationCount(int invocationCount){
        if(invocationCount < 0){
            throw new IllegalArgumentException("The supplied invocationCount is less than zero.");
        }

        rng = new SplittableRandom(seed);

        for (trainInvocationCounter = 0; trainInvocationCounter < invocationCount; trainInvocationCounter++){
            SplittableRandom localRNG = rng.split();
        }

    }

    /**
     * Compute the accuracy of a set of predictions.
     * @param predictions The base learner predictions.
     * @param examples The training examples.
     * @param weights The current example weights.
     * @return The accuracy.
     */
    private static float accuracy(List<Prediction<Label>> predictions, Dataset<Label> examples, float[] weights) {
        float correctSum = 0;
        float total = 0;
        for (int i = 0; i < predictions.size(); i++) {
            if (predictions.get(i).getOutput().equals(examples.getExample(i).getOutput())) {
                correctSum += weights[i];
            }
            total += weights[i];
        }

        logger.log(Level.FINEST, "Correct count = " + correctSum + " size = " + examples.size());

        return correctSum / total;
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}
