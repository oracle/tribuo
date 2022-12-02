/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.classification.sequence.viterbi;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.protos.ViterbiModelProto;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.protos.core.SequenceModelProto;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.sequence.SequenceDataset;
import org.tribuo.sequence.SequenceExample;
import org.tribuo.sequence.SequenceModel;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * An implementation of a viterbi model.
 */
public class ViterbiModel extends SequenceModel<Label> {

    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * Types of label score aggregation.
     */
    public enum ScoreAggregation {
        /**
         * Adds the scores.
         */
        ADD,
        /**
         * Multiplies the scores.
         */
        MULTIPLY
    }

    private final Model<Label> model;

    private final LabelFeatureExtractor labelFeatureExtractor;

    /**
     * Specifies the maximum number of candidate paths to keep track of. In general, this number
     * should be higher than the number of possible classifications at any given point in the
     * sequence. This guarantees that highest-possible scoring sequence will be returned. If,
     * however, the number of possible classifications is quite high and/or you are concerned about
     * throughput performance, then you may want to reduce the number of candidate paths to
     * maintain.
     */
    private final int stackSize;

    /**
     * Specifies the score aggregation algorithm.
     */
    private final ScoreAggregation scoreAggregation;

    ViterbiModel(String name, ModelProvenance description,
                        Model<Label> model, LabelFeatureExtractor labelFeatureExtractor, int stackSize, ScoreAggregation scoreAggregation) {
        super(name, description, model.getFeatureIDMap(), model.getOutputIDInfo());
        this.model = model;
        this.labelFeatureExtractor = labelFeatureExtractor;
        this.stackSize = stackSize;
        this.scoreAggregation = scoreAggregation;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static ViterbiModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        ViterbiModelProto proto = message.unpack(ViterbiModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(Label.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a label domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<Label> outputDomain = (ImmutableOutputInfo<Label>) carrier.outputDomain();

        Model<?> model = Model.deserialize(proto.getModel());
        if (!model.validate(Label.class)) {
            throw new IllegalStateException("Invalid protobuf, expected a classification model, found " + model);
        }
        Model<Label> labelModel = model.castModel(Label.class);

        LabelFeatureExtractor labelFeatureExtractor = LabelFeatureExtractor.deserialize(proto.getLabelFeatureExtractor());

        int stackSize = proto.getStackSize();

        ScoreAggregation scoreAggregation = ScoreAggregation.valueOf(proto.getScoreAggregation());

        return new ViterbiModel(carrier.name(),carrier.provenance(),labelModel,labelFeatureExtractor,stackSize,scoreAggregation);
    }

    @Override
    public List<List<Prediction<Label>>> predict(SequenceDataset<Label> examples) {
        List<List<Prediction<Label>>> predictions = new ArrayList<>();
        for (SequenceExample<Label> e : examples) {
            predictions.add(predict(e));
        }
        return predictions;
    }

    @Override
    public List<Prediction<Label>> predict(SequenceExample<Label> examples) {
        if (stackSize == 1) {
            List<Label> labels = new ArrayList<>();
            List<Prediction<Label>> returnValues = new ArrayList<>();
            for (Example<Label> example : examples) {
                List<Feature> labelFeatures = extractFeatures(labels);
                example.addAll(labelFeatures);
                Prediction<Label> prediction = model.predict(example);
                labels.add(prediction.getOutput());
                returnValues.add(prediction);
            }
            return returnValues;
        } else {
            return viterbi(examples);
        }
    }

    /**
     * Returns the inner model used for sequence element predictions.
     * @return The inner model.
     */
    public Model<Label> getInnerModel() {
        return model;
    }

    private List<Feature> extractFeatures(List<Label> labels) {
        List<Feature> labelFeatures = new ArrayList<>();
        for (Feature labelFeature : labelFeatureExtractor.extractFeatures(labels, 1.0)) {
            int id = featureIDMap.getID(labelFeature.getName());
            if (id > -1) {
                labelFeatures.add(labelFeature);
            }
        }
        return labelFeatures;
    }

    /**
     * This implementation of Viterbi requires at most stackSize * sequenceLength calls to the
     * classifier. If this proves to be too expensive, then consider using a smaller stack size.
     *
     * @param examples a sequence-worth of features. Each {@code List<Feature>} in features should correspond to
     *                 all of the features for a given element in a sequence to be classified.
     * @return a list of Predictions - one for each member of the sequence.
     * @see LabelFeatureExtractor
     */
    private List<Prediction<Label>> viterbi(SequenceExample<Label> examples) {
        // find the best paths through the label lattice
        Collection<Path> paths = null;
        int[] numUsed = new int[examples.size()];
        int i = 0;
        for (Example<Label> example : examples) {
            // if this is the first instance, start new paths for each label
            if (paths == null) {
                paths = new ArrayList<>();
                Prediction<Label> prediction = this.model.predict(example);
                numUsed[i] = prediction.getNumActiveFeatures();
                Map<String, Label> distribution = prediction.getOutputScores();
                for (Label label : this.getTopLabels(distribution)) {
                    paths.add(new Path(label, label.getScore(), null));
                }
            } else {
                // for later instances, find the best previous path for each label
                Map<Label, Path> maxPaths = new HashMap<>();
                for (Path path : paths) {
                    Example<Label> clonedExample = example.copy();
                    List<Label> previousLabels = new ArrayList<>(path.labels);
                    List<Feature> labelFeatures = extractFeatures(previousLabels);
                    clonedExample.addAll(labelFeatures);
                    Prediction<Label> prediction = this.model.predict(clonedExample);
                    // TODO this isn't quite correct as it includes label features.
                    numUsed[i] = prediction.getNumActiveFeatures();
                    Map<String, Label> distribution = prediction.getOutputScores();

                    for (Label label : this.getTopLabels(distribution)) {
                        double labelScore = label.getScore();
                        double score = this.scoreAggregation == ScoreAggregation.ADD ? path.score + labelScore : path.score * labelScore;
                        Path maxPath = maxPaths.get(label);
                        if (maxPath == null || score > maxPath.score) {
                            maxPaths.put(label, new Path(label, score, path));
                        }
                    }
                }
                paths = maxPaths.values();
            }
            i++;
        }

        Path maxPath = Collections.max(paths);

        ArrayList<Prediction<Label>> output = new ArrayList<>();

        for (int j = 0; j < examples.size(); j++) {
            Example<Label> e = examples.get(j);
            output.add(new Prediction<>(maxPath.labels.get(j), numUsed[j], e));
        }

        return output;
    }

    /**
     * Gets the most likely labels.
     * @param distribution The label distribution.
     * @return The most likely labels in descending order.
     */
    protected List<Label> getTopLabels(Map<String, Label> distribution) {
        return getTopLabels(distribution, this.stackSize);
    }

    /**
     * Gets the {@code stackSize} most likely labels.
     * @param distribution The label distribution.
     * @param stackSize The number of labels to pick.
     * @return The most likely labels in descending order.
     */
    protected static List<Label> getTopLabels(Map<String, Label> distribution, int stackSize) {
        return distribution.values().stream().sorted(Comparator.comparingDouble(Label::getScore).reversed()).limit(stackSize)
                .collect(Collectors.toList());
        // get just the labels that fit within the stack
    }

    private static class Path implements Comparable<Path> {

        public final double score;

        public final Path parent;

        public final List<Label> labels;

        public Path(Label label, double score, Path parent) {
            this.score = score;
            this.parent = parent;
            this.labels = new ArrayList<>();
            if (this.parent != null) {
                this.labels.addAll(this.parent.labels);
            }
            this.labels.add(label);
        }

        @Override
        public int compareTo(Path that) {
            return Double.compare(this.score, that.score);
        }

    }

    /**
     * Gets the stack size of this model.
     * @return The stack size.
     */
    public int getStackSize() {
        return stackSize;
    }

    /**
     * Gets the score aggregation function.
     * @return The score aggregation function.
     */
    public ScoreAggregation getScoreAggregation() {
        return scoreAggregation;
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return model.getTopFeatures(n);
    }

    @Override
    public SequenceModelProto serialize() {
        ModelDataCarrier<Label> carrier = createDataCarrier();

        ViterbiModelProto.Builder modelBuilder = ViterbiModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setModel(model.serialize());
        modelBuilder.setLabelFeatureExtractor(labelFeatureExtractor.serialize());
        modelBuilder.setStackSize(stackSize);
        modelBuilder.setScoreAggregation(scoreAggregation.name());

        SequenceModelProto.Builder builder = SequenceModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(ViterbiModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }
}
