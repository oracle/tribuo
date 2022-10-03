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

package org.tribuo.classification.mnb;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.mnb.protos.MultinomialNaiveBayesProto;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.math.la.DenseSparseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.math.util.ExpNormalizer;
import org.tribuo.math.util.VectorNormalizer;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.ModelProvenance;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * A {@link Model} for multinomial Naive Bayes with Laplace smoothing.
 * <p>
 * All feature values must be non-negative, otherwise it will throw IllegalArgumentException.
 * <p>
 * See:
 * <pre>
 * Wang S, Manning CD.
 * "Baselines and Bigrams: Simple, Good Sentiment and Topic Classification"
 * Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics, 2012.
 * </pre>
 */
public class MultinomialNaiveBayesModel extends Model<Label> {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final DenseSparseMatrix labelWordProbs;
    private final double alpha;

    private static final VectorNormalizer normalizer = new ExpNormalizer();

    MultinomialNaiveBayesModel(String name, ModelProvenance description, ImmutableFeatureMap featureInfos, ImmutableOutputInfo<Label> labelInfos, DenseSparseMatrix labelWordProbs, double alpha) {
        super(name, description, featureInfos, labelInfos, true);
        this.labelWordProbs = labelWordProbs;
        this.alpha = alpha;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static MultinomialNaiveBayesModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        MultinomialNaiveBayesProto proto = message.unpack(MultinomialNaiveBayesProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(Label.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a label domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<Label> outputDomain = (ImmutableOutputInfo<Label>) carrier.outputDomain();

        Tensor weights = Tensor.deserialize(proto.getLabelWordProbs());
        if (!(weights instanceof DenseSparseMatrix)) {
            throw new IllegalStateException("Invalid protobuf, label word probs must be a sparse matrix, found " + weights.getClass());
        }
        DenseSparseMatrix labelWordProbs = (DenseSparseMatrix) weights;
        if (labelWordProbs.getDimension1Size() != carrier.outputDomain().size()) {
            throw new IllegalStateException("Invalid protobuf, labelWordProbs not the right size, expected " + carrier.outputDomain().size() + ", found " + labelWordProbs.getDimension1Size());
        }
        if (labelWordProbs.getDimension2Size() != carrier.featureDomain().size()) {
            throw new IllegalStateException("Invalid protobuf, labelWordProbs not the right size, expected " + carrier.featureDomain().size() + ", found " + labelWordProbs.getDimension2Size());
        }

        double alpha = proto.getAlpha();

        if (alpha < 0.0) {
            throw new IllegalStateException("Invalid protobuf, alpha must be non-negative, found " + alpha);
        }

        return new MultinomialNaiveBayesModel(carrier.name(),carrier.provenance(),carrier.featureDomain(),outputDomain,labelWordProbs,alpha);
    }

    @Override
    public Prediction<Label> predict(Example<Label> example) {
        SparseVector exVector = SparseVector.createSparseVector(example, featureIDMap, false);

        if (exVector.minValue() < 0.0) {
            throw new IllegalArgumentException("Example has negative feature values, example = " + example.toString());
        }
        if (exVector.numActiveElements() == 0) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }

        /* Since we keep the label by feature matrix sparse, we need to manually
         * add the weights contributed by smoothing unobserved features. We need to
         * add in the portion of the inner product for the indices that are active
         * in the example but are not active in the labelWordProbs matrix (but are
         * still non-zero due to smoothing).
         */
        double[] alphaOffsets = new double[outputIDInfo.size()];
        int vocabSize = labelWordProbs.getDimension2Size();
        if (alpha > 0.0) {
            for (int i = 0; i < outputIDInfo.size(); i++) {
                double unobservedProb = Math.log(alpha / (labelWordProbs.getRow(i).oneNorm() + (vocabSize * alpha)));
                int[] mismatchedIndices = exVector.difference(labelWordProbs.getRow(i));
                double inExampleFactor = 0.0;
                for (int idx = 0; idx < mismatchedIndices.length; idx++) {
                    // TODO - exVector.get is slow as it does a binary search into the vector.
                    inExampleFactor += exVector.get(mismatchedIndices[idx]) * unobservedProb;
                }
                alphaOffsets[i] = inExampleFactor;
            }
        }

        DenseVector prediction = labelWordProbs.leftMultiply(exVector);
        prediction.intersectAndAddInPlace(DenseVector.createDenseVector(alphaOffsets));
        prediction.normalize(normalizer);
        Map<String,Label> distribution = new LinkedHashMap<>();
        Label maxLabel = null;
        double maxScore = Double.NEGATIVE_INFINITY;
        for(VectorTuple vt : prediction) {
            String name = outputIDInfo.getOutput(vt.index).getLabel();
            Label label = new Label(name, vt.value);
            if (vt.value > maxScore) {
                maxScore = vt.value;
                maxLabel = label;
            }
            distribution.put(name,label);
        }
        Prediction<Label> p = new Prediction<>(maxLabel, distribution, exVector.numActiveElements(), example, true);
        return p;
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        int maxFeatures = n < 0 ? featureIDMap.size() : n;
        Map<String, List<Pair<String, Double>>> topFeatures = new HashMap<>();

        for (Pair<Integer,Label> label : outputIDInfo) {
            List<Pair<String, Double>> features = new ArrayList<>(labelWordProbs.numActiveElements(label.getA()));
            for(VectorTuple vt : labelWordProbs.getRow(label.getA())) {
                features.add(new Pair<>(featureIDMap.get(vt.index).getName(), vt.value));
            }
            features.sort(Comparator.comparing(x -> -x.getB()));
            if(maxFeatures < featureIDMap.size()) {
                features = features.subList(0, maxFeatures);
            }
            topFeatures.put(label.getB().getLabel(), features);
        }
        return topFeatures;
    }

    @Override
    public Optional<Excuse<Label>> getExcuse(Example<Label> example) {
        Map<String, List<Pair<String, Double>>> explanation = new HashMap<>();
        for (Pair<Integer,Label> label : outputIDInfo) {
            List<Pair<String, Double>> scores = new ArrayList<>();
            for(Feature f : example) {
                int id = featureIDMap.getID(f.getName());
                if (id > -1) {
                    scores.add(new Pair<>(f.getName(),labelWordProbs.getRow(label.getA()).get(id)));
                }
            }
            explanation.put(label.getB().getLabel(), scores);
        }
        return Optional.of(new Excuse<>(example, predict(example), explanation));
    }

    @Override
    protected MultinomialNaiveBayesModel copy(String newName, ModelProvenance newProvenance) {
        return new MultinomialNaiveBayesModel(newName,newProvenance,featureIDMap,outputIDInfo,new DenseSparseMatrix(labelWordProbs),alpha);
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<Label> carrier = createDataCarrier();

        MultinomialNaiveBayesProto.Builder modelBuilder = MultinomialNaiveBayesProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setLabelWordProbs(labelWordProbs.serialize());
        modelBuilder.setAlpha(alpha);

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(MultinomialNaiveBayesModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }
}
