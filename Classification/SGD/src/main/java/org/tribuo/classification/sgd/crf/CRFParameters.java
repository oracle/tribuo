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

package org.tribuo.classification.sgd.crf;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.classification.sgd.protos.CRFParametersProto;
import org.tribuo.math.Parameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseSparseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.protos.ParametersProto;
import org.tribuo.math.util.HeapMerger;
import org.tribuo.math.util.Merger;
import org.tribuo.protos.ProtoUtil;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A {@link Parameters} for training a CRF using SGD.
 */
public class CRFParameters implements Parameters, Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final int numLabels;
    private final int numFeatures;

    private static final Merger merger = new HeapMerger();

    /**
     * This variable is an array with 3 elements corresponding to the three weight matrices.
     */
    private Tensor[] weights;

    private DenseVector biases; //weights[0];
    private DenseMatrix featureLabelWeights; //weights[1];
    private DenseMatrix labelLabelWeights; //weights[2];

    CRFParameters(int numFeatures, int numLabels) {
        this.biases = new DenseVector(numLabels);
        this.featureLabelWeights = new DenseMatrix(numLabels,numFeatures);
        this.labelLabelWeights = new DenseMatrix(numLabels,numLabels);
        this.weights = new Tensor[3];
        weights[0] = biases;
        weights[1] = featureLabelWeights;
        weights[2] = labelLabelWeights;
        this.numLabels = numLabels;
        this.numFeatures = numFeatures;
    }

    private CRFParameters(DenseVector biases, DenseMatrix featureLabelWeights, DenseMatrix labelLabelWeights) {
        this.weights = new Tensor[3];
        weights[0] = biases;
        weights[1] = featureLabelWeights;
        weights[2] = labelLabelWeights;
        this.numLabels = biases.size();
        this.numFeatures = featureLabelWeights.getDimension2Size();
        this.biases = biases;
        this.featureLabelWeights = featureLabelWeights;
        this.labelLabelWeights = labelLabelWeights;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static CRFParameters deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        CRFParametersProto proto = message.unpack(CRFParametersProto.class);
        int numLabels = proto.getNumLabels();
        int numFeatures = proto.getNumFeatures();
        Tensor biasTensor = ProtoUtil.deserialize(proto.getBiases());
        Tensor featureLabelTensor = ProtoUtil.deserialize(proto.getFeatureLabelWeights());
        Tensor labelLabelTensor = ProtoUtil.deserialize(proto.getLabelLabelWeights());
        if (!(biasTensor instanceof DenseVector)) {
            throw new IllegalArgumentException("Invalid protobuf, expected bias vector, found " + biasTensor.getClass().getSimpleName());
        } else if (((DenseVector)biasTensor).size() != numLabels) {
            throw new IllegalArgumentException("Invalid protobuf, expected bias vector with " + numLabels + " elements, but found " + ((DenseVector)biasTensor).size());
        }

        if (!(featureLabelTensor instanceof DenseMatrix)) {
            throw new IllegalArgumentException("Invalid protobuf, expected feature/label matrix, found " + featureLabelTensor.getClass().getSimpleName());
        }
        DenseMatrix featureLabelMatrix = (DenseMatrix) featureLabelTensor;
        if ((featureLabelMatrix.getDimension1Size() != numLabels) || (featureLabelMatrix.getDimension2Size() != numFeatures)) {
            throw new IllegalArgumentException("Invalid protobuf, expected feature/label matrix of size [" + numLabels + ", " + numFeatures + "], found " + Arrays.toString(featureLabelMatrix.getShape()));
        }

        if (!(labelLabelTensor instanceof DenseMatrix)) {
            throw new IllegalArgumentException("Invalid protobuf, expected label/label matrix, found " + labelLabelTensor.getClass().getSimpleName());
        }
        DenseMatrix labelLabelMatrix = (DenseMatrix) labelLabelTensor;
        if ((labelLabelMatrix.getDimension1Size() != numLabels) || (labelLabelMatrix.getDimension2Size() != numLabels)) {
            throw new IllegalArgumentException("Invalid protobuf, expected label/label matrix of size [" + numLabels + ", " + numLabels + "], found " + Arrays.toString(labelLabelMatrix.getShape()));
        }

        return new CRFParameters((DenseVector) biasTensor, featureLabelMatrix, labelLabelMatrix);
    }

    @Override
    public ParametersProto serialize() {
        ParametersProto.Builder builder = ParametersProto.newBuilder();

        builder.setVersion(CURRENT_VERSION);
        builder.setClassName(CRFParameters.class.getName());
        CRFParametersProto.Builder crfParamsBuilder = CRFParametersProto.newBuilder();
        crfParamsBuilder.setNumFeatures(numFeatures);
        crfParamsBuilder.setNumLabels(numLabels);
        crfParamsBuilder.setBiases(biases.serialize());
        crfParamsBuilder.setFeatureLabelWeights(featureLabelWeights.serialize());
        crfParamsBuilder.setLabelLabelWeights(labelLabelWeights.serialize());
        builder.setSerializedData(Any.pack(crfParamsBuilder.build()));

        return builder.build();
    }

    /**
     * Gets a copy of the weights for the specified label id.
     * @param id The label id.
     * @return The feature weights.
     */
    public DenseVector getFeatureWeights(int id) {
        return featureLabelWeights.getColumn(id);
    }

    /**
     * Returns the bias for the specified label id.
     * @param id The label id.
     * @return The bias.
     */
    public double getBias(int id) {
        return biases.get(id);
    }

    /**
     * Returns the feature/label weight for the specified feature and label id.
     * @param labelID The label id.
     * @param featureID The feature id.
     * @return The feature/label weight.
     */
    public double getWeight(int labelID, int featureID) {
        return featureLabelWeights.get(labelID, featureID);
    }

    /**
     * Generate the local scores (i.e., the linear classifier for each token).
     * @param features An array of {@link SGDVector}s, one per token.
     * @return An array of DenseVectors representing the scores per label for each token.
     */
    public DenseVector[] getLocalScores(SGDVector[] features) {
        DenseVector[] localScores = new DenseVector[features.length];
        for (int i = 0; i < features.length; i++) {
            DenseVector scores = featureLabelWeights.leftMultiply(features[i]);
            scores.intersectAndAddInPlace(biases);
            localScores[i] = scores;
        }
        return localScores;
    }

    /**
     * Generates the local scores and tuples them with the label - label transition weights.
     * @param features The per token {@link SGDVector} of features.
     * @return A tuple containing the array of {@link DenseVector} scores and the label - label transition weights.
     */
    public ChainHelper.ChainCliqueValues getCliqueValues(SGDVector[] features) {
        DenseVector[] localScores = getLocalScores(features);
        return new ChainHelper.ChainCliqueValues(localScores, labelLabelWeights);
    }

    /**
     * Generate a prediction using Viterbi.
     * @param features The per token {@link SGDVector} of features.
     * @return An int array giving the predicted label per token.
     */
    public int[] predict(SGDVector[] features) {
        ChainHelper.ChainViterbiResults result = ChainHelper.viterbi(getCliqueValues(features));
        return result.mapValues;
    }

    /**
     * Generate a prediction using Belief Propagation.
     * @param features The per token {@link SGDVector} of features.
     * @return A {@link DenseVector} per token containing the marginal distribution over labels.
     */
    public DenseVector[] predictMarginals(SGDVector[] features) {
        ChainHelper.ChainBPResults result = ChainHelper.beliefPropagation(getCliqueValues(features));
        DenseVector[] marginals = new DenseVector[features.length];
        for (int i = 0; i < features.length; i++) {
            marginals[i] = result.alphas[i].add(result.betas[i]);
            marginals[i].expNormalize(result.logZ);
        }
        return marginals;
    }

    /**
     * This predicts per chunk confidence using the constrained forward backward algorithm from
     * Culotta and McCallum 2004.
     * <p>
     * Runs one pass of BP to get the normalizing constant, and then a further chunks.size() passes
     * of constrained forward backward.
     * @param features The per token {@link SGDVector} of features.
     * @param chunks A list of extracted chunks to pin constrain the labels to.
     * @return A list containing the confidence value for each chunk.
     */
    public List<Double> predictConfidenceUsingCBP(SGDVector[] features, List<Chunk> chunks) {
        ChainHelper.ChainCliqueValues cliqueValues = getCliqueValues(features);
        ChainHelper.ChainBPResults bpResult = ChainHelper.beliefPropagation(cliqueValues);
        double bpLogZ = bpResult.logZ;

        int[] constraints = new int[features.length];

        List<Double> output = new ArrayList<>();
        for (Chunk chunk : chunks) {
            Arrays.fill(constraints,-1);
            chunk.unpack(constraints);
            double chunkScore = ChainHelper.constrainedBeliefPropagation(cliqueValues,constraints);
            output.add(Math.exp(chunkScore - bpLogZ));
        }

        return output;
    }

    /**
     * Generates predictions based on the input features and labels, then scores those predictions to
     * produce a loss for the example and a gradient update.
     * <p>
     * Assumes all the features in this example are either SparseVector or DenseVector.
     * Mixing the two will cause undefined behaviour.
     * @param features The per token {@link SGDVector} of features.
     * @param labels The per token ground truth labels.
     * @return A {@link Pair} containing the loss for this example and the associated gradient.
     */
    public Pair<Double, Tensor[]> valueAndGradient(SGDVector[] features, int[] labels) {
        ChainHelper.ChainCliqueValues scores = getCliqueValues(features);
        // Infer the marginal distribution over labels for each token.
        ChainHelper.ChainBPResults bpResults = ChainHelper.beliefPropagation(scores);
        double logZ = bpResults.logZ;
        DenseVector[] alphas = bpResults.alphas;
        DenseVector[] betas = bpResults.betas;

        //Calculate the gradients for the parameters.
        Tensor[] gradient = new Tensor[3];
        DenseSparseMatrix[] featureGradients = new DenseSparseMatrix[features.length];
        DenseMatrix denseFeatureGradients = null;
        boolean sparseFeatures = false;
        gradient[0] = new DenseVector(biases.size());
        Matrix transGradient = new DenseMatrix(numLabels, numLabels);
        gradient[2] = transGradient;
        double score = -logZ;
        for (int i = 0; i < features.length; i++) {
            int curLabel = labels[i];
            // Increment the loss based on the score for the true label.
            DenseVector curLocalScores = scores.localValues[i];
            score += curLocalScores.get(curLabel);
            // Generate the predicted local marginal from the BP run.
            DenseVector curAlpha = alphas[i];
            DenseVector curBeta = betas[i];
            DenseVector localMarginal = curAlpha.add(curBeta);
            // Generate the gradient for the biases based on the true label and predicted label.
            localMarginal.expNormalize(logZ);
            localMarginal.scaleInPlace(-1.0);
            localMarginal.add(curLabel,1.0);
            gradient[0].intersectAndAddInPlace(localMarginal);
            // Generate the gradient for the feature - label weights
            Matrix tmpFeatureGradient = localMarginal.outer(features[i]);
            if (tmpFeatureGradient instanceof DenseSparseMatrix) {
                featureGradients[i] = (DenseSparseMatrix) tmpFeatureGradient;
                sparseFeatures = true;
            } else {
                if (denseFeatureGradients == null) {
                    denseFeatureGradients = (DenseMatrix) tmpFeatureGradient;
                } else {
                    denseFeatureGradients.intersectAndAddInPlace(tmpFeatureGradient);
                }
            }
            // If the sequence has more than one token generate the gradient for the label - label transitions.
            if (i >= 1) {
                DenseVector prevAlpha = alphas[i - 1];
                for (int ii = 0; ii < numLabels; ii++) {
                    double prevAlphaVal = prevAlpha.get(ii);
                    for (int jj = 0; jj < numLabels; jj++) {
                        double update = -Math.exp(prevAlphaVal + labelLabelWeights.get(ii,jj) + curBeta.get(jj) + curLocalScores.get(jj) - logZ);
                        transGradient.add(ii, jj, update);
                    }
                }
                int prevLabel = labels[i-1];
                // Increment the loss based on the transition from the previous predicted label to the true label.
                score += labelLabelWeights.get(prevLabel,curLabel);
                transGradient.add(prevLabel, curLabel, 1.0);
            }
        }

        if (sparseFeatures) {
            // Merge together all the sparse feature - label gradients.
            gradient[1] = merger.merge(featureGradients);
            // throw if we found any dense features as well as the sparse.
            if (denseFeatureGradients != null) {
                throw new IllegalStateException("Mixture of dense and sparse features found.");
            }
        } else {
            gradient[1] = denseFeatureGradients;
        }

        return new Pair<>(score,gradient);
    }

    /**
     * Returns a 3 element {@link Tensor} array.
     *
     * The first element is a {@link DenseVector} of label biases.
     * The second element is a {@link DenseMatrix} of feature-label weights.
     * The third element is a {@link DenseMatrix} of label-label transition weights.
     * @return A {@link Tensor} array.
     */
    @Override
    public Tensor[] getEmptyCopy() {
        Tensor[] output = new Tensor[3];
        output[0] = new DenseVector(biases.size());
        output[1] = new DenseMatrix(featureLabelWeights.getDimension1Size(),featureLabelWeights.getDimension2Size());
        output[2] = new DenseMatrix(labelLabelWeights.getDimension1Size(),labelLabelWeights.getDimension2Size());
        return output;
    }

    @Override
    public Tensor[] get() {
        return weights;
    }

    @Override
    public void set(Tensor[] newWeights) {
        if (newWeights.length == weights.length) {
            weights = newWeights;
            biases = (DenseVector) weights[0];
            featureLabelWeights = (DenseMatrix) weights[1];
            labelLabelWeights = (DenseMatrix) weights[2];
        }
    }

    @Override
    public void update(Tensor[] gradients) {
        for (int i = 0; i < gradients.length; i++) {
            weights[i].intersectAndAddInPlace(gradients[i]);
        }
    }

    @Override
    public Tensor[] merge(Tensor[][] gradients, int size) {
        DenseVector biasUpdate = new DenseVector(biases.size());
        List<DenseSparseMatrix> updates = new ArrayList<>(size);
        DenseMatrix denseUpdates = null;
        DenseMatrix labelLabelUpdate = new DenseMatrix(labelLabelWeights.getDimension1Size(),labelLabelWeights.getDimension2Size());
        for (int j = 0; j < gradients.length; j++) {
            biasUpdate.intersectAndAddInPlace(gradients[j][0]);
            Matrix tmpUpdate = (Matrix) gradients[j][1];
            if (tmpUpdate instanceof DenseSparseMatrix) {
                updates.add((DenseSparseMatrix)tmpUpdate);
            } else {
                // is dense
                if (denseUpdates == null) {
                    denseUpdates = (DenseMatrix) tmpUpdate;
                } else {
                    denseUpdates.intersectAndAddInPlace(tmpUpdate);
                }
            }
            labelLabelUpdate.intersectAndAddInPlace(gradients[j][2]);
        }

        // Merge the combination of any dense and sparse updates
        Matrix featureLabelUpdate;
        if (updates.size() > 0) {
            featureLabelUpdate = merger.merge(updates.toArray(new DenseSparseMatrix[0]));
            if (denseUpdates != null) {
                denseUpdates.intersectAndAddInPlace(featureLabelUpdate);
                featureLabelUpdate = denseUpdates;
            }
        } else {
            featureLabelUpdate = denseUpdates;
        }

        return new Tensor[]{biasUpdate,featureLabelUpdate,labelLabelUpdate};
    }
}
