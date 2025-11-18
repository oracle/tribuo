/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.common.tree;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.ensemble.EnsembleCombiner;
import org.tribuo.ensemble.WeightedEnsembleModel;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.EnsembleModelProvenance;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.protos.core.WeightedEnsembleModelProto;
import org.tribuo.util.Util;

import java.util.ArrayList;
import java.util.List;

/**
 * Optimized ensemble model for tree-based models.
 * <p>
 * This subclass of {@link WeightedEnsembleModel} provides optimized prediction performance
 * for ensembles containing {@link TreeModel}s. Non-tree models in the ensemble fall back
 * to standard prediction.
 * </p>
 */
public final class TreeEnsembleModel<T extends Output<T>> extends WeightedEnsembleModel<T> {

	/**
	 * Protobuf serialization version.
	 */
	public static final int CURRENT_VERSION = 0;

	/**
	 * Constructs a tree ensemble model with uniform weights.
	 * @param name The model name.
	 * @param provenance The model provenance.
	 * @param featureIDMap The feature domain.
	 * @param outputIDInfo The output domain.
	 * @param newModels The list of ensemble members.
	 * @param combiner The combination function.
	 */
	public TreeEnsembleModel(String name, EnsembleModelProvenance provenance, ImmutableFeatureMap featureIDMap,
	                         ImmutableOutputInfo<T> outputIDInfo, List<Model<T>> newModels, EnsembleCombiner<T> combiner) {
		super(name, provenance, featureIDMap, outputIDInfo, newModels, combiner);
	}

	/**
	 * Constructs a tree ensemble model with specified weights.
	 * @param name The model name.
	 * @param provenance The model provenance.
	 * @param featureIDMap The feature domain.
	 * @param outputIDInfo The output domain.
	 * @param newModels The list of ensemble members.
	 * @param combiner The combination function.
	 * @param weights The model combination weights.
	 */
	public TreeEnsembleModel(String name, EnsembleModelProvenance provenance, ImmutableFeatureMap featureIDMap,
	                         ImmutableOutputInfo<T> outputIDInfo, List<Model<T>> newModels,
	                         EnsembleCombiner<T> combiner, float[] weights) {
		super(name, provenance, featureIDMap, outputIDInfo, newModels, combiner, weights);
	}

	/**
	 * Deserialization factory.
	 * <p>
	 * Delegates to the parent class's deserialization logic for validation and parsing,
	 * then reconstructs as a TreeEnsembleModel.
	 * </p>
	 * @param version The serialized object version.
	 * @param className The class name.
	 * @param message The serialized data.
	 * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
	 * @return The deserialized object.
	 */
	@SuppressWarnings({"unchecked","rawtypes"}) // Guarded by getClass checks to ensure all outputs are the same type.
	public static TreeEnsembleModel<?> deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
		if (version < 0 || version > CURRENT_VERSION) {
			throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
		}
		WeightedEnsembleModelProto proto = message.unpack(WeightedEnsembleModelProto.class);
		// Delegate to parent for validation and parsing
		WeightedEnsembleModel<?> parent = WeightedEnsembleModel.deserializeFromProto(version, className, message);
		// Extract weights and combiner from proto (since they're protected in parent)
		float[] weights = Util.toPrimitiveFloat(proto.getWeightsList());
		EnsembleCombiner<?> combiner = EnsembleCombiner.deserialize(proto.getCombiner());
		return new TreeEnsembleModel(parent.getName(), parent.getProvenance(),
			parent.getFeatureIDMap(), parent.getOutputIDInfo(), parent.getModels(),
			combiner, weights);
	}

	@Override
	protected TreeEnsembleModel<T> copy(String name, EnsembleModelProvenance newProvenance, List<Model<T>> newModels) {
		return new TreeEnsembleModel<>(name, newProvenance, featureIDMap, outputIDInfo, newModels, combiner);
	}

	@Override
	public Prediction<T> predict(Example<T> example) {
		List<Prediction<T>> predictions = new ArrayList<>();

		// Optimization: Create sparse vector once and reuse across all trees.
		// TreeModel has a predict(SparseVector, Example) overload for this purpose.
		// Non-tree models fall back to standard prediction.
		SparseVector vec = SparseVector.createSparseVector(
			example, featureIDMap, false);

		for (Model<T> model : models) {
			if (model instanceof TreeModel) {
				TreeModel<T> treeModel = (TreeModel<T>) model;
				predictions.add(treeModel.predict(vec, example));
			} else {
				predictions.add(model.predict(example));
			}
		}

		return combiner.combine(outputIDInfo, predictions, weights);
	}

	@Override
	public ModelProto serialize() {
		// Call parent's serialization, then update the class name
		ModelProto parentProto = super.serialize();
		return ModelProto.newBuilder(parentProto)
			.setClassName(TreeEnsembleModel.class.getName())
			.setVersion(CURRENT_VERSION)
			.build();
	}
}

