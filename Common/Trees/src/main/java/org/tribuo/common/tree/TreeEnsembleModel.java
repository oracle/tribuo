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
}

