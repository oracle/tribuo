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

package org.tribuo.regression.rtree;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.Trainer;
import org.tribuo.common.tree.ExtraTreesTrainer;
import org.tribuo.common.tree.RandomForestTrainer;
import org.tribuo.common.tree.TreeEnsembleModel;
import org.tribuo.common.tree.TreeModel;
import org.tribuo.ensemble.EnsembleModel;
import org.tribuo.math.la.SparseVector;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.ensemble.AveragingCombiner;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.tribuo.regression.rtree.impurity.MeanSquaredError;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.tribuo.common.tree.AbstractCARTTrainer.MIN_EXAMPLES;

/**
 * Tests for the TreeEnsembleModel sparse vector reuse optimization.
 */
public class TreeEnsembleOptimizationTest {

	/**
	 * Verifies that RandomForestTrainer creates TreeEnsembleModel.
	 */
	@Test
	public void testRandomForestCreatesTreeEnsembleModel() {
		Pair<Dataset<Regressor>, Dataset<Regressor>> data =
			RegressionDataGenerator.denseTrainTest();

		CARTRegressionTrainer treeTrainer = new CARTRegressionTrainer(
			Integer.MAX_VALUE, MIN_EXAMPLES, 0.0f, 0.5f, false,
			new MeanSquaredError(), Trainer.DEFAULT_SEED);

		RandomForestTrainer<Regressor> trainer = new RandomForestTrainer<>(
			treeTrainer, new AveragingCombiner(), 10);

		EnsembleModel<Regressor> model = trainer.train(data.getA());

		assertTrue(model instanceof TreeEnsembleModel,
			"RandomForestTrainer should create TreeEnsembleModel for optimization");
	}

	/**
	 * Verifies that ExtraTreesTrainer creates TreeEnsembleModel.
	 */
	@Test
	public void testExtraTreesCreatesTreeEnsembleModel() {
		Pair<Dataset<Regressor>, Dataset<Regressor>> data =
			RegressionDataGenerator.denseTrainTest();

		CARTRegressionTrainer treeTrainer = new CARTRegressionTrainer(
			Integer.MAX_VALUE, MIN_EXAMPLES, 0.0f, 0.5f, true,
			new MeanSquaredError(), Trainer.DEFAULT_SEED);

		ExtraTreesTrainer<Regressor> trainer = new ExtraTreesTrainer<>(
			treeTrainer, new AveragingCombiner(), 10);

		EnsembleModel<Regressor> model = trainer.train(data.getA());

		assertTrue(model instanceof TreeEnsembleModel,
			"ExtraTreesTrainer should create TreeEnsembleModel for optimization");
	}

	/**
	 * Verifies that TreeModel.predict(SGDVector, Example) produces
	 * identical results to TreeModel.predict(Example).
	 */
	@Test
	public void testTreeModelSparseVectorOverloadEquivalence() {
		Pair<Dataset<Regressor>, Dataset<Regressor>> data =
			RegressionDataGenerator.denseTrainTest();

		CARTRegressionTrainer trainer = new CARTRegressionTrainer();
		IndependentRegressionTreeModel model =
			(IndependentRegressionTreeModel) trainer.train(data.getA());

		// Test on multiple examples
		for (Example<Regressor> testExample : data.getB()) {
			// Prediction using standard method
			Prediction<Regressor> standardPrediction = model.predict(testExample);

			// Prediction using sparse vector overload
			SparseVector vec = SparseVector.createSparseVector(
				testExample, model.getFeatureIDMap(), false);
			Prediction<Regressor> optimizedPrediction = model.predict(vec, testExample);

			// Should produce identical results
			assertEquals(standardPrediction.getOutput().size(),
				optimizedPrediction.getOutput().size(),
				"Output dimensions should match");

			double[] standardValues = standardPrediction.getOutput().getValues();
			double[] optimizedValues = optimizedPrediction.getOutput().getValues();

			for (int i = 0; i < standardValues.length; i++) {
				assertEquals(standardValues[i], optimizedValues[i], 0.0001,
					"Predictions should be identical for dimension " + i);
			}
		}
	}

	/**
	 * Verifies that TreeEnsembleModel produces correct predictions
	 * by comparing to manual averaging of individual tree predictions.
	 */
	@Test
	public void testTreeEnsembleModelCorrectness() {
		Pair<Dataset<Regressor>, Dataset<Regressor>> data =
			RegressionDataGenerator.denseTrainTest();

		CARTRegressionTrainer treeTrainer = new CARTRegressionTrainer(
			Integer.MAX_VALUE, MIN_EXAMPLES, 0.0f, 0.5f, false,
			new MeanSquaredError(), Trainer.DEFAULT_SEED);

		RandomForestTrainer<Regressor> trainer = new RandomForestTrainer<>(
			treeTrainer, new AveragingCombiner(), 10);

		TreeEnsembleModel<Regressor> model =
			(TreeEnsembleModel<Regressor>) trainer.train(data.getA());

		// Test on multiple examples
		Example<Regressor> testExample = data.getB().iterator().next();

		// Prediction through TreeEnsembleModel (uses optimization)
		Prediction<Regressor> ensemblePrediction = model.predict(testExample);

		// Manual prediction: create sparse vector and call each tree
		SparseVector vec = SparseVector.createSparseVector(
			testExample, model.getFeatureIDMap(), false);

		double manualSum = 0.0;
		for (Model<Regressor> tree : model.getModels()) {
			IndependentRegressionTreeModel treeModel =
				(IndependentRegressionTreeModel) tree;
			Prediction<Regressor> treePred = treeModel.predict(vec, testExample);
			manualSum += treePred.getOutput().getValues()[0];
		}
		double manualAverage = manualSum / model.getModels().size();

		// Should match ensemble prediction (averaging combiner)
		assertEquals(manualAverage,
			ensemblePrediction.getOutput().getValues()[0], 0.0001,
			"TreeEnsembleModel should produce same result as manual averaging");
	}

}
