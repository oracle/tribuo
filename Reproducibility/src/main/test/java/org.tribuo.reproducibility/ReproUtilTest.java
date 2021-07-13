package org.tribuo.reproducibility;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Trainer;
import org.tribuo.common.sgd.AbstractLinearSGDModel;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.tribuo.regression.sgd.linear.LinearSGDTrainer;
import org.tribuo.regression.sgd.objectives.SquaredLoss;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

class ReproUtilTest {


}