package org.tribuo.interop.tensorflow;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.framework.optimizers.AdaDelta;
import org.tensorflow.framework.optimizers.AdaGrad;
import org.tensorflow.framework.optimizers.AdaGradDA;
import org.tensorflow.framework.optimizers.Adam;
import org.tensorflow.framework.optimizers.Adamax;
import org.tensorflow.framework.optimizers.Ftrl;
import org.tensorflow.framework.optimizers.GradientDescent;
import org.tensorflow.framework.optimizers.Momentum;
import org.tensorflow.framework.optimizers.Nadam;
import org.tensorflow.framework.optimizers.Optimizer;
import org.tensorflow.framework.optimizers.RMSProp;
import org.tensorflow.op.Op;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.family.TNumber;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * An enum for the gradient optimizers exposed by TensorFlow-Java.
 */
public enum GradientOptimiser {

    /**
     * The AdaDelta optimizer.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the overall learning rate.</li>
     *     <li>rho - the decay factor.</li>
     *     <li>epsilon - for numerical stability.</li>
     * </ul>
     */
    ADADELTA("learningRate","rho","epsilon"),
    /**
     * The AdaGrad optimizer.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the overall learning rate.</li>
     *     <li>initialAccumulatorValue - the initialisation value for the gradient accumulator.</li>
     * </ul>
     */
    ADAGRAD("learningRate","initialAccumulatorValue"),
    /**
     * The AdaGrad Dual Averaging optimizer.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the overall learning rate.</li>
     *     <li>initialAccumulatorValue - the initialisation value for the gradient accumulator.</li>
     *     <li>l1Strength - the strength of l1 regularisation.</li>
     *     <li>l2Strength - the strength of l2 regularisation.</li>
     * </ul>
     */
    ADAGRADDA("learningRate","initialAccumulatorValue","l1Strength","l2Strength"),
    /**
     * The Adam optimizer.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the learning rate.</li>
     *     <li>betaOne - the exponential decay rate for the 1st moment estimates.</li>
     *     <li>betaTwo - the exponential decay rate for the exponentially weighted infinity norm.</li>
     *     <li>epsilon - a small constant for numerical stability.</li>
     * </ul>
     */
    ADAM("learningRate","betaOne","betaTwo","epsilon"),
    /**
     * The Adamax optimizer.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the learning rate.</li>
     *     <li>betaOne - the exponential decay rate for the 1st moment estimates.</li>
     *     <li>betaTwo - the exponential decay rate for the exponentially weighted infinity norm.</li>
     *     <li>epsilon - a small constant for numerical stability.</li>
     * </ul>
     */
    ADAMAX("learningRate","betaOne","betaTwo","epsilon"),
    /**
     * The FTRL optimizer.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the learning rate.</li>
     *     <li>learningRatePower - controls how the learning rate decreases during training. Use zero for
     *     a fixed learning rate.</li>
     *     <li>initialAccumulatorValue - the starting value for accumulators. Only zero or positive
     *     values are allowed.</li>
     *     <li>l1Strength - the L1 Regularization strength, must be greater than or equal to zero.</li>
     *     <li>l2Strength - the L2 Regularization strength, must be greater than or equal to zero.</li>
     *     <li>l2ShrinkageRegularizationStrength - this differs from L2 above in that the L2 above is a
     *     stabilization penalty, whereas this L2 shrinkage is a magnitude penalty. must be greater
     *     than or equal to zero.</li>
     * </ul>
     */
    FTRL("learningRate","learningRatePower","initialAccumulatorValue","l1Strength","l2Strength","l2ShrinkageRegularizationStrength"),
    /**
     * A standard gradient descent optimizer with a fixed learning rate.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the learning rate.</li>
     * </ul>
     */
    GRADIENT_DESCENT("learningRate"),
    /**
     * Gradient descent with momentum.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the learning rate.</li>
     *     <li>momentum - the momentum scalar.</li>
     * </ul>
     */
    MOMENTUM("learningRate","momentum"),
    /**
     * Gradient descent with Nesterov momentum.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the learning rate.</li>
     *     <li>momentum - the momentum scalar.</li>
     * </ul>
     */
    NESTEROV("learningRate","momentum"),
    /**
     * The Nadam optimizer.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the learning rate.</li>
     *     <li>betaOne - the exponential decay rate for the 1st moment estimates.</li>
     *     <li>betaTwo - the exponential decay rate for the exponentially weighted infinity norm.</li>
     *     <li>epsilon - a small constant for numerical stability.</li>
     * </ul>
     */
    NADAM("learningRate","betaOne","betaTwo","epsilon"),
    /**
     * The AdaDelta optimizer.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the overall learning rate.</li>
     *     <li>decay - the decay factor.</li>
     *     <li>momentum - the momentum scalar.</li>
     *     <li>epsilon - for numerical stability.</li>
     * </ul>
     * This optimizer is currently uncentered.
     */
    RMSPROP("learningRate","decay","momentum","epsilon");

    private final Set<String> args;

    /**
     * Construct the enum, storing the optimiser argument names.
     * @param args The optimiser argument names.
     */
    private GradientOptimiser(String... args) {
        this.args = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(args)));
    }

    /**
     * An unmodifiable view of the parameter names used by this gradient optimiser.
     * @return The parameter names.
     */
    public Set<String> getParameterNames() {
        return args;
    }

    /**
     * Checks that the parameter names in the supplied set are an exact
     * match for the parameter names that this gradient optimizer expects.
     * @param paramNames The gradient optimizer parameter names.
     * @return True if the two sets intersection and union are equal.
     */
    public boolean validateParamNames(Set<String> paramNames) {
        return (args.size() == paramNames.size()) && args.containsAll(paramNames);
    }

    /**
     * Applies the optimizer to the graph and returns the optimiser step operation.
     * @param graph The graph to optimise.
     * @param loss The loss to minimise.
     * @param optimiserParams The optimiser parameters.
     * @param <T> The loss type (most of the time this will be {@link TFloat32}.
     * @return The optimiser step operation.
     */
    public <T extends TNumber> Op applyOptimizer(Graph graph, Operand<T> loss, Map<String,Float> optimiserParams) {
        if (!validateParamNames(optimiserParams.keySet())) {
            throw new IllegalArgumentException("Invalid optimiser parameters, expected " + args.toString() + ", found " + optimiserParams.keySet().toString());
        }
        Optimizer optimizer;
        switch (this) {
            case ADADELTA:
                optimizer = new AdaDelta(graph,"tribuo-adadelta",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("rho"),
                        optimiserParams.get("epsilon"));
                break;
            case ADAGRAD:
                optimizer = new AdaGrad(graph,"tribuo-adagrad",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("initialAccumulatorValue"));
                break;
            case ADAGRADDA:
                optimizer = new AdaGradDA(graph,"tribuo-adagradda",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("initialAccumulatorValue"),
                        optimiserParams.get("l1Strength"),
                        optimiserParams.get("l2Strength"));
                break;
            case ADAM:
                optimizer = new Adam(graph,"tribuo-adam",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("betaOne"),
                        optimiserParams.get("betaTwo"),
                        optimiserParams.get("epsilon"));
                break;
            case ADAMAX:
                optimizer = new Adamax(graph,"tribuo-adamax",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("betaOne"),
                        optimiserParams.get("betaTwo"),
                        optimiserParams.get("epsilon"));
                break;
            case FTRL:
                optimizer = new Ftrl(graph,"tribuo-ftrl",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("learningRatePower"),
                        optimiserParams.get("initialAccumulatorValue"),
                        optimiserParams.get("l1Strength"),
                        optimiserParams.get("l2Strength"),
                        optimiserParams.get("l2ShrinkageRegularizationStrength"));
                break;
            case GRADIENT_DESCENT:
                optimizer = new GradientDescent(graph,"tribuo-sgd",
                        optimiserParams.get("learningRate"));
                break;
            case MOMENTUM:
                optimizer = new Momentum(graph,"tribuo-momentum",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("momentum"),
                        false);
                break;
            case NESTEROV:
                optimizer = new Momentum(graph,"tribuo-nesterov",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("momentum"),
                        true);
                break;
            case NADAM:
                optimizer = new Nadam(graph,"tribuo-nadam",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("betaOne"),
                        optimiserParams.get("betaTwo"),
                        optimiserParams.get("epsilon"));
                break;
            case RMSPROP:
                optimizer = new RMSProp(graph,"tribuo-rmsprop",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("decay"),
                        optimiserParams.get("momentum"),
                        optimiserParams.get("epsilon"),
                        false);
                break;
            default:
                throw new IllegalStateException("Unimplemented switch branch " + this.toString());
        }
        return optimizer.minimize(loss,"tribuo-" + this.toString() + "-minimize");
    }
}
