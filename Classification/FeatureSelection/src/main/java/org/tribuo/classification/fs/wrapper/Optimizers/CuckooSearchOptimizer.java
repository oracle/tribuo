package org.tribuo.classification.fs.wrapper.Optimizers;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.*;
import org.tribuo.classification.Label;
import org.tribuo.classification.ensemble.VotingCombiner;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.fs.wrapper.Discreeting.TransferFunction;
import org.tribuo.common.nearest.KNNModel;
import org.tribuo.common.nearest.KNNTrainer;
import org.tribuo.dataset.SelectedFeatureDataset;
import org.tribuo.evaluation.CrossValidation;
import org.tribuo.math.distance.L1Distance;
import org.tribuo.math.neighbour.NeighboursQueryFactoryType;
import org.tribuo.provenance.FeatureSelectorProvenance;
import org.tribuo.provenance.FeatureSetProvenance;
import org.tribuo.provenance.impl.FeatureSelectorProvenanceImpl;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * Select features based on Cuckoo Search algorithm with binary transfer functions, KNN classifier and 10-fold cross validation
 * <p>
 * see:
 * <pre>
 * Xin-She Yang and Suash Deb.
 * "Cuckoo Search via L´evy Flights", 2010.
 *
 * L. A. M. Pereira et al.
 * "A Binary Cuckoo Search and its Application for Feature Selection", 2014.
 * </pre>
 */
public class CuckooSearchOptimizer implements FeatureSelector<Label> {
    private final TransferFunction transferFunction;
    private final double stepSizeScaling;
    private final double lambda;
    private final double worstNestProbability;
    private final double delta;
    private final int populationSize;
    private int [][] setOfSolutions;
    private final int maxIteration;

    /**
     * The default constructor for feature selection based on Cuckoo Search Algorithm
     */
    public CuckooSearchOptimizer() {
        this.transferFunction = TransferFunction.V2;
        this.populationSize = 50;
        this.stepSizeScaling = 2d;
        this.lambda = 2d;
        this.worstNestProbability = 0.1d;
        this.delta = 1.5d;
        this.maxIteration = 30;
    }

    /**
     * Constructs the wrapper feature selection based on cuckoo search algorithm
     * @param transferFunction The transfer function to convert continuous values to binary ones
     * @param populationSize The size of the solution in the initial population
     * @param maxIteration The number of times that is used to enhance generation
     */
    public CuckooSearchOptimizer(TransferFunction transferFunction, int populationSize, int maxIteration) {
        this.transferFunction = transferFunction;
        this.populationSize = populationSize;
        this.stepSizeScaling = 2d;
        this.lambda = 2d;
        this.worstNestProbability = 1.5d;
        this.delta = 1.5d;
        this.maxIteration = maxIteration;
    }

    /**
     * Constructs the wrapper feature selection based on cuckoo search algorithm
     * @param transferFunction The transfer function to convert continuous values to binary ones
     * @param populationSize The size of the solution in the initial population
     * @param stepSizeScaling The cuckoo step size
     * @param lambda The lambda of the levy flight function
     * @param worstNestProbability The fraction of the nests to be abandoned
     * @param delta The delta that is used in the abandon nest function
     * @param maxIteration The number of times that is used to enhance generation
     */
    public CuckooSearchOptimizer(TransferFunction transferFunction, int populationSize, double stepSizeScaling, double lambda, double worstNestProbability, double delta, int maxIteration) {
        this.transferFunction = transferFunction;
        this.populationSize = populationSize;
        this.stepSizeScaling = stepSizeScaling;
        this.lambda = lambda;
        this.worstNestProbability = worstNestProbability;
        this.delta = delta;
        this.maxIteration = maxIteration;
    }

    /**
     * Constructs the wrapper feature selection based on cuckoo search algorithm
     * @param dataPath The path of the dataset
     * @param transferFunction The transfer function to convert continuous values to binary ones
     * @param populationSize The size of the solution in the initial population
     * @param stepSizeScaling The cuckoo step size
     * @param lambda The lambda of the levy flight function
     * @param worstNestProbability The fraction of the nests to be abandoned
     * @param delta The delta that is used in the abandon nest function
     * @param maxIteration The number of times that is used to enhance generation
     */
    public CuckooSearchOptimizer(String dataPath, TransferFunction transferFunction, int populationSize, double stepSizeScaling, double lambda, double worstNestProbability, double delta, int maxIteration) {
        this.transferFunction = transferFunction;
        this.populationSize = populationSize;
        this.stepSizeScaling = stepSizeScaling;
        this.lambda = lambda;
        this.worstNestProbability = worstNestProbability;
        this.delta = delta;
        this.maxIteration = maxIteration;
    }

    /**
     * This method is used to generate the initial population (set of solutions)
     * @param totalNumberOfFeatures The number of features in the given dataset
     * @return The population of subsets of selected features
     */
    private int[][] GeneratePopulation(int totalNumberOfFeatures) {
        SplittableRandom sRand = new SplittableRandom(12345);
        setOfSolutions = new int[this.populationSize][totalNumberOfFeatures];
        for (int[] subSet : setOfSolutions) {
            int[] values = new int[subSet.length];
            for (int i = 0; i < values.length; i++) {
                values[i] = sRand.nextInt(2);
            }
            System.arraycopy(values, 0, subSet, 0, setOfSolutions[0].length);
        }
        return setOfSolutions;
    }

    /**
     * Does this feature selection algorithm return an ordered feature set?
     *
     * @return True if the set is ordered.
     */
    @Override
    public boolean isOrdered() {
        return true;
    }

    /**
     * Selects features according to this selection algorithm from the specified dataset.
     * @param dataset The dataset to use.
     * @return A selected feature set.
     */
    @Override
    public SelectedFeatureSet select(Dataset<Label> dataset) {
        ImmutableFeatureMap FMap = new ImmutableFeatureMap(dataset.getFeatureMap());
        setOfSolutions = GeneratePopulation(dataset.getFeatureMap().size());
        List<FeatureSet_FScore_Container> subSet_fScores = new ArrayList<>();
        SelectedFeatureSet selectedFeatureSet = null;
        // Update the solution based on the levy flight function
        for (int i = 0; i < maxIteration; i++) {
            IntStream.range(0, setOfSolutions.length).parallel().forEach(subSet -> {
                AtomicInteger currentIter = new AtomicInteger(subSet);
                int[] evolvedSolution = Arrays.stream(setOfSolutions[subSet]).map(x -> (int) transferFunction.applyAsDouble(x + stepSizeScaling * Math.pow(currentIter.get() + 1, -lambda))).toArray();
                int[] randomCuckoo = setOfSolutions[new Random().nextInt(setOfSolutions.length)];
                if (EvaluateSolution(this, dataset, FMap, evolvedSolution) > EvaluateSolution(this, dataset, FMap, randomCuckoo))
                    System.arraycopy(evolvedSolution, 0, setOfSolutions[subSet], 0, evolvedSolution.length);
                // Update the solution based on the abandone nest function
                if (new Random().nextDouble() < worstNestProbability) {
                    int r1 = new Random().nextInt(setOfSolutions.length);
                    int r2 = new Random().nextInt(setOfSolutions.length);
                    for (var j = 0; j < setOfSolutions[subSet].length; j++)
                        evolvedSolution[j] = (int) transferFunction.applyAsDouble(setOfSolutions[subSet][j] + delta * (setOfSolutions[r1][j] - setOfSolutions[r2][j]));
                    if (EvaluateSolution(this, dataset, FMap, evolvedSolution) > EvaluateSolution(this, dataset, FMap, setOfSolutions[subSet]))
                        System.arraycopy(evolvedSolution, 0, setOfSolutions[subSet], 0, evolvedSolution.length);
                    }
                subSet_fScores.add(new FeatureSet_FScore_Container(setOfSolutions[subSet], EvaluateSolution(this, dataset, FMap, setOfSolutions[subSet])));
            });
            subSet_fScores.sort(Comparator.comparing(FeatureSet_FScore_Container::score).reversed());
            selectedFeatureSet = getSFS(this, dataset, FMap, subSet_fScores.get(0).subSet);
        }
        return selectedFeatureSet;
    }

    @Override
    public FeatureSelectorProvenance getProvenance() {
        return new FeatureSelectorProvenanceImpl(this);
    }
    /**
     * This method is used to compute the fitness score of each solution of the population
     * @param optimizer The optimizer that is used for FS
     * @param dataset The dataset to use
     * @param Fmap The dataset feature map
     * @param solution The current subset of features
     * @return The fitness score of the given subset*/

    private  <T extends FeatureSelector<Label>> double EvaluateSolution(T optimizer, Dataset<Label> dataset, ImmutableFeatureMap Fmap, int... solution) {
        SelectedFeatureDataset<Label> selectedFeatureDataset = new SelectedFeatureDataset<>(dataset,getSFS(optimizer, dataset, Fmap, solution));
        KNNTrainer<Label> KnnTrainer =  new KNNTrainer<Label>(1, new L1Distance(), Runtime.getRuntime().availableProcessors(), new VotingCombiner(), KNNModel.Backend.THREADPOOL, NeighboursQueryFactoryType.BRUTE_FORCE);
        CrossValidation<Label, LabelEvaluation> crossValidation = new CrossValidation<>(KnnTrainer, selectedFeatureDataset, new LabelEvaluator(), 10);
        double avgAccuracy = 0D;
        for (Pair<LabelEvaluation, Model<Label>> ACC : crossValidation.evaluate()) {
            avgAccuracy += ACC.getA().accuracy();}
        avgAccuracy /= crossValidation.getK();
        double correlation = 0;

        return avgAccuracy + 0.001 * (1 - ((double) selectedFeatureDataset.getSelectedFeatures().size() / Fmap.size()) - correlation);
    }

    /**
     * This methid is used to return the selected subset of features
     * @param optimizer The optimizer that is used for FS
     * @param dataset The dataset to use
     * @param Fmap The dataset feature map
     * @param solution The current subset of featurs
     * @return The selected feature set*/

    private  <T extends FeatureSelector<Label>> SelectedFeatureSet getSFS(T optimizer, Dataset<Label> dataset, ImmutableFeatureMap Fmap, int... solution) {
        List<String> names = new ArrayList<>();
        List<Double> scores = new ArrayList<>();
        for (int i = 0; i < solution.length; i++)
            if (solution[i] == 1) {
                names.add(Fmap.get(i).getName());
                scores.add(1D);
            }
        FeatureSetProvenance provenance = new FeatureSetProvenance(SelectedFeatureSet.class.getName(), dataset.getProvenance(), optimizer.getProvenance());

        return new SelectedFeatureSet(names, scores, optimizer.isOrdered(), provenance);
    }

    /**
     * This record is used to hold subset of features with its corresponding fitness score
     */
    record FeatureSet_FScore_Container(int[] subSet, double score) { }
}