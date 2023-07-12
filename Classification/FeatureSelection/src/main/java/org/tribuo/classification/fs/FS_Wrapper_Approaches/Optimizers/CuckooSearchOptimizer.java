package FS_Wrapper_Approaches.Optimizers;

import FS.Discreeting.TransferFunction;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.FeatureSelector;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;
import org.tribuo.SelectedFeatureSet;
import org.tribuo.classification.Label;
import org.tribuo.classification.ensemble.VotingCombiner;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.common.nearest.KNNModel;
import org.tribuo.common.nearest.KNNTrainer;
import org.tribuo.dataset.SelectedFeatureDataset;
import org.tribuo.evaluation.CrossValidation;
import org.tribuo.math.distance.L1Distance;
import org.tribuo.math.neighbour.NeighboursQueryFactoryType;
import org.tribuo.provenance.FeatureSelectorProvenance;
import org.tribuo.provenance.FeatureSetProvenance;
import org.tribuo.provenance.impl.FeatureSelectorProvenanceImpl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.SplittableRandom;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
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
public  final class CuckooSearchOptimizer implements FeatureSelector<Label> {
    private final Trainer<Label> trainer;
    private final TransferFunction transferFunction;
    private final int populationSize;
    private final double stepSizeScaling;
    private final double lambda;
    private final double worstNestProbability;
    private final double delta;
    private final double mutationRate;
    private int [][] setOfSolutions;
    private final int maxIteration;
    private final SplittableRandom rng;
    private final int seed;

    /**
     * The default constructor for feature selection based on Cuckoo Search Algorithm
     */
    public CuckooSearchOptimizer() {
        this.trainer =  new KNNTrainer<>(1, new L1Distance(), Runtime.getRuntime().availableProcessors(), new VotingCombiner(), KNNModel.Backend.THREADPOOL, NeighboursQueryFactoryType.BRUTE_FORCE);
        this.transferFunction = TransferFunction.V2;
        this.populationSize = 50;
        this.stepSizeScaling = 2d;
        this.lambda = 2d;
        this.worstNestProbability = 0.1d;
        this.delta = 1.5d;
        this.mutationRate = 0.2d;
        this.maxIteration = 3;
        this.seed = 12345;
        this.rng = new SplittableRandom(seed);
    }

    /**
     * Constructs the wrapper feature selection based on cuckoo search algorithm
     * @param trainer The used trainer in the evaluation process
     * @param transferFunction The transfer function to convert continuous values to binary ones
     * @param populationSize The size of the solution in the initial population
     * @param maxIteration The number of times that is used to enhance generation
     * @param seed This seed is required for the SplittableRandom
     */
    public CuckooSearchOptimizer(Trainer<Label> trainer, TransferFunction transferFunction, int populationSize, int maxIteration, int seed) {
        this.trainer = trainer;
        this.transferFunction = transferFunction;
        this.populationSize = populationSize;
        this.stepSizeScaling = 2d;
        this.lambda = 2d;
        this.worstNestProbability = 1.5d;
        this.delta = 1.5d;
        this.mutationRate = 0.2d;
        this.maxIteration = maxIteration;
        this.seed = seed;
        this.rng = new SplittableRandom(seed);
    }

    /**
     * Constructs the wrapper feature selection based on cuckoo search algorithm
     * @param trainer The used trainer in the evaluation process
     * @param transferFunction The transfer function to convert continuous values to binary ones
     * @param populationSize The size of the solution in the initial population
     * @param stepSizeScaling The cuckoo step size
     * @param lambda The lambda of the levy flight function
     * @param worstNestProbability The fraction of the nests to be abandoned
     * @param delta The delta that is used in the abandon nest function
     * @param mutationRate The proportion to apply the mutation operator
     * @param maxIteration The number of times that is used to enhance generation
     * @param seed This seed is required for the SplittableRandom
     */
    public CuckooSearchOptimizer(Trainer<Label> trainer, TransferFunction transferFunction, int populationSize, double stepSizeScaling, double lambda, double worstNestProbability, double delta, double mutationRate, int maxIteration, int seed) {
        this.trainer = trainer;
        this.transferFunction = transferFunction;
        this.populationSize = populationSize;
        this.stepSizeScaling = stepSizeScaling;
        this.lambda = lambda;
        this.worstNestProbability = worstNestProbability;
        this.delta = delta;
        this.mutationRate = mutationRate;
        this.maxIteration = maxIteration;
        this.seed = seed;
        this.rng = new SplittableRandom(seed);
    }

    /**
     * This method is used to generate the initial population (set of solutions)
     * @param totalNumberOfFeatures The number of features in the given dataset
     * @return The population of subsets of selected features
     */
    private int[][] GeneratePopulation(int totalNumberOfFeatures) {
        setOfSolutions = new int[this.populationSize][totalNumberOfFeatures];
        for (int[] subSet : setOfSolutions) {
            int[] values = new int[subSet.length];
            for (int i = 0; i < values.length; i++) {
                values[i] = rng.nextInt(2);
            }
            System.arraycopy(values, 0, subSet, 0, setOfSolutions[0].length);
        }
        return setOfSolutions;
    }

    /**
     * Does this feature selection algorithm return an ordered feature set?
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
        List<CuckooSearchFeatureSet> subSet_fScores = Arrays.stream(setOfSolutions).map(setOfSolution -> new CuckooSearchFeatureSet(setOfSolution, evaluateSolution(this, trainer,dataset, FMap, setOfSolution))).sorted(Comparator.comparing(CuckooSearchFeatureSet::score).reversed()).collect(Collectors.toList());
        SelectedFeatureSet selectedFeatureSet = null;
        for (int i = 0; i < maxIteration; i++) {
            for (int solution = 0; solution < populationSize; solution++) {
                AtomicInteger subSet = new AtomicInteger(solution);
                // Update the solution based on the levy flight function
                int[] evolvedSolution = Arrays.stream(setOfSolutions[subSet.get()]).map(x -> (int) transferFunction.applyAsDouble(x + stepSizeScaling * Math.pow(subSet.get() + 1, -lambda))).toArray();
                int[] randomCuckoo = setOfSolutions[rng.nextInt(setOfSolutions.length)];
                keepBestAfterEvaluation(dataset, trainer, FMap, evolvedSolution, randomCuckoo);
                // Update the solution based on the abandone nest function
                if (rng.nextDouble() < worstNestProbability) {
                    int r1 = rng.nextInt(setOfSolutions.length);
                    int r2 = rng.nextInt(setOfSolutions.length);
                    for (int j = 0; j < setOfSolutions[subSet.get()].length; j++) {
                        evolvedSolution[j] = (int) transferFunction.applyAsDouble(setOfSolutions[subSet.get()][j] + delta * (setOfSolutions[r1][j] - setOfSolutions[r2][j]));
                    }
                    keepBestAfterEvaluation(dataset, trainer, FMap, evolvedSolution, setOfSolutions[subSet.get()]);
                }
                // Update the solution based on mutation operator
                int[] mutedSolution = mutation(setOfSolutions[subSet.get()]);
                keepBestAfterEvaluation(dataset, trainer, FMap, mutedSolution, setOfSolutions[subSet.get()]);
                // Update the solution based on inversion mutation
                mutedSolution = inversionMutation(setOfSolutions[subSet.get()]);
                keepBestAfterEvaluation(dataset, trainer, FMap, mutedSolution, setOfSolutions[subSet.get()]);
                // Update the solution based on swapped mutation
                mutedSolution = swappedMutation(setOfSolutions[subSet.get()]);
                keepBestAfterEvaluation(dataset, trainer, FMap, mutedSolution, setOfSolutions[subSet.get()]);
                // Updata the solution based on Jaya operator
                int[] jayaSolution = jayaOperator(setOfSolutions[subSet.get()], subSet_fScores.get(0).subSet(), subSet_fScores.get(subSet_fScores.size() - 1).subSet());
                keepBestAfterEvaluation(dataset, trainer, FMap, jayaSolution, setOfSolutions[subSet.get()]);
            }
            Arrays.stream(setOfSolutions).map(subSet -> new CuckooSearchFeatureSet(subSet, evaluateSolution(this, trainer, dataset, FMap, subSet))).forEach(subSet_fScores::add);
            subSet_fScores.sort(Comparator.comparing(CuckooSearchFeatureSet::score).reversed());
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
     * @param trainer The used trainer in the evaluation process
     * @param dataset The dataset to use
     * @param Fmap The dataset feature map
     * @param solution The current subset of features
     * @return The fitness score of the given subset
     */
    private  <T extends FeatureSelector<Label>> double evaluateSolution(T optimizer, Trainer<Label> trainer, Dataset<Label> dataset, ImmutableFeatureMap Fmap, int... solution) {
        SelectedFeatureDataset<Label> selectedFeatureDataset = new SelectedFeatureDataset<>(dataset,getSFS(optimizer, dataset, Fmap, solution));
        CrossValidation<Label, LabelEvaluation> crossValidation = new CrossValidation<>(trainer, selectedFeatureDataset, new LabelEvaluator(), 10);
        double avgAccuracy = 0d;
        for (Pair<LabelEvaluation, Model<Label>> ACC : crossValidation.evaluate()) {
            avgAccuracy += ACC.getA().accuracy();
        }
        avgAccuracy /= crossValidation.getK();

        return avgAccuracy + 0.001 * (1 - ((double) selectedFeatureDataset.getSelectedFeatures().size() / Fmap.size()));
    }

    /**
     * This methid is used to return the selected subset of features
     * @param optimizer The optimizer that is used for FS
     * @param dataset The dataset to use
     * @param featureMap The dataset feature map
     * @param solution The current subset of featurs
     * @return The selected feature set
     */
    private  <T extends FeatureSelector<Label>> SelectedFeatureSet getSFS(T optimizer, Dataset<Label> dataset, ImmutableFeatureMap featureMap, int... solution) {
        List<String> names = new ArrayList<>();
        List<Double> scores = new ArrayList<>();
        for (int i = 0; i < solution.length; i++) {
            if (solution[i] == 1) {
                names.add(featureMap.get(i).getName());
                scores.add(1d);
            }
        }
        FeatureSetProvenance provenance = new FeatureSetProvenance(SelectedFeatureSet.class.getName(), dataset.getProvenance(), optimizer.getProvenance());

        return new SelectedFeatureSet(names, scores, optimizer.isOrdered(), provenance);
    }

    /**
     * @param dataset The dataset to use
     * @param trainer The used trainer in the evaluation process
     * @param FMap The map of selected features
     * @param alteredSolution The modified solution
     * @param oldSolution The old solution
     */
    private void keepBestAfterEvaluation(Dataset<Label> dataset, Trainer<Label> trainer, ImmutableFeatureMap FMap, int[] alteredSolution, int[] oldSolution) {
        if (evaluateSolution(this, trainer, dataset, FMap, alteredSolution) > evaluateSolution(this, trainer, dataset, FMap, oldSolution)) {
            System.arraycopy(alteredSolution, 0, oldSolution, 0, alteredSolution.length);
        }
    }

    /**
     * The simple mutation method of Genetic algorithm
     * <p>
     * see:
     * <pre>
     * Steven Bayer and Lui Wang.
     * "A Genetic Algorithm Programming Environment: Splicer", 1991.
     * </pre>
     * @param currentSolution The solution to be altered by the mutation operator
     * @return The altered solution after mutation
     */
    private int[] mutation(int... currentSolution) {
        return Arrays.stream(currentSolution).map(x -> ThreadLocalRandom.current().nextDouble() < mutationRate ? 1 - x : x).toArray();
    }

    /**
     * The inversion mutation
     * <p>
     * see:
     * <pre>
     * Nitashs Soni and Tapsa Kumar.
     * "Study of Various Mutation Operators in Genetic Algorithms", 2014.
     * </pre>
     * @param currentSolution The solution to be altered by the mutation operator
     * @return The altered solution after inversion mutation
     */
    private int[] inversionMutation(int... currentSolution) {
        int rand1 = new Random().nextInt(currentSolution.length);
        int rand2 = new Random().nextInt(currentSolution.length);
        while (rand1 >= rand2) {
            rand1 = new Random().nextInt(currentSolution.length);
            rand2 = new Random().nextInt(currentSolution.length);
        }
        for (; rand1 < rand2; rand1++) {
            currentSolution[rand1] = 1 - currentSolution[rand1];
        }
        return currentSolution;
    }

    /**
     * Sswapped mutation
     * <p>
     * see:
     * <pre>
     * Ming-Wen Tsai et al.
     * "A Two-Dimensional Genetic Algorithm and Its Application to Aircraft Scheduling Problem", 2015.
     * </pre>
     * @param currentSolution The solution to be altered by the mutation operator
     * @return The altered solution after swapped mutation
     */
    private int[] swappedMutation(int... currentSolution) {
        int firstGeneIndex = new Random().nextInt(currentSolution.length);
        int secondGeneIndex = new Random().nextInt(currentSolution.length);
        int secondGene = currentSolution[secondGeneIndex];
        currentSolution[secondGeneIndex] = currentSolution[firstGeneIndex];
        currentSolution[firstGeneIndex] = secondGene;
        return currentSolution;
    }

    /**
     * The main equation of Jaya optimization algorithm
     * <p>
     * see:
     * <pre>
     * Venkata Rao.
     * "Jaya: A simple and new optimization algorithm for solving constrained and unconstrained optimization problems", 2016.
     * </pre>
     * @param currentSolution The solution to be altered by the jaya operator
     * @param currentBest The best solution in the current generation
     * @param currentWorst The worst solution in the current generation
     * @return The altered solution after appling jaya operator
     */
    private int[] jayaOperator(int[] currentSolution, int[] currentBest, int[] currentWorst) {
        int[] newSolution = new int[currentSolution.length];
        Arrays.setAll(newSolution, i -> (int) transferFunction.applyAsDouble(currentSolution[i] + new Random().nextDouble() * (currentBest[i] - currentSolution[i]) - new Random().nextDouble() * (currentWorst[i] - currentSolution[i])));
        return newSolution;
    }

    /**
     * This record is used to hold subset of features with its corresponding fitness score
     */
    record CuckooSearchFeatureSet(int[] subSet, double score) { }
}
