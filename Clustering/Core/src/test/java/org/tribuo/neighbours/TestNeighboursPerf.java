package org.tribuo.neighbours;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.DataSource;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.MutableDataset;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.example.GaussianClusterDataSource;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.neighbour.NeighboursQuery;
import org.tribuo.math.neighbour.bruteforce.NeighboursBruteForce;
import org.tribuo.math.neighbour.bruteforce.NeighboursBruteForceFactory;
import org.tribuo.math.neighbour.bruteforce.NeighboursBruteForceNaive;
import org.tribuo.math.neighbour.bruteforce.NeighboursBruteForceNaiveFactory;
import org.tribuo.util.Util;

import java.util.logging.Level;
import java.util.logging.Logger;


public class TestNeighboursPerf {

    private static final Logger logger = Logger.getLogger(NeighboursBruteForce.class.getName());

    @BeforeAll
    public static void setup() {
        logger.setLevel(Level.INFO);
    }

    private SGDVector[] getVectors(int numSamples, long seed) {
        DataSource<ClusterID> dataSource = new GaussianClusterDataSource(numSamples, seed);
        Dataset<ClusterID> dataset = new MutableDataset<>(dataSource);

        ImmutableFeatureMap featureMap = dataset.getFeatureIDMap();
        SGDVector[] vectors = new SGDVector[dataset.size()];
        int n = 0;
        for (Example<ClusterID> example : dataset) {
            vectors[n] = DenseVector.createDenseVector(example, featureMap, false);
            n++;
        }
        return vectors;
    }

    private void executeQuery(NeighboursQuery nq, int k) {
        long startTime = System.currentTimeMillis();
        nq.queryAll(k);
        long endTime = System.currentTimeMillis();
        logger.info("Query took " + Util.formatDuration(startTime,endTime));
        logger.info("");
    }

    private void doTestIteration(NeighboursBruteForceFactory nbfFactory, NeighboursBruteForceNaiveFactory nbfnFactory) {
        SGDVector[] data = getVectors(20000, 1L);

        logger.info("Target implementation: small dataset, small k");
        NeighboursBruteForce nbf = nbfFactory.createNeighboursQuery(data);
        executeQuery(nbf, 5);

        logger.info("Naive implementation: small dataset, small k");
        NeighboursBruteForceNaive nbfn = nbfnFactory.createNeighboursQuery(data);
        executeQuery(nbfn, 5);
        logger.info("");

        logger.info("Target implementation: small dataset, medium k");
        executeQuery(nbf, 50);

        logger.info("Naive implementation: small dataset, medium k");
        executeQuery(nbfn, 50);
        logger.info("");

        logger.info("Target implementation: small dataset, large k");
        executeQuery(nbf, 200);

        logger.info("Naive implementation: small dataset, large k");
        executeQuery(nbfn, 200);
        logger.info("");

        logger.info("");
        ////

        data = getVectors(100000, 2L);

        logger.info("Target implementation: medium dataset, small k");
        nbf = nbfFactory.createNeighboursQuery(data);
        executeQuery(nbf, 5);

        logger.info("Naive implementation:: medium dataset, small k");
        nbfn = nbfnFactory.createNeighboursQuery(data);
        executeQuery(nbfn, 5);
        logger.info("");

        logger.info("Target implementation: medium dataset, medium k");
        executeQuery(nbf, 50);

        logger.info("Naive implementation: medium dataset, medium k");
        executeQuery(nbfn, 50);
        logger.info("");

        logger.info("Target implementation: medium dataset, large k");
        executeQuery(nbf, 200);

        logger.info("Naive implementation: medium dataset, large k");
        executeQuery(nbfn, 200);

        logger.info("");
        ////

        data = getVectors(200000, 3L);

        logger.info("Target implementation: big dataset, small k");
        nbf = nbfFactory.createNeighboursQuery(data);
        executeQuery(nbf, 5);

        logger.info("Naive implementation: big dataset, small k");
        nbfn = nbfnFactory.createNeighboursQuery(data);
        executeQuery(nbfn, 5);

        logger.info("Target implementation: big dataset, medium k");
        executeQuery(nbf, 50);

        logger.info("Naive implementation: big dataset, medium k");
        executeQuery(nbfn, 50);

        logger.info("Target implementation: big dataset, large k");
        executeQuery(nbf, 200);

        logger.info("Naive implementation: big dataset, large k");
        executeQuery(nbfn, 200);

        logger.info("");
        ////
    }

    //@Test
    public void testSingleThreadedQueries() {
        NeighboursBruteForceFactory nbfFactory = new NeighboursBruteForceFactory(DistanceType.L2, 1);
        NeighboursBruteForceNaiveFactory nbfnFactory = new NeighboursBruteForceNaiveFactory(DistanceType.L2, 1);

        logger.info("PERFORMING SINGLE THREADED TESTS...");
        logger.info("");
        doTestIteration(nbfFactory, nbfnFactory);
    }

    //@Test
    public void testMultiThreadQueries() {
        NeighboursBruteForceFactory nbfFactory = new NeighboursBruteForceFactory(DistanceType.L2, 4);
        NeighboursBruteForceNaiveFactory nbfnFactory = new NeighboursBruteForceNaiveFactory(DistanceType.L2, 4);

        logger.info("PERFORMING MULTI-THREADED TESTS...");
        logger.info("");
        doTestIteration(nbfFactory, nbfnFactory);
    }
}
