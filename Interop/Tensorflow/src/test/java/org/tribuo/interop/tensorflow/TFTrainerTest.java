package org.tribuo.interop.tensorflow;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.Test;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Init;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Reshape;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.nn.Conv2d;
import org.tensorflow.op.nn.MaxPool;
import org.tensorflow.op.nn.Relu;
import org.tensorflow.op.random.TruncatedNormal;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TUint8;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.datasource.IDXDataSource;
import org.tribuo.datasource.LibSVMDataSource;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.tribuo.interop.tensorflow.TensorflowModel.INPUT_NAME;

public class TFTrainerTest {

    private static final int PIXEL_DEPTH = 255;
    private static final int IMAGE_SIZE = 28;
    private static final int NUM_LABELS = 10;
    private static final String PADDING_TYPE = "SAME";


    public static Pair<Graph, String> buildGraph() {
        Graph graph = new Graph();

        Ops tf = Ops.create(graph);

        // Inputs
        Placeholder<TFloat32> input = tf.withName(INPUT_NAME).placeholder(TFloat32.class,
                Placeholder.shape(Shape.of(-1, IMAGE_SIZE, IMAGE_SIZE, 1)));

        // Scaling the features
        Constant<TFloat32> centeringFactor = tf.constant(PIXEL_DEPTH / 2.0f);
        Constant<TFloat32> scalingFactor = tf.constant((float) PIXEL_DEPTH);
        Operand<TFloat32> scaledInput = tf.math.div(tf.math.sub(input, centeringFactor), scalingFactor);

        // First conv layer
        Variable<TFloat32> conv1Weights = tf.variable(tf.math.mul(tf.random
                .truncatedNormal(tf.array(5, 5, 1, 32), TFloat32.class,
                        TruncatedNormal.seed(Trainer.DEFAULT_SEED)), tf.constant(0.1f)));
        Conv2d<TFloat32> conv1 = tf.nn.conv2d(scaledInput, conv1Weights, Arrays.asList(1L, 1L, 1L, 1L), PADDING_TYPE);
        Variable<TFloat32> conv1Biases = tf.variable(tf.fill(tf.array(new int[]{32}), tf.constant(0.0f)));
        Relu<TFloat32> relu1 = tf.nn.relu(tf.nn.biasAdd(conv1, conv1Biases));

        // First pooling layer
        MaxPool<TFloat32> pool1 = tf.nn
                .maxPool(relu1, tf.array(1, 2, 2, 1), tf.array(1, 2, 2, 1), PADDING_TYPE);

        // Second conv layer
        Variable<TFloat32> conv2Weights = tf.variable(tf.math.mul(tf.random
                .truncatedNormal(tf.array(5, 5, 32, 64), TFloat32.class,
                        TruncatedNormal.seed(Trainer.DEFAULT_SEED)), tf.constant(0.1f)));
        Conv2d<TFloat32> conv2 = tf.nn.conv2d(pool1, conv2Weights, Arrays.asList(1L, 1L, 1L, 1L), PADDING_TYPE);
        Variable<TFloat32> conv2Biases = tf.variable(tf.fill(tf.array(new int[]{64}), tf.constant(0.1f)));
        Relu<TFloat32> relu2 = tf.nn.relu(tf.nn.biasAdd(conv2, conv2Biases));

        // Second pooling layer
        MaxPool<TFloat32> pool2 = tf.nn.maxPool(relu2, tf.array(1, 2, 2, 1), tf.array(1, 2, 2, 1),
                        PADDING_TYPE);

        // Flatten inputs
        Reshape<TFloat32> flatten = tf.reshape(pool2, tf.concat(Arrays
                .asList(tf.slice(tf.shape(pool2), tf.array(new int[]{0}), tf.array(new int[]{1})),
                        tf.array(new int[]{-1})), tf.constant(0)));

        // Fully connected layer
        Variable<TFloat32> fc1Weights = tf.variable(tf.math.mul(tf.random
                .truncatedNormal(tf.array(IMAGE_SIZE * IMAGE_SIZE * 4, 512), TFloat32.class,
                        TruncatedNormal.seed(Trainer.DEFAULT_SEED)), tf.constant(0.1f)));
        Variable<TFloat32> fc1Biases = tf.variable(tf.fill(tf.array(new int[]{512}), tf.constant(0.1f)));
        Relu<TFloat32> relu3 = tf.nn.relu(tf.math.add(tf.linalg.matMul(flatten, fc1Weights), fc1Biases));

        // Softmax layer
        Variable<TFloat32> fc2Weights = tf.variable(tf.math.mul(tf.random
                .truncatedNormal(tf.array(512, NUM_LABELS), TFloat32.class,
                        TruncatedNormal.seed(Trainer.DEFAULT_SEED)), tf.constant(0.1f)));
        Variable<TFloat32> fc2Biases = tf.variable(tf.fill(tf.array(new int[]{NUM_LABELS}), tf.constant(0.1f)));

        Add<TFloat32> logits = tf.math.add(tf.linalg.matMul(relu3, fc2Weights), fc2Biases);

        tf.init();

        return new Pair<>(graph, logits.op().name());
    }

    @Test
    public void testCNN() throws IOException {
        Pair<Graph, String> p = buildGraph();

        Map<String, Float> gradientParams = new HashMap<>();
        gradientParams.put("learningRate", 0.01f);
        gradientParams.put("initialAccumulatorValue", 0f);

        ExampleTransformer<Label> imageTransformer = new ImageTransformer<>(28, 28, 1);
        OutputTransformer<Label> outputTransformer = new LabelTransformer();

        TFTrainer<Label> trainer = new TFTrainer<>(p.getA(),
                INPUT_NAME,
                p.getB(),
                Init.DEFAULT_NAME,
                GradientOptimiser.ADAGRAD,
                gradientParams,
                imageTransformer,
                outputTransformer,
                16,
                2,
                16);

        LabelFactory labelFactory = new LabelFactory();
        String base = "/Users/apocock/Development/Tribuo/tutorials/";
        System.out.println("Loading data");
        IDXDataSource<Label> trainMNIST = new IDXDataSource<>(Paths.get(base,"train-images-idx3-ubyte.gz"),Paths.get(base,"train-labels-idx1-ubyte.gz"),labelFactory);
        IDXDataSource<Label> testMNIST = new IDXDataSource<>(Paths.get(base,"t10k-images-idx3-ubyte.gz"),Paths.get(base,"t10k-labels-idx1-ubyte.gz"),labelFactory);

        Dataset<Label> train = new MutableDataset<>(trainMNIST);
        Dataset<Label> test = new MutableDataset<>(testMNIST);

        System.out.println("Training model");
        Model<Label> model = trainer.train(train);

        System.out.println("Evaluating model");
        LabelEvaluation eval = new LabelEvaluator().evaluate(model,test);

        System.out.println(eval.toString());

        System.out.println(eval.getConfusionMatrix().toString());
    }
}