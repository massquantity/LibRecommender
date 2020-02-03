import org.tensorflow.*;
import org.tensorflow.Graph;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;


public class TFJava {
    public static void main(String[] args) {
        byte[] graphDef = loadTFModel("serving/models/Java/FM_Java.pb");
        int[][] indicesOrig = {{0, 4, 10, 45, 45, 45, 3985, 6053},
                              {1, 4, 22, 36, 43, 45, 2341, 6071}};
        float[][] valuesOrig = {{1, 1, 1, 1, 1, 1, 1, 1},
                                {1, 1, 1, 1, 1, 1, 1, 1}};

        Tensor<Integer> indices = convertArrayToTensor(indicesOrig);
        Tensor<Float> values = convertArrayToTensor(valuesOrig);
        Graph g = new Graph();
        g.importGraphDef(graphDef);
        Session sess = new Session(g);
        Tensor probs = sess.runner().feed("indices", indices).feed("values", values).
                fetch("prob").run().get(0);
        Tensor preds = sess.runner().feed("indices", indices).feed("values", values).
                fetch("pred").run().get(0);

        long[] resShape = probs.shape();
        int rs = (int) resShape[0];
        float probResult[] = new float[rs];
        float predResult[] = new float[rs];
        probs.copyTo(probResult);
        preds.copyTo(predResult);

        System.out.println("prob: " + probResult[0] + ", pred: " + (int)predResult[0]);
    }


    private static byte[] loadTFModel(String path) {
        try {
            return Files.readAllBytes(Paths.get(path));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    private static Tensor<Float> convertArrayToTensor(float[][] input) {
        return Tensors.create(input);
    }

    private static Tensor<Integer> convertArrayToTensor(int[][] input) {
        return Tensors.create(input);
    }

}
