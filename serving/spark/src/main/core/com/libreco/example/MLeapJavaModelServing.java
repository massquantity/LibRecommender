package com.libreco.example;

import com.libreco.serving.mleap.JavaModelServer;
import com.libreco.utils.Scala2JavaConverter;
import ml.combust.mleap.core.types.StructField;
import ml.combust.mleap.core.types.StructType;
import ml.combust.mleap.runtime.frame.Row;
import ml.combust.mleap.runtime.javadsl.LeapFrameBuilder;
import ml.combust.mleap.tensor.DenseTensor;

import java.util.ArrayList;
import java.util.List;


public class MLeapJavaModelServing {
    public static void main(String[] args) {
        LeapFrameBuilder builder = new LeapFrameBuilder();
        List<StructField> fields = new ArrayList<>();
        fields.add(builder.createField("name", builder.createString()));
        fields.add(builder.createField("type", builder.createString()));
        fields.add(builder.createField("episodes", builder.createInt()));
        fields.add(builder.createField("web_rating", builder.createDouble()));
        fields.add(builder.createField("members", builder.createInt()));
        StructType schema = builder.createSchema(fields);

        Row features = builder.createRow("Gintama", "TV", 13, 8.8, 10000);
        JavaModelServer jms = new JavaModelServer(
                "serving/spark/src/main/resources/mleap_model/GBDT_model.zip", schema);
        Row result = jms.predict(features);
        for (int i = 0; i < result.size(); ++i) {
            System.out.println(result.get(i));
        }
        System.out.println();
        double pred = (double) result.get(14);
        System.out.println("prediction: " + pred);
        DenseTensor probs = result.getTensor(13).toDense();
        double prob0 = Scala2JavaConverter.parseTensor(probs, 0);
        double prob1 = Scala2JavaConverter.parseTensor(probs, 1);
        double prob2 = Scala2JavaConverter.parseTensor(probs, 2);
        System.out.printf("probabilities(size = %d):%n", probs.size());
        System.out.printf("label  prob%n");
        System.out.printf(" 0.0   %.2f%n", prob0);
        System.out.printf(" 1.0   %.2f%n", prob1);
        System.out.printf(" 2.0   %.2f%n", prob2);
    }
}
