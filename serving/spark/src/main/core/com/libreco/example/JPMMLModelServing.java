package com.libreco.example;

import com.libreco.serving.jpmml.JavaModelServer;
import org.dmg.pmml.FieldName;
import org.jpmml.evaluator.JavaModel;

import java.util.HashMap;
import java.util.Map;

public class JPMMLModelServing {
    public static void main(String[] args) {
        JavaModelServer jms = new JavaModelServer(
                "serving/spark/src/main/resources/jpmml_model/jpmml_model.xml");

        HashMap<String, Object> featureMap = new HashMap<>();
        featureMap.put("episodes", 13);
        featureMap.put("web_rating", 8.8);
        featureMap.put("members", 10000);
        featureMap.put("type", "TV");
    //    featureMap.put("genre", "Action, Fantasy, Magic, Military, Shounen");

        Map<FieldName, ?> result = jms.predict(featureMap);
        System.out.println(result);
        for (Map.Entry<FieldName, ?> field : result.entrySet()) {
            System.out.println(field.getKey() + "\t" + field.getValue());
        }

        System.out.println();
        System.out.println("prediction: " + result.get(new FieldName("pred")));
        System.out.printf("probabilities:%n");
        System.out.printf("label  prob%n");
        System.out.printf(" 0.0  %.4f%n", result.get(new FieldName("prob(0)")));
        System.out.printf(" 1.0  %.4f%n", result.get(new FieldName("prob(1)")));
        System.out.printf(" 2.0  %.4f%n", result.get(new FieldName("prob(2)")));
    }
}
