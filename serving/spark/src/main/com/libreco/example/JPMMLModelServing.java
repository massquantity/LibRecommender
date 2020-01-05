package libreco.example;

import libreco.serving.jpmml.JavaModelServer;
import org.dmg.pmml.FieldName;
import org.jpmml.evaluator.JavaModel;

import java.util.HashMap;
import java.util.Map;

public class JPMMLModelServing {
    public static void main(String[] args) {
        JavaModelServer jms = new JavaModelServer(
                "serving/spark/src/main/resources/jpmml_model/GBDT_model.xml");

        HashMap<String, Object> featureMap = new HashMap<>();
        featureMap.put("episodes", 13);
        featureMap.put("web_rating", 8.8);
        featureMap.put("members", 10000);
        featureMap.put("type", "TV");
    //    featureMap.put("genre", "Action, Fantasy, Magic, Military, Shounen");

        Map<FieldName, ?> result = jms.predict(featureMap);
        for (Map.Entry<FieldName, ?> field : result.entrySet()) {
            System.out.println(field.getKey() + "\t" + field.getValue());
        }

        for (int i = 0; i < result.size(); ++i) {
            System.out.println(result);
        }

        System.out.println(result.get(new FieldName("probability(1)")));
    }
}
