package libreco.example;

import libreco.serving.mleap.JavaModelServer;
import libreco.Scala2JavaConverter;
import ml.combust.mleap.core.types.StructField;
import ml.combust.mleap.core.types.StructType;
import ml.combust.mleap.runtime.frame.Row;
import ml.combust.mleap.runtime.javadsl.LeapFrameBuilder;
import ml.combust.mleap.tensor.DenseTensor;

import java.util.ArrayList;
import java.util.List;

import scala.collection.Seq;

public class MLeapJavaModelServing {
    // "type", "episodes", "web_rating", "members"
    public static void main(String[] args) {
        LeapFrameBuilder builder = new LeapFrameBuilder();
        List<StructField> fields = new ArrayList<>();
        fields.add(builder.createField("type", builder.createString()));
        fields.add(builder.createField("episodes", builder.createInt()));
        fields.add(builder.createField("web_rating", builder.createDouble()));
        fields.add(builder.createField("members", builder.createInt()));
        StructType schema = builder.createSchema(fields);

        Row features = builder.createRow("TV", 13, 8.8, 10000);
        JavaModelServer jms = new JavaModelServer(
                "serving/spark/src/main/resources/mleap_model/GBDT_model.zip", schema);
        Row result = jms.predict(features);

        for (int i = 0; i < result.size(); ++i) {
            System.out.println(result.get(i));
        }
        System.out.println();
        double rating = (double) result.get(7);
    //    System.out.println(rating.size() + " "  + rating.dimensions());
    //    System.out.println(Scala2JavaConverter.parseCtr(rating));
        System.out.println(rating);
    }
}
