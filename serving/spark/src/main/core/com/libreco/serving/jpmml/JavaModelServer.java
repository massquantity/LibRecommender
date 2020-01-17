package libreco.serving.jpmml;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.Evaluator;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.InputField;
import org.jpmml.evaluator.ModelEvaluatorFactory;
import org.jpmml.model.PMMLUtil;
import org.xml.sax.SAXException;

import javax.xml.bind.JAXBException;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class JavaModelServer {
    private String modelPath;
    private Evaluator model;

    public JavaModelServer(String modelPath) {
        this.modelPath = modelPath;
    }

    private void loadModel() {
        PMML pmml;
        InputStream is = null;
        try {
            is = new FileInputStream(new File(modelPath));
            pmml = PMMLUtil.unmarshal(is);
            ModelEvaluatorFactory mef = ModelEvaluatorFactory.newInstance();
            this.model = mef.newModelEvaluator(pmml);
            this.model.verify();
            List<InputField> inputFields = model.getInputFields();
            for (InputField inputField : inputFields) {
                System.out.println(inputField.getName().getValue());
            }
        } catch (IOException | SAXException | JAXBException e) {
            System.err.println(e);
        } finally {
            try {
                assert is != null;
                is.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public Map<FieldName, ?> predict(Map<String, ?> featureMap) {
        if (model == null)
            loadModel();
        if (featureMap == null)
        {
            System.err.println("features are null");
            return null;
        }

        List<InputField> inputFields = model.getInputFields();
        Map<FieldName, FieldValue> pmmlFeatureMap = new LinkedHashMap<>();
        for (InputField inputField: inputFields) {
            if (featureMap.containsKey(inputField.getName().getValue())) {
                Object value = featureMap.get(inputField.getName().getValue());
                pmmlFeatureMap.put(inputField.getName(), inputField.prepare(value));
            } else {
                System.err.println("lack of feature: " + inputField.getName().getValue());
                return null;
            }
        }
        return model.evaluate(pmmlFeatureMap);
    }
}




