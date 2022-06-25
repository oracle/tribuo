/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import org.tribuo.Model;

import java.util.HashMap;
import java.util.Map;

public class ModelDetails {
    private static final ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);;
    private final String schemaVersion;
    private final String modelType;
    private final String modelPackage;
    private final String tribuoVersion;
    private final String javaVersion;
    private final Map<String, Object> configuredParams = new HashMap<>();

    public ModelDetails(Model<?> model) {
        schemaVersion = "1.0";
        modelType = model.getClass().getSimpleName();
        modelPackage = model.getClass().getTypeName();
        tribuoVersion = model.getProvenance().getTribuoVersion();
        javaVersion = model.getProvenance().getJavaVersion();

        Map<String,?> parameters = model.getProvenance().getTrainerProvenance().getConfiguredParameters();
        for (String key : parameters.keySet())
            configuredParams.put(key, parameters.get(key));
    }

    public ModelDetails(JsonNode modelDetailsJson) throws JsonProcessingException {
        schemaVersion = modelDetailsJson.get("schema-version").textValue();
        modelType = modelDetailsJson.get("model-type").textValue();
        modelPackage = modelDetailsJson.get("model-package").textValue();
        tribuoVersion = modelDetailsJson.get("tribuo-version").textValue();
        javaVersion = modelDetailsJson.get("java-version").textValue();
        Map<?,?> params = mapper.readValue(modelDetailsJson.get("configured-parameters").toString(), Map.class);
        for (var key : params.keySet())
            configuredParams.put(key.toString(), params.get(key));
    }

    public String getSchemaVersion() {
        return schemaVersion;
    }

    public String getModelType() {
        return modelType;
    }

    public String getModelPackage() {
        return modelPackage;
    }

    public String getTribuoVersion() {
        return tribuoVersion;
    }

    public String getJavaVersion() {
        return javaVersion;
    }

    public Map<String, Object> getConfiguredParams() {
        return configuredParams;
    }

    public ObjectNode toJson() {
        ObjectNode modelDetailsObject = mapper.createObjectNode();
        modelDetailsObject.put("schema-version", schemaVersion);
        modelDetailsObject.put("model-type", modelType);
        modelDetailsObject.put("model-package", modelPackage);
        modelDetailsObject.put("tribuo-version", tribuoVersion);
        modelDetailsObject.put("java-version", javaVersion);
        ObjectNode paramsArr = paramsToJson(null, configuredParams);
        modelDetailsObject.set("configured-parameters", paramsArr);
        return modelDetailsObject;
    }

    private ObjectNode paramsToJson(String name, Map<?,?> params) {
        ObjectNode paramsArr = mapper.createObjectNode();
        if (name != null) paramsArr.put("className", name);
        for (var key : params.keySet()) {
            if (params.get(key) instanceof ConfiguredObjectProvenance) {
                ConfiguredObjectProvenance prov = (ConfiguredObjectProvenance) params.get(key);
                ObjectNode nestedParam = paramsToJson(prov.getClassName(), prov.getConfiguredParameters());
                paramsArr.set(key.toString(), nestedParam);
            } else if (params.get(key) instanceof Map) {
                Map<?,?> map = (Map<?,?>) params.get(key);
                ObjectNode nestedParam = paramsToJson(null, map);
                paramsArr.set(key.toString(), nestedParam);
            } else if (isNumeric(params.get(key).toString())) {
                if (params.get(key).toString().contains("."))
                    paramsArr.put(key.toString(), Double.parseDouble(params.get(key).toString()));
                else
                    paramsArr.put(key.toString(), Integer.parseInt(params.get(key).toString()));
            } else paramsArr.put(key.toString(), params.get(key).toString());
        }
        return paramsArr;
    }

    private boolean isNumeric(String str) {
        try {
            double val = Double.parseDouble(str);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    @Override
    public String toString() {
        return toJson().toPrettyString();
    }
}
