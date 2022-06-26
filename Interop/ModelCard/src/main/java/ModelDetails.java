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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import org.tribuo.Model;

import java.util.Map;

public final class ModelDetails {
    private static final ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);;
    private static final String schemaVersion = "1.0";
    private final String modelType;
    private final String modelPackage;
    private final String tribuoVersion;
    private final String javaVersion;
    private final JsonNode configuredParams;

    public ModelDetails(Model<?> model) {
        modelType = model.getClass().getSimpleName();
        modelPackage = model.getClass().getTypeName();
        tribuoVersion = model.getProvenance().getTribuoVersion();
        javaVersion = model.getProvenance().getJavaVersion();
        configuredParams = processNestedParams(null, model.getProvenance().getTrainerProvenance().getConfiguredParameters());
    }

    public ModelDetails(JsonNode modelDetailsJson) {
        modelType = modelDetailsJson.get("model-type").textValue();
        modelPackage = modelDetailsJson.get("model-package").textValue();
        tribuoVersion = modelDetailsJson.get("tribuo-version").textValue();
        javaVersion = modelDetailsJson.get("java-version").textValue();
        configuredParams = modelDetailsJson.get("configured-parameters");
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

    public String getConfiguredParams() {
        return configuredParams.toPrettyString();
    }

    public ObjectNode toJson() {
        ObjectNode modelDetailsObject = mapper.createObjectNode();
        modelDetailsObject.put("schema-version", schemaVersion);
        modelDetailsObject.put("model-type", modelType);
        modelDetailsObject.put("model-package", modelPackage);
        modelDetailsObject.put("tribuo-version", tribuoVersion);
        modelDetailsObject.put("java-version", javaVersion);
        modelDetailsObject.set("configured-parameters", configuredParams);
        return modelDetailsObject;
    }

    private ObjectNode processNestedParams(String name, Map<?,?> params) {
        ObjectNode paramsObject = mapper.createObjectNode();
        if (name != null) {
            paramsObject.put("className", name);
        }
        for (Map.Entry<?,?> entry : params.entrySet()) {
            if (entry.getValue() instanceof ConfiguredObjectProvenance prov) {
                ObjectNode nestedParam = processNestedParams(prov.getClassName(), prov.getConfiguredParameters());
                paramsObject.set(entry.getKey().toString(), nestedParam);
            } else if (entry.getValue() instanceof Map<?, ?> map) {
                ObjectNode nestedParam = processNestedParams(null, map);
                paramsObject.set(entry.getKey().toString(), nestedParam);
            } else if (isNumeric(entry.getValue().toString())) {
                if (entry.getValue().toString().contains(".")) {
                    paramsObject.put(entry.getKey().toString(), Double.parseDouble(entry.getValue().toString()));
                } else {
                    paramsObject.put(entry.getKey().toString(), Integer.parseInt(entry.getValue().toString()));
                }
            } else {
                paramsObject.put(entry.getKey().toString(), entry.getValue().toString());
            }
        }
        return paramsObject;
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
