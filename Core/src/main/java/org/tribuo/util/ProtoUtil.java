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

package org.tribuo.util;

import com.google.protobuf.Any;
import com.google.protobuf.GeneratedMessageV3.Builder;
import com.oracle.labs.mlrg.olcut.util.Pair;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;

import org.tribuo.ProtoSerializable;
import org.tribuo.ProtobufClass;
import org.tribuo.ProtobufField;
import org.tribuo.protos.core.HasherProto;
import org.tribuo.protos.core.ModHashCodeHasherProto;
import org.tribuo.protos.core.VariableInfoProto;

/**
 * Utilities for working with Tribuo protobufs.
 */
public final class ProtoUtil {

    public static final String DESERIALIZATION_METHOD_NAME = "deserializeFromProto";

    private static final Map<Pair<Integer, String>, String> REDIRECT_MAP = new HashMap<>();

    /**
     * Adds a redirect mapping to the internal redirection map.
     * <p>
     * This is used when a class name changes, to allow old protobufs to be deserialized into
     * the new class.
     * @param input The version and class name to redirect.
     * @param targetClassName The class name that should be used to deserialize the protobuf.
     */
    public static void registerRedirect(Pair<Integer, String> input, String targetClassName) {
        if (REDIRECT_MAP.containsKey(input)) {
            throw new IllegalArgumentException("Redirect map is append only, key " + input + " already has mapping " + REDIRECT_MAP.get(input));
        } else {
            REDIRECT_MAP.put(input, targetClassName);
        }
    }

    /**
     * Instantiates the class from the supplied protobuf fields.
     * <p>
     * Deserialization proceeds as follows:
     * <ul>
     *     <li>Check to see if there is a valid redirect for this version & class name tuple.
     *     If there is then the new class name is used for the following steps.</li>
     *     <li>Lookup the class name and instantiate the {@link Class} object.</li>
     *     <li>Find the 3 arg static method {@code  deserializeFromProto(int version, String className, com.google.protobuf.Any message)}.</li>
     *     <li>Call the method passing along the original three arguments (note this uses the
     *     original class name even if a redirect has been applied).</li>
     *     <li>Return the freshly constructed object, or rethrow any runtime exceptions.</li>
     * </ul>
     * <p>
     * Throws {@link IllegalStateException} if:
     * <ul>
     *     <li>the requested class could not be found on the classpath/modulepath</li>
     *     <li>the requested class does not have the necessary 3 arg constructor</li>
     *     <li>the constructor could not be invoked due to its accessibility, or is in some other way invalid</li>
     *     <li>the constructor threw an exception</li>
     * </ul>
     * @param version The version number of the protobuf.
     * @param className The class name of the serialized object.
     * @param message The object's serialized representation.
     * @return The deserialized object.
     */
    public static Object instantiate(int version, String className, Any message) {
        Pair<Integer, String> key = new Pair<>(version, className);
        String targetClassName = REDIRECT_MAP.getOrDefault(key, className);
        try {
            Class<?> targetClass = Class.forName(targetClassName);
            Method method = targetClass.getDeclaredMethod(DESERIALIZATION_METHOD_NAME, int.class, String.class, Any.class);
            method.setAccessible(true);
            Object o = method.invoke(null, version, className, message);
            method.setAccessible(false);
            return o;
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException("Failed to find class " + targetClassName, e);
        } catch (NoSuchMethodException e) {
            throw new IllegalStateException("Failed to find deserialization method " + DESERIALIZATION_METHOD_NAME + "(int, String, com.google.protobuf.Any) on class " + targetClassName, e);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException("Failed to invoke deserialization method " + DESERIALIZATION_METHOD_NAME + "(int, String, com.google.protobuf.Any) on class " + targetClassName, e);
        } catch (InvocationTargetException e) {
            throw new IllegalStateException("The deserialization method for " + DESERIALIZATION_METHOD_NAME + "(int, String, com.google.protobuf.Any) on class " + targetClassName + " threw an exception", e);
        }
    }

    /**
     * Private final constructor for static utility class.
     */
    private ProtoUtil() {}

    
    public static <SERIALIZED_CLASS extends com.google.protobuf.GeneratedMessageV3, 
                   SERIALIZED_DATA extends com.google.protobuf.GeneratedMessageV3, 
                   PROTO_SERIALIZABLE extends ProtoSerializable<SERIALIZED_CLASS>> 
        SERIALIZED_CLASS serialize(PROTO_SERIALIZABLE protoSerializable) throws NoSuchMethodException, SecurityException, IllegalAccessException, IllegalArgumentException, InvocationTargetException {
        
        ProtobufClass annotation = protoSerializable.getClass().getAnnotation(ProtobufClass.class);

        
        
        Class<? extends com.google.protobuf.GeneratedMessageV3> serializedClass = annotation.serializedClass();
        Class<? extends com.google.protobuf.GeneratedMessageV3> serializedData = annotation.serializedData();
        com.google.protobuf.GeneratedMessageV3.Builder<?> serializedClassBuilder = (com.google.protobuf.GeneratedMessageV3.Builder<?>) serializedClass.getMethod("newBuilder").invoke(null);
        com.google.protobuf.GeneratedMessageV3.Builder<?> serializedDataBuilder = (com.google.protobuf.GeneratedMessageV3.Builder<?>) serializedData.getMethod("newBuilder").invoke(null);
        Class<? extends com.google.protobuf.GeneratedMessageV3.Builder> serializedClassBuilderClass = serializedClassBuilder.getClass();
        Class<? extends com.google.protobuf.GeneratedMessageV3.Builder> serializedDataBuilderClass = serializedDataBuilder.getClass();
        
        
        for(Field field : protoSerializable.getClass().getDeclaredFields()) {
            ProtobufField protobufField = field.getAnnotation(ProtobufField.class);
            if(protobufField == null) continue;
            String fieldName = protobufField.name();
            if(fieldName.equals(ProtobufField.DEFAULT_FIELD_NAME)) {
                fieldName = field.getName();
            }
            Method setter = serializedDataBuilderClass.getMethod(setterName(fieldName), Integer.TYPE);
            setter.setAccessible(true);
            field.setAccessible(true);
            Object obj = field.get(protoSerializable);
            setter.invoke(serializedDataBuilder, obj);
        }
        
        serializedClassBuilderClass.getMethod("setVersion", Integer.TYPE).invoke(serializedClassBuilder, 0);
        serializedClassBuilderClass.getMethod("setClassName", String.class).invoke(serializedClassBuilder, protoSerializable.getClass().getName());

        ModHashCodeHasherProto mhchProto = (ModHashCodeHasherProto) serializedDataBuilder.build();

        
        serializedClassBuilderClass.getMethod("setSerializedData", com.google.protobuf.Any.class).invoke(serializedClassBuilder, Any.pack(mhchProto));

        return (SERIALIZED_CLASS) serializedClassBuilder.build();
    }
    
    public static String setterName(String name) {
        StringBuilder sb = new StringBuilder();
        sb.append("set");
        sb.append((""+name.charAt(0)).toUpperCase());
        sb.append(name.substring(1));
        return sb.toString();
    }
}
