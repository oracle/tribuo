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

package org.tribuo.protos;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.tribuo.util.ReflectUtil;

import com.google.protobuf.Any;
import com.google.protobuf.Descriptors.FieldDescriptor;
import com.google.protobuf.GeneratedMessageV3;
import com.google.protobuf.Message;
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.util.MutableLong;

/**
 * Utilities for working with Tribuo protobufs.
 */
public final class ProtoUtil {

    public static final String DESERIALIZATION_METHOD_NAME = "deserializeFromProto";

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

    public static <SERIALIZED extends Message, PROTO_SERIALIZABLE extends ProtoSerializable<SERIALIZED>> PROTO_SERIALIZABLE deserialize(SERIALIZED serialized) {

        try {
            //extract version from serialized
            FieldDescriptor fieldDescriptor = serialized.getDescriptorForType().findFieldByName("version");
            int version = ((Integer) serialized.getField(fieldDescriptor)).intValue();
            //extract class_name of return value from serialized
            fieldDescriptor = serialized.getDescriptorForType().findFieldByName("class_name");
            String className = (String) serialized.getField(fieldDescriptor);
            @SuppressWarnings("unchecked")
            Class<PROTO_SERIALIZABLE> protoSerializableClass = (Class<PROTO_SERIALIZABLE>) Class.forName(className);

            fieldDescriptor = serialized.getDescriptorForType().findFieldByName("serialized_data");
            Any serializedData = (Any) serialized.getField(fieldDescriptor);

            try {
                Method method = protoSerializableClass.getDeclaredMethod(DESERIALIZATION_METHOD_NAME, int.class, String.class, Any.class);
                method.setAccessible(true);
                @SuppressWarnings("unchecked")
                PROTO_SERIALIZABLE protoSerializable = (PROTO_SERIALIZABLE) method.invoke(null, version, className, serializedData);
                method.setAccessible(false);
                return protoSerializable;
            } catch (NoSuchMethodException nsme) {
                return ProtoUtil.deserialize(version, className, serializedData);
            }
        } catch (ClassNotFoundException | IllegalAccessException | IllegalArgumentException | InvocationTargetException
                | SecurityException e) {
            throw new RuntimeException(e);
        }
    }
    
    public static <SERIALIZED extends Message, SERIALIZED_DATA extends Message, PROTO_SERIALIZABLE extends ProtoSerializable<SERIALIZED>> PROTO_SERIALIZABLE deserialize(
            int version, String className, Any serializedData) {
        try {
            System.out.println("ProtoUtil.deserialize");

            @SuppressWarnings("unchecked")
            Class<PROTO_SERIALIZABLE> protoSerializableClass = (Class<PROTO_SERIALIZABLE>) Class.forName(className);

            //initialize return value
            Constructor<PROTO_SERIALIZABLE> declaredConstructor = protoSerializableClass.getDeclaredConstructor();
            declaredConstructor.setAccessible(true);
            PROTO_SERIALIZABLE protoSerializable = declaredConstructor.newInstance();

            //get @ProtobuffClass annotation from class definition of serialized ProtoSerializable
            ProtoSerializableClass protobufClassAnnotation = protoSerializableClass.getAnnotation(ProtoSerializableClass.class);
            Class<SERIALIZED> serializedClass = getSerializedClass(protoSerializable);
            @SuppressWarnings("unchecked")
            Class<SERIALIZED_DATA> serializedDataClass = (Class<SERIALIZED_DATA>) protobufClassAnnotation.serializedDataClass();

            System.out.println("serialized_class: " + serializedClass.getName());
            System.out.println("version: " + version);
            System.out.println("class_name: " + className);

            //e.g. HashCodeHasher has no serializable data so exit early
            if (serializedData.getValue().size() == 0) {
                return protoSerializable;
            }
            
            SERIALIZED_DATA proto = serializedData.unpack(serializedDataClass);

            System.out.println("serialized_data: " + proto.getClass().getName());
            System.out.println("protoSerializable: " + protoSerializable.getClass().getName());

            for (Field field : getFields(protoSerializableClass)) {
                System.out.println("field: " + field.getName());
                ProtoSerializableField protobufField = field.getAnnotation(ProtoSerializableField.class);
                String fieldName = protobufField.name();
                if (fieldName.equals(ProtoSerializableField.DEFAULT_FIELD_NAME)) {
                    fieldName = field.getName();
                    System.out.println("field: " + field.getName());
                }
                field.setAccessible(true);

                Method getter = findMethod(serializedDataClass, "get", fieldName, 0);
                Object obj = getter.invoke(proto);
                if (obj instanceof GeneratedMessageV3) {
                    System.out.println("calling nested deserialize for " + fieldName);
                    obj = deserialize((GeneratedMessageV3) obj);
                }
                System.out.println("obj = " + obj.getClass().getName());
                field.set(protoSerializable, obj);
            }

            if (protoSerializable instanceof Configurable) {
                ((Configurable) protoSerializable).postConfig();
            }
            return protoSerializable;

        } catch (ClassNotFoundException | PropertyException | IOException | IllegalArgumentException | IllegalAccessException | InvocationTargetException | InstantiationException | NoSuchMethodException | SecurityException e) {
            throw new RuntimeException(e);
        }
    }
    
    /**
     * Private final constructor for static utility class.
     */
    private ProtoUtil() {}

    @SuppressWarnings({ "rawtypes", "unchecked" })
    public static <SERIALIZED_CLASS extends Message, SERIALIZED_DATA extends Message, PROTO_SERIALIZABLE extends ProtoSerializable<SERIALIZED_CLASS>> SERIALIZED_CLASS serialize(PROTO_SERIALIZABLE protoSerializable) {
        try {

            ProtoSerializableClass annotation = protoSerializable.getClass().getAnnotation(ProtoSerializableClass.class);
            if (annotation == null) {
                throw new IllegalArgumentException("instance of ProtoSerializable must be annotated with @ProtoSerializableClass to be serialized with ProtoUtil.serialize()");
            }

            Class<SERIALIZED_CLASS> serializedClass = getSerializedClass(protoSerializable);
            SERIALIZED_CLASS.Builder serializedClassBuilder = (SERIALIZED_CLASS.Builder) serializedClass.getMethod("newBuilder").invoke(null);
            Class<SERIALIZED_CLASS.Builder> serializedClassBuilderClass = (Class<SERIALIZED_CLASS.Builder>) serializedClassBuilder.getClass();
            serializedClassBuilderClass.getMethod("setVersion", Integer.TYPE).invoke(serializedClassBuilder, annotation.version());
            serializedClassBuilderClass.getMethod("setClassName", String.class).invoke(serializedClassBuilder, protoSerializable.getClass().getName());

            Class<SERIALIZED_DATA> serializedDataClass = (Class<SERIALIZED_DATA>) annotation.serializedDataClass();
            //the default value for ProtoSerializableClass.serializedDataClass is GeneratedMessageV3 which is how you can signal that
            //there is no serialized data to be serialized.
            if (serializedDataClass != GeneratedMessageV3.class) {
                SERIALIZED_DATA.Builder serializedDataBuilder = (SERIALIZED_DATA.Builder) serializedDataClass.getMethod("newBuilder").invoke(null);
                Class<SERIALIZED_DATA.Builder> serializedDataBuilderClass = (Class<SERIALIZED_DATA.Builder>) serializedDataBuilder.getClass();

                for (Field field : getFields(protoSerializable.getClass())) {
                    field.setAccessible(true);
                    Object obj = field.get(protoSerializable);
                    
                    ProtoSerializableField protoSerializableField = field.getAnnotation(ProtoSerializableField.class);
                    if (protoSerializableField != null) {
                        String fieldName = protoSerializableField.name();
                        if (fieldName.equals(ProtoSerializableField.DEFAULT_FIELD_NAME)) {
                            fieldName = field.getName();
                        }
                        
                        Method setter = findMethod(serializedDataBuilderClass, "set", fieldName, 1);
                        obj = convert(obj);
                        setter.setAccessible(true);
                        setter.invoke(serializedDataBuilder, obj);
                    }

                    ProtoSerializableArrayField psaf = field.getAnnotation(ProtoSerializableArrayField.class);
                    if (psaf != null) {
                        String fieldName = psaf.name();
                        if (fieldName.equals(ProtoSerializableArrayField.DEFAULT_FIELD_NAME)) {
                            fieldName = field.getName();
                        }
                        
                        Method setter = findMethod(serializedDataBuilderClass, "addAll", fieldName, 1);
                        obj = toList(obj);
                        setter.setAccessible(true);
                        setter.invoke(serializedDataBuilder, obj);
                    }

                    ProtoSerializableKeysValuesField pskvf = field.getAnnotation(ProtoSerializableKeysValuesField.class);
                    if (pskvf != null) {
                        Method keyAdder = findMethod(serializedDataBuilderClass, "add", pskvf.keysName(), 1);
                        keyAdder.setAccessible(true);
                        Method valueAdder = findMethod(serializedDataBuilderClass, "add", pskvf.valuesName(), 1);
                        valueAdder.setAccessible(true);
                        Map map = (Map) obj;
                        if(map != null) {
                            Set<Map.Entry> entrySet = map.entrySet();
                            for (Map.Entry e : entrySet) {
                                keyAdder.invoke(serializedDataBuilder, convert(e.getKey()));
                                valueAdder.invoke(serializedDataBuilder, convert(e.getValue()));
                            }
                        }
                    }
                    
                    ProtoSerializableMapValuesField psmvf = field.getAnnotation(ProtoSerializableMapValuesField.class);
                    if (psmvf != null) {
                        Method valuesAdder = findMethod(serializedDataBuilderClass, "addAll", psmvf.valuesName(), 1);
                        valuesAdder.setAccessible(true);
                        obj = toList((Map) obj);
                        valuesAdder.setAccessible(true);
                        valuesAdder.invoke(serializedDataBuilder, obj);
                    }
                }
                serializedClassBuilderClass.getMethod("setSerializedData", com.google.protobuf.Any.class).invoke(serializedClassBuilder, Any.pack(serializedDataBuilder.build()));
            }
            return (SERIALIZED_CLASS) serializedClassBuilder.build();
        } catch (InvocationTargetException | IllegalAccessException | IllegalArgumentException | NoSuchMethodException
                | SecurityException e) {
            throw new RuntimeException(e);
        }
    }

    private static Object toList(Object obj) {
        if(obj instanceof double[]) {
            List<Double> doubles = new ArrayList<>();
            for(double db : (double[])obj) {
                doubles.add(db);
            }
            return doubles;
        }
        throw new RuntimeException("unable to convert "+obj+" to list");
    }

    @SuppressWarnings("unchecked")
    private static <SERIALIZED_CLASS extends Message,PROTO_SERIALIZABLE extends ProtoSerializable<SERIALIZED_CLASS>> Class<SERIALIZED_CLASS> getSerializedClass(PROTO_SERIALIZABLE protoSerializable) {
        List<Class<?>> typeParameterTypes = ReflectUtil.getTypeParameterTypes(ProtoSerializable.class, protoSerializable.getClass());
        return (Class<SERIALIZED_CLASS>) typeParameterTypes.get(0);
    }

    @SuppressWarnings({ "unchecked", "rawtypes" })
    private static List toList(Map obj) {
        List values = new ArrayList();
        for(Object value : obj.values()) {
            values.add(convert(value));
        }
        return values;
    }

    @SuppressWarnings("rawtypes")
    private static Object convert(Object obj) {
        if (obj instanceof ProtoSerializable) {
            return ((ProtoSerializable) obj).serialize();
        }
        if (obj instanceof MutableLong) {
            return ((MutableLong) obj).longValue();
        }
        if (obj.getClass().isEnum()) {
            return ((Enum) obj).name();
        }
        return obj;
    }

    private static List<Field> getFields(Class<?> class1) {
        Set<String> fieldNameSet = new HashSet<>();
        List<Field> fields = new ArrayList<>();    
        _getFields(class1, fieldNameSet, fields);
        return fields;
    }
    
    private static void _getFields(Class<?> class1, Set<String> fieldNameSet, List<Field> fields) {
        for (Field field : class1.getDeclaredFields()) {
            String protoFieldName = null;
            ProtoSerializableField psf = field.getAnnotation(ProtoSerializableField.class);
            if(psf != null) {
                protoFieldName = psf.name();
                if (protoFieldName.equals(ProtoSerializableField.DEFAULT_FIELD_NAME)) {
                    protoFieldName = field.getName();
                }
                if (fieldNameSet.contains(protoFieldName))
                    continue;
                fields.add(field);
                fieldNameSet.add(field.getName());
                continue;
            }

            ProtoSerializableArrayField psaf = field.getAnnotation(ProtoSerializableArrayField.class);
            if (psaf !=null) {
                protoFieldName = psaf.name();
                if (protoFieldName.equals(ProtoSerializableField.DEFAULT_FIELD_NAME)) {
                    protoFieldName = field.getName();
                }
                if (fieldNameSet.contains(protoFieldName))
                    continue;
                fields.add(field);
                fieldNameSet.add(field.getName());
                continue;
            }
            
            
            ProtoSerializableKeysValuesField pskvf = field.getAnnotation(ProtoSerializableKeysValuesField.class);
            if (pskvf !=null) {
                String keyName = pskvf.keysName();
                String valueName = pskvf.valuesName();
                if(fieldNameSet.contains(keyName) && fieldNameSet.contains(valueName)) {
                    continue;
                }
                if(fieldNameSet.contains(keyName) || fieldNameSet.contains(valueName)) {
                    throw new RuntimeException("ProtoSerializableKeysValuesField on "+class1.getName()+"."+field.getName()+" collides with another protoserializable annotation");
                }
                fields.add(field);
                fieldNameSet.add(keyName);
                fieldNameSet.add(valueName);
                continue;
            }
            
            ProtoSerializableMapValuesField psmvf = field.getAnnotation(ProtoSerializableMapValuesField.class);
            if (psmvf !=null) {
                String valuesName = psmvf.valuesName();
                if(fieldNameSet.contains(valuesName)) {
                    continue;
                }
                fields.add(field);
                fieldNameSet.add(valuesName);
                continue;
            }

        }

        Class<?> superclass = class1.getSuperclass();
        if(superclass != null && !superclass.equals(Object.class)) {
            _getFields(superclass, fieldNameSet, fields);
        }
    }

    private static Method findMethod(Class<?> serializedDataBuilderClass, String prefixName, String fieldName, int expectedParamCount) {
        String methodName = generateMethodName(prefixName, fieldName);

        for (Method method : serializedDataBuilderClass.getMethods()) {
            if (method.getName().equals(methodName)) {
                if(method.getParameterTypes().length != expectedParamCount) {
                    continue;
                }
                if(expectedParamCount == 0) {
                    return method;
                }
                Class<?> class1 = method.getParameterTypes()[0];
                if(com.google.protobuf.GeneratedMessageV3.Builder.class.isAssignableFrom(class1)) {
                    continue;
                }
                return method;
            }
        }
        throw new IllegalArgumentException("unable to find method "+methodName+" for field name: " + fieldName + " in class: "
                + serializedDataBuilderClass.getName());
    }

    public static String generateMethodName(String prefix, String name) {
        StringBuilder sb = new StringBuilder();
        sb.append(prefix);
        sb.append(("" + name.charAt(0)).toUpperCase());
        sb.append(name.substring(1));
        return sb.toString();
    }
    
    
    
}
