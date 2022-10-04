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

import com.google.protobuf.Any;
import com.google.protobuf.Descriptors.FieldDescriptor;
import com.google.protobuf.Message;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.util.MutableDouble;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import com.oracle.labs.mlrg.olcut.util.Pair;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.lang.reflect.TypeVariable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * Utilities for working with Tribuo protobufs.
 */
public final class ProtoUtil {
    private static final Logger logger = Logger.getLogger(ProtoUtil.class.getName());

    /**
     * Used to provide internal Tribuo redirects as classes are redefined.
     * <p>
     * Must only be used for namespaces which are owned by Tribuo modules.
     */
    private static final Map<Pair<Integer, String>, String> REDIRECT_MAP = new HashMap<>();

    /**
     * Private final constructor for static utility class.
     */
    private ProtoUtil() {}

    /**
     * Instantiates the class from the supplied protobuf fields.
     * <p>
     * Deserialization proceeds as follows:
     * <ul>
     *     <li>Check to see if there is a valid redirect for this version and class name tuple.
     *     If there is then the new class name is used for the following steps.</li>
     *     <li>Lookup the class name and instantiate the {@link Class} object.</li>
     *     <li>Find the 3 arg static method {@code deserializeFromProto(int version, String className, com.google.protobuf.Any message)}.</li>
     *     <li>Call the method passing along the original three arguments (note this uses the
     *     original class name even if a redirect has been applied).</li>
     *     <li>Return the freshly constructed object, or rethrow any runtime exceptions.</li>
     * </ul>
     * <p>
     * Throws {@link IllegalStateException} if:
     * <ul>
     *     <li>the requested class could not be found on the classpath/modulepath</li>
     *     <li>the requested class does not have the necessary 3 arg constructor</li>
     *     <li>the deserialization method could not be invoked due to its accessibility, or is in some other way invalid</li>
     *     <li>the deserialization method threw an exception</li>
     * </ul>
     *
     * @param serialized The protobuf to deserialize.
     * @param <SERIALIZED> The protobuf type.
     * @param <PROTO_SERIALIZABLE> The deserialized type.
     * @return The deserialized object.
     */
    public static <SERIALIZED extends Message, PROTO_SERIALIZABLE extends ProtoSerializable<SERIALIZED>> PROTO_SERIALIZABLE deserialize(SERIALIZED serialized) {

        // Extract version from serialized
        FieldDescriptor fieldDescriptor = serialized.getDescriptorForType().findFieldByName("version");
        int version = (Integer) serialized.getField(fieldDescriptor);
        // Extract class_name of return value from serialized
        fieldDescriptor = serialized.getDescriptorForType().findFieldByName("class_name");
        // Allow redirect for Tribuo's classes.
        String className = (String) serialized.getField(fieldDescriptor);
        Pair<Integer, String> key = new Pair<>(version, className);
        String targetClassName = REDIRECT_MAP.getOrDefault(key, className);

        try {
            @SuppressWarnings("unchecked")
            Class<PROTO_SERIALIZABLE> protoSerializableClass = (Class<PROTO_SERIALIZABLE>) Class.forName(targetClassName);
            if (!ProtoSerializable.class.isAssignableFrom(protoSerializableClass)) {
                throw new IllegalStateException("Class " + targetClassName + " does not implement ProtoSerializable");
            }

            fieldDescriptor = serialized.getDescriptorForType().findFieldByName("serialized_data");
            Any serializedData = (Any) serialized.getField(fieldDescriptor);

            Method method = protoSerializableClass.getDeclaredMethod(ProtoSerializable.DESERIALIZATION_METHOD_NAME, int.class, String.class, Any.class);
            Class<?> deserializationReturnType = method.getReturnType();
            if (!ProtoSerializable.class.isAssignableFrom(deserializationReturnType)) {
                throw new IllegalStateException("Method " + protoSerializableClass + "." + ProtoSerializable.DESERIALIZATION_METHOD_NAME + " does not return an instance of " + protoSerializableClass);
            }
            method.setAccessible(true);
            @SuppressWarnings("unchecked")
            PROTO_SERIALIZABLE protoSerializable = (PROTO_SERIALIZABLE) method.invoke(null, version, targetClassName, serializedData);
            method.setAccessible(false);
            return protoSerializable;
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException("Failed to find class " + targetClassName, e);
        } catch (NoSuchMethodException e) {
            throw new IllegalStateException("Failed to find deserialization method " + ProtoSerializable.DESERIALIZATION_METHOD_NAME + "(int, String, com.google.protobuf.Any) on class " + targetClassName, e);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException("Failed to invoke deserialization method " + ProtoSerializable.DESERIALIZATION_METHOD_NAME + "(int, String, com.google.protobuf.Any) on class " + targetClassName, e);
        } catch (InvocationTargetException e) {
            throw new IllegalStateException("The deserialization method for " + ProtoSerializable.DESERIALIZATION_METHOD_NAME + "(int, String, com.google.protobuf.Any) on class " + targetClassName + " threw an exception", e);
        }
    }

    /**
     * Serializes a {@link ProtoSerializable} class which has the appropriate {@link ProtoSerializableClass} annotations.
     * @param protoSerializable The object to serialize.
     * @return The protobuf representation of the input object.
     * @param <SERIALIZED_CLASS> The protobuf class.
     * @param <SERIALIZED_DATA> The serialized data class inside the protobuf.
     * @param <PROTO_SERIALIZABLE> The class of the proto serializable object.
     */
    @SuppressWarnings("unchecked")
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
            //the default value for ProtoSerializableClass.serializedDataClass is Message which is how you can signal that
            //there is no serialized data to be serialized.
            if (serializedDataClass != Message.class) {
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

                        Class<?> fieldClazz = field.getType();
                        String nameStub = (fieldClazz.isArray() || Collection.class.isAssignableFrom(fieldClazz)) ? "addAll" : "set";
                        Method setter = findMethod(serializedDataBuilderClass, nameStub, fieldName, 1);
                        obj = convert(obj);
                        setter.setAccessible(true);
                        setter.invoke(serializedDataBuilder, obj);
                        setter.setAccessible(false);
                    }

                    ProtoSerializableMapField psmf = field.getAnnotation(ProtoSerializableMapField.class);
                    if (psmf != null) {
                        if (!(obj instanceof Map) && (obj != null)) {
                            throw new IllegalStateException("Field of type " + obj.getClass() + " was annotated with ProtoSerializableMapField which is only supported on java.util.Map fields");
                        }
                        String fieldName = psmf.name();
                        if (fieldName.equals(ProtoSerializableField.DEFAULT_FIELD_NAME)) {
                            fieldName = field.getName();
                        }

                        Method setter = findMethod(serializedDataBuilderClass, "put", fieldName, 2);
                        Map<?,?> map = (Map<?,?>) obj;
                        if (map != null) {
                            for (Map.Entry<?,?> e : map.entrySet()) {
                                Object key = convert(e.getKey());
                                Object value = convert(e.getValue());
                                setter.invoke(serializedDataBuilder, key, value);
                            }
                        }
                    }

                    ProtoSerializableKeysValuesField pskvf = field.getAnnotation(ProtoSerializableKeysValuesField.class);
                    if (pskvf != null) {
                        if (!(obj instanceof Map) && (obj != null)) {
                            throw new IllegalStateException("Field of type " + obj.getClass() + " was annotated with ProtoSerializableKeysValuesField which is only supported on java.util.Map fields");
                        }
                        Method keyAdder = findMethod(serializedDataBuilderClass, "add", pskvf.keysName(), 1);
                        keyAdder.setAccessible(true);
                        Method valueAdder = findMethod(serializedDataBuilderClass, "add", pskvf.valuesName(), 1);
                        valueAdder.setAccessible(true);
                        Map<?,?> map = (Map<?,?>) obj;
                        if (map != null) {
                            for (Map.Entry<?,?> e : map.entrySet()) {
                                keyAdder.invoke(serializedDataBuilder, convert(e.getKey()));
                                valueAdder.invoke(serializedDataBuilder, convert(e.getValue()));
                            }
                        }
                        keyAdder.setAccessible(false);
                        valueAdder.setAccessible(false);
                    }

                    ProtoSerializableMapValuesField psmvf = field.getAnnotation(ProtoSerializableMapValuesField.class);
                    if (psmvf != null) {
                        if (!(obj instanceof Map) && (obj != null)) {
                            throw new IllegalStateException("Field of type " + obj.getClass() + " was annotated with ProtoSerializableMapValuesField which is only supported on java.util.Map fields");
                        }
                        Method valuesAdder = findMethod(serializedDataBuilderClass, "addAll", psmvf.valuesName(), 1);
                        obj = convertValues((Map<?,?>) obj);
                        valuesAdder.setAccessible(true);
                        valuesAdder.invoke(serializedDataBuilder, obj);
                        valuesAdder.setAccessible(false);
                    }

                    field.setAccessible(false);
                }
                serializedClassBuilderClass.getMethod("setSerializedData", com.google.protobuf.Any.class).invoke(serializedClassBuilder, Any.pack(serializedDataBuilder.build()));
            }
            return (SERIALIZED_CLASS) serializedClassBuilder.build();
        } catch (InvocationTargetException | IllegalAccessException | NoSuchMethodException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * If this class is annotated with {@link ProtoSerializableClass} returns the
     * version number, otherwise returns -1.
     *
     * @param clazz The class to check.
     * @return The version number, or -1 if it does not use the annotation.
     */
    public static int getCurrentVersion(Class<? extends ProtoSerializable<?>> clazz) {
        ProtoSerializableClass ann = clazz.getAnnotation(ProtoSerializableClass.class);
        if (ann != null) {
            return ann.version();
        } else {
            return -1;
        }
    }

    static <SERIALIZED_CLASS extends Message, PROTO_SERIALIZABLE extends ProtoSerializable<SERIALIZED_CLASS>> Class<SERIALIZED_CLASS> getSerializedClass(PROTO_SERIALIZABLE protoSerializable) {
        @SuppressWarnings("unchecked")
        Class<SERIALIZED_CLASS> serializedClass = (Class<SERIALIZED_CLASS>) resolveTypeParameter(ProtoSerializable.class, protoSerializable.getClass(), ProtoSerializable.class.getTypeParameters()[0]);
        if (serializedClass != null) {
            return serializedClass;
        }
        String tpName = ProtoSerializable.class.getTypeParameters()[0].getName();
        throw new IllegalArgumentException("unable to resolve type parameter '" + tpName + "' in ProtoSerializable<" + tpName + "> for class " + protoSerializable.getClass().getName());
    }

    private static List<Object> convertValues(Map<?, ?> obj) {
        List<Object> values = new ArrayList<>();
        for (Object value : obj.values()) {
            values.add(convert(value));
        }
        return values;
    }

    private static Object convert(Object obj) {
        if (obj instanceof ProtoSerializable) {
            return ((ProtoSerializable<?>) obj).serialize();
        } else if (obj instanceof ObjectProvenance) {
            return ProtoSerializable.PROVENANCE_SERIALIZER.serializeToProto(
                    ProvenanceUtil.marshalProvenance((ObjectProvenance) obj));
        } else if (obj instanceof MutableLong) {
            return ((MutableLong) obj).longValue();
        } else if (obj instanceof MutableDouble) {
            return ((MutableDouble) obj).doubleValue();
        } else if (obj.getClass().isEnum()) {
            return ((Enum<?>) obj).name();
        } else if (obj instanceof List) {
            List<?> list = (List<?>) obj;
            return list.stream().map(ProtoUtil::convert).collect(Collectors.toList());
        } else if (obj instanceof int[]) {
            int[] t = (int[]) obj;
            return Arrays.stream(t).boxed().collect(Collectors.toList());
        } else if (obj instanceof long[]) {
            long[] t = (long[]) obj;
            return Arrays.stream(t).boxed().collect(Collectors.toList());
        } else if (obj instanceof float[]) {
            float[] t = (float[]) obj;
            List<Float> floats = new ArrayList<>();
            for (float f : t) {
                floats.add(f);
            }
            return floats;
        } else if (obj instanceof double[]) {
            double[] t = (double[]) obj;
            return Arrays.stream(t).boxed().collect(Collectors.toList());
        } else if (obj instanceof String[]) {
            return Arrays.asList((String[]) obj);
        } else if (obj instanceof Double || obj instanceof Float || obj instanceof Byte || obj instanceof Short || obj instanceof Character || obj instanceof Integer || obj instanceof Long || obj instanceof Boolean || obj instanceof String) {
            return obj;
        } else {
            throw new IllegalArgumentException("Invalid protobuf field type " + obj.getClass());
        }
    }

    private static List<Field> getFields(Class<?> clazz) {
        Set<String> fieldNameSet = new HashSet<>();
        List<Field> fields = new ArrayList<>();
        getFields(clazz, fieldNameSet, fields);
        return fields;
    }

    private static void getFields(Class<?> clazz, Set<String> fieldNameSet, List<Field> fields) {
        for (Field field : clazz.getDeclaredFields()) {
            ProtoSerializableField psf = field.getAnnotation(ProtoSerializableField.class);
            ProtoSerializableMapField psmf = field.getAnnotation(ProtoSerializableMapField.class);
            ProtoSerializableKeysValuesField pskvf = field.getAnnotation(ProtoSerializableKeysValuesField.class);
            ProtoSerializableMapValuesField psmvf = field.getAnnotation(ProtoSerializableMapValuesField.class);
            if ((psf != null) && (psmf == null) && (pskvf == null) && (psmvf == null)) {
                String protoFieldName;
                if (psf.name().equals(ProtoSerializableField.DEFAULT_FIELD_NAME)) {
                    protoFieldName = field.getName();
                } else {
                    protoFieldName = psf.name();
                }
                if (!fieldNameSet.contains(protoFieldName)) {
                    fields.add(field);
                    fieldNameSet.add(field.getName());
                } else {
                    logger.warning(
                        "Duplicate field named " + protoFieldName + " detected in class " + clazz);
                }
            } else if ((psf == null) && (psmf != null) && (pskvf == null) && (psmvf == null)) {
                String protoFieldName;
                if (psmf.name().equals(ProtoSerializableField.DEFAULT_FIELD_NAME)) {
                    protoFieldName = field.getName();
                } else {
                    protoFieldName = psmf.name();
                }
                if (!fieldNameSet.contains(protoFieldName)) {
                    fields.add(field);
                    fieldNameSet.add(field.getName());
                } else {
                    logger.warning("Duplicate field named " + protoFieldName + " detected in class " + clazz);
                }
            } else if ((psf == null) && (psmf == null) && (pskvf != null) && (psmvf == null)) {
                String keyName = pskvf.keysName();
                String valueName = pskvf.valuesName();
                if (fieldNameSet.contains(keyName) && fieldNameSet.contains(valueName)) {
                    continue;
                }
                if (fieldNameSet.contains(keyName) || fieldNameSet.contains(valueName)) {
                    throw new IllegalStateException("ProtoSerializableKeysValuesField on " + clazz.getName() + "." + field.getName() + " collides with another ProtoSerializable annotation");
                }
                fields.add(field);
                fieldNameSet.add(keyName);
                fieldNameSet.add(valueName);
            } else if ((psf == null) && (psmf == null) && (pskvf == null) && (psmvf != null)) {
                String valuesName = psmvf.valuesName();
                if (!fieldNameSet.contains(valuesName)) {
                    fields.add(field);
                    fieldNameSet.add(valuesName);
                }
            } else if ((psf != null) || (psmf != null) || (pskvf != null) || (psmvf != null)) {
                String message = "Multiple ProtoSerializable annotations applied to the same field, found "
                        + "ProtoSerializableField = " + psf
                        + ", ProtoSerializableMapField = " + psmf
                        + ", ProtoSerializableKeysValuesField = " + pskvf
                        + ", ProtoSerializableMapValuesField = " + psmvf
                        + " on field named " + field.getName() + " of class " + clazz;
                throw new IllegalStateException(message);
            }
        }

        Class<?> superclass = clazz.getSuperclass();
        if (superclass != null && !superclass.equals(Object.class)) {
            getFields(superclass, fieldNameSet, fields);
        }
    }

    private static Method findMethod(Class<?> serializedDataBuilderClass, String prefixName, String fieldName, int expectedParamCount) {
        String methodName = generateMethodName(prefixName, fieldName);

        for (Method method : serializedDataBuilderClass.getMethods()) {
            if (method.getName().equals(methodName)) {
                if (method.getParameterTypes().length != expectedParamCount) {
                    continue;
                }
                if (expectedParamCount == 0) {
                    return method;
                }
                Class<?> clazz = method.getParameterTypes()[0];
                if (Message.Builder.class.isAssignableFrom(clazz)) {
                    continue;
                }
                return method;
            }
        }
        throw new IllegalArgumentException("Unable to find method " + methodName + " for field name: " + fieldName + " in class: "
                + serializedDataBuilderClass.getName());
    }

    /**
     * Generate the method name for the type from the protobuf.
     * @param prefix The prefix, usually "addAll", "set", "put" etc.
     * @param name The field name.
     * @return The method name.
     */
    public static String generateMethodName(String prefix, String name) {
        StringBuilder sb = new StringBuilder();
        sb.append(prefix);
        sb.append(("" + name.charAt(0)).toUpperCase());
        sb.append(name.substring(1));
        return sb.toString();
    }

    /**
     * @param <I>          the type of the generic interface
     * @param <C>          the type of the subclass we are trying to resolve the parameter for
     * @param intrface     a generically typed interface/class that has a type parameter that we want to resolve
     * @param clazz        a subclass of intrface that we to resolve the type parameter for (if possible)
     * @param typeVariable the type variable/parameter that we want resolved (e.g. the type variable corresponding to &lt;T&gt;)
     * @return null if the type parameter has not been resolved, otherwise the class type of the type parameter for the provided class.
     */
    private static <I, C extends I> Class<?> resolveTypeParameter(Class<I> intrface, Class<C> clazz, TypeVariable<?> typeVariable) {
        return resolveTypeParameter(intrface, clazz, new MutableTypeVariable(typeVariable));
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static <I, C extends I> Class<?> resolveTypeParameter(Class<I> intrface, Class<C> clazz, MutableTypeVariable typeVariable) {
        //go up the class hierarchy until super class is no longer assignable to the interface
        Class superClass = clazz.getSuperclass();
        if (superClass != null && intrface.isAssignableFrom(superClass)) {
            Class<?> serializedClass = resolveTypeParameter(intrface, superClass, typeVariable);
            if (serializedClass != null) {
                return serializedClass;
            }
        }
        //go up the interfaces hierarchy - but only those assignable to the interface
        for (Class iface : clazz.getInterfaces()) {
            if (intrface.isAssignableFrom(iface) && !intrface.equals(iface)) {
                Class<?> serializedClass = resolveTypeParameter(intrface, iface, typeVariable);
                if (serializedClass != null) {
                    return serializedClass;
                }
            }
        }

        //get all the generic supertypes that are parameterized types (but only those
        //assignable to the interface)
        List<ParameterizedType> pts = getGenericSuperParameterizedTypes(intrface, clazz);

        //loop over each and get the type of the type variable for the typed generic interface
        for (ParameterizedType genericInterface : pts) {
            Type t = getParameterType(genericInterface, typeVariable.var);
            //if the resulting type is another type variable, then we need to update the 
            //type variable that we are trying to match against
            if (t instanceof TypeVariable) {
                typeVariable.var = (TypeVariable) t;
            }
            //otherwise, we are done and we can return the class type of the type variable.
            else if (t instanceof Class) {
                return (Class<?>) t;
            }
        }

        return null;
    }

    /**
     * Puts the generic superclass and generic interfaces into a single list
     * to make it easy to iterate over each in a single collection.  All
     * returned values will be of type ParameterizedType and assignable to the
     * intface (i.e. subclass sub-interface).
     *
     * @param clazz the class definition to find the  generic parameterized types for.
     * @return A list of the parameterized types.
     */
    private static List<ParameterizedType> getGenericSuperParameterizedTypes(Class<?> intrface, Class<?> clazz) {
        List<ParameterizedType> pts = new ArrayList<>();
        Type genericSuperclass = clazz.getGenericSuperclass();
        if (genericSuperclass instanceof ParameterizedType) {
            pts.add((ParameterizedType) genericSuperclass);
        }
        for (Type genericInterface : clazz.getGenericInterfaces()) {
            if (genericInterface instanceof ParameterizedType) {
                ParameterizedType pt = (ParameterizedType) genericInterface;
                if (intrface.isAssignableFrom((Class<?>) pt.getRawType())) {
                    pts.add((ParameterizedType) genericInterface);
                }
            }
        }
        return pts;
    }

    /**
     * a class or type variable corresponding to the parameterized type's parameter
     * type for the given type variable The returned type will either be a class if
     * specified or another type variable if not.
     *
     * @param parameterizedType either a generic superclass or a generic interface
     * @param typeVariable      a type variable corresponding to e.g. &lt;T&gt;
     * @return a class or type variable corresponding to the parameterized type's
     * parameter type for the given type variable
     */
    @SuppressWarnings("rawtypes")
    private static Type getParameterType(ParameterizedType parameterizedType, TypeVariable<?> typeVariable) {
        Type rawType = parameterizedType.getRawType();
        if (rawType instanceof Class) {
            TypeVariable<?>[] typeParameters = ((Class<?>) rawType).getTypeParameters();
            Type[] actualTypeArguments = parameterizedType.getActualTypeArguments();
            if (typeParameters.length == actualTypeArguments.length) {
                for (int i = 0; i < typeParameters.length; i++) {
                    TypeVariable<?> tp = typeParameters[i];
                    if (tp.getName().equals(typeVariable.getName())) {
                        Type actualTypeArgument = actualTypeArguments[i];
                        return actualTypeArgument;
                    }
                }
            }
        }
        return null;
    }

    private static final class MutableTypeVariable {
        private TypeVariable<?> var;

        MutableTypeVariable(TypeVariable<?> var) {
            this.var = var;
        }
    }

}
