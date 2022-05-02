package org.tribuo.util;

import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.lang.reflect.TypeVariable;
import java.util.ArrayList;
import java.util.List;

import org.tribuo.protos.ProtoSerializable;

import com.google.protobuf.Message;

public class ReflectUtil {

    public static <SERIALIZED_CLASS extends Message> Class<SERIALIZED_CLASS> resolveTypeParameter(Class<?> clazz, TypeVariable<?> typeVariable) {
        return resolveTypeParameter(clazz, new MutableTypeVariable(typeVariable));
    }
    
    private static <SERIALIZED_CLASS extends Message> Class<SERIALIZED_CLASS> resolveTypeParameter(Class<?> clazz, MutableTypeVariable typeVariable) {
        System.out.println("resolveTypeParameter("+clazz.getName()+", "+typeVariable.var.getName()+")");
        Class superClass = clazz.getSuperclass();
        //go up the class hierarchy
        if(superClass != null && ProtoSerializable.class.isAssignableFrom(superClass)) {
            Class<SERIALIZED_CLASS> serializedClass = resolveTypeParameter(superClass, typeVariable);
            if(serializedClass != null) {
                return serializedClass;
            }
        }
        //go up the implemented interfaces hierarchy
        for(Class intrface : clazz.getInterfaces()) {
            if(ProtoSerializable.class.isAssignableFrom(intrface) && !ProtoSerializable.class.equals(intrface)) {
                Class<SERIALIZED_CLASS> serializedClass = resolveTypeParameter(intrface, typeVariable);
                if(serializedClass != null) {
                    return serializedClass;
                }
            }
        }

        List<ParameterizedType> pts = getGenericSuperParameterizedTypes(clazz); 

        for(ParameterizedType genericInterface : pts) {
            Type t = getTypeParameter(genericInterface, typeVariable.var);
            if(t instanceof TypeVariable) {
                typeVariable.var = (TypeVariable) t;
            } else if(t instanceof Class) {
                return (Class) t;
            }
        }
        
        return null;
    }

    private static List<ParameterizedType> getGenericSuperParameterizedTypes(Class clazz){
        List<ParameterizedType> pts = new ArrayList<>();
        Type genericSuperclass = clazz.getGenericSuperclass();
        if(genericSuperclass instanceof ParameterizedType) {
            pts.add((ParameterizedType)genericSuperclass);
        }
        for(Type genericInterface : clazz.getGenericInterfaces()) {
            if(genericInterface instanceof ParameterizedType) {
                pts.add((ParameterizedType) genericInterface);
            }
        }
        return pts;
    }
    
    private static Type getTypeParameter(ParameterizedType t, TypeVariable typeVariable) {
       System.out.println("t="+t);
       ParameterizedType pt = (ParameterizedType) t;
       System.out.println("pt="+pt);
        
       Type rawType = pt.getRawType();
       if(rawType instanceof Class) {
           TypeVariable[] typeParameters = ((Class) rawType).getTypeParameters();
           Type[] actualTypeArguments = pt.getActualTypeArguments();
           System.out.println("raw type params="+typeParameters.length);
           for(TypeVariable tp: typeParameters) {
               System.out.println(tp);
           }
           for(Type at : actualTypeArguments) {
               System.out.println("at="+at);
           }
           for(int i=0; i<typeParameters.length; i++) {
               TypeVariable tp = typeParameters[i];
               if(tp.getName().equals(typeVariable.getName())) {
                   Type actualTypeArgument = actualTypeArguments[i];
                   return actualTypeArgument;
               }
           }
        }
        return null;
    }

    private static class MutableTypeVariable{
        private TypeVariable var;

        public MutableTypeVariable(TypeVariable var) {
            this.var = var;
        }
    }
    

}
