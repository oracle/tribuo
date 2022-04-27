package org.tribuo.util;

import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.lang.reflect.TypeVariable;
import java.util.ArrayList;
import java.util.List;

public class ReflectUtil {

    public static <I, C extends I> List<Class> getLineage(Class<I> intface, Class<C> clazz){
        if(!intface.isAssignableFrom(clazz)) {
            throw new RuntimeException(""+intface.getName()+" is not assignable from "+clazz.getName());
        }
        List<Class> lineage = new ArrayList<>();
        lineage.add(clazz);
        return getLineage(intface, clazz, lineage);
    }
    
    private static <I, C extends I> List<Class> getLineage(Class<I> intface, Class<C> clazz, List<Class> lineage) {
        if(clazz.equals(intface)) {
            return lineage;
        }
        Class superClass = clazz.getSuperclass();
        if(superClass != null && intface.isAssignableFrom(superClass)) {
            lineage.add(superClass);
            return getLineage(intface, superClass, lineage);
        }
        for(Class intClass : clazz.getInterfaces()) {
            if(intface.isAssignableFrom(intClass)) {
                lineage.add(intClass);
                return getLineage(intface, intClass, lineage);
            }
        }
        throw new RuntimeException("unable to create lineage from class="+clazz.getName()+" to ancestor="+intface.getName());
    }

    public static <I, C extends I> List<Class<?>> getTypeParameterTypes(Class<I> intface, Class<C> clazz) {
        TypeVariable[] typeVariables = intface.getTypeParameters();
        List<Object> typeParameterTypes = new ArrayList<>();
        for(TypeVariable typeVariable : typeVariables) {
            typeParameterTypes.add(typeVariable.getName());
        }
        
        List<Class> lineage = getLineage(intface, clazz);
        
        for(int i=lineage.size()-2; i>=0; i--) {
            Class cls = lineage.get(i);

            List<String> typeVariableNames = new ArrayList<>();
            for(TypeVariable tv : cls.getTypeParameters()) {
                typeVariableNames.add(tv.getName());
            }

            for(Type genericInterface : cls.getGenericInterfaces()) {
                if(genericInterface instanceof ParameterizedType) {
                    ParameterizedType pt = (ParameterizedType) genericInterface;
                    if(lineage.get(i+1).equals(pt.getRawType())) {
                        int j=0;
                        for(Type at : pt.getActualTypeArguments()) {
                            Object tpt = typeParameterTypes.get(j);
                            while(tpt instanceof Class) {
                                tpt = typeParameterTypes.get(++j);
                            }
                            if(at instanceof Class) {
                                typeParameterTypes.set(j, at);
                            }
                            if(at instanceof TypeVariable) {
                                typeParameterTypes.set(j, ((TypeVariable)at).getName());
                            }
                        }
                    }
                }
            }

            Type t = cls.getGenericSuperclass();
            if(t instanceof ParameterizedType) {
                ParameterizedType pt = (ParameterizedType) t;
                if(lineage.get(i+1).equals(pt.getRawType())) {
                    int j=0;
                    for(Type at : pt.getActualTypeArguments()) {
                        Object tpt = typeParameterTypes.get(j);
                        while(tpt instanceof Class) {
                            tpt = typeParameterTypes.get(++j);
                        }
                        if(at instanceof Class) {
                            typeParameterTypes.set(j, at);
                        }
                        if(at instanceof TypeVariable) {
                            typeParameterTypes.set(j, ((TypeVariable)at).getName());
                        }
                    }
                }
            }
            
        }
        
        List<Class<?>> returnValues = new ArrayList<>();
        for(Object obj : typeParameterTypes) {
            if(obj instanceof Class) {
                returnValues.add((Class<?>)obj);
            } else {
                throw new RuntimeException("type parameter unresolved: "+obj);
            }
        }
        
        return returnValues;
    }
}
