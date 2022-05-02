package org.tribuo.util;

import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.lang.reflect.TypeVariable;
import java.util.ArrayList;
import java.util.List;

public class ReflectUtil {

    /**
     * @param <I> the type of the generic interface
     * @param <C> the type of the subclass we are trying to resolve the parameter for
     * @param intrface a generically typed interface/class that has a type parameter that we want to resolve
     * @param clazz a subclass of intrface that we to resolve the type parameter for (if possible)
     * @param typeVariable the type variable/parameter that we want resolved (e.g. the type variable corresponding to &lt;T&gt;)
     * @return null if the type parameter has not been resolved, otherwise the class type of the type parameter for the provided class.
     */
    public static <I, C extends I> Class<?> resolveTypeParameter(Class<I> intrface, Class<C> clazz, TypeVariable<?> typeVariable) {
        return resolveTypeParameter(intrface, clazz, new MutableTypeVariable(typeVariable));
    }
    
    @SuppressWarnings({ "unchecked", "rawtypes" })
    private static <I, C extends I> Class<?> resolveTypeParameter(Class<I> intrface, Class<C> clazz, MutableTypeVariable typeVariable) {
        //go up the class hierarchy until super class is no longer a assignable to the interface
        Class superClass = clazz.getSuperclass();
        if(superClass != null && intrface.isAssignableFrom(superClass)) {
            Class<?> serializedClass = resolveTypeParameter(intrface, superClass, typeVariable);
            if(serializedClass != null) {
                return serializedClass;
            }
        }
        //go up the interfaces hierarchy - but only those assignable to the interface
        for(Class iface : clazz.getInterfaces()) {
            if(intrface.isAssignableFrom(iface) && !intrface.equals(iface)) {
                Class<?> serializedClass = resolveTypeParameter(intrface, iface, typeVariable);
                if(serializedClass != null) {
                    return serializedClass;
                }
            }
        }

        //get all the generic supertypes that are parameterized types (but only those
        //assignable to the interface)
        List<ParameterizedType> pts = getGenericSuperParameterizedTypes(intrface, clazz); 

        //loop over each and get the type of the type variable for the typed generic interface
        for(ParameterizedType genericInterface : pts) {
            Type t = getParameterType(genericInterface, typeVariable.var);
            //if the resulting type is another type variable, then we need to update the 
            //type variable that we are trying to match against
            if(t instanceof TypeVariable) {
                typeVariable.var = (TypeVariable) t;
            }
            //otherwise, we are done and we can return the class typ of the type variable.
            else if(t instanceof Class) {
                return (Class) t;
            }
        }
        
        return null;
    }

    /**
     * Puts the generic superclass and generic interfaces into a single list
     * to make it easy to iterate over each in a single collection.  All 
     * returned values will be of type ParameterizedType and assignable to the
     * intface (i.e. subclass sub-interface).
     * @param clazz the class definition to find the  generic parameterized types for.
     * @return
     */
    @SuppressWarnings({ "rawtypes", "unchecked" })
    private static List<ParameterizedType> getGenericSuperParameterizedTypes(Class intrface, Class clazz){
        List<ParameterizedType> pts = new ArrayList<>();
        Type genericSuperclass = clazz.getGenericSuperclass();
        if(genericSuperclass instanceof ParameterizedType) {
            pts.add((ParameterizedType)genericSuperclass);
        }
        for(Type genericInterface : clazz.getGenericInterfaces()) {
            if(genericInterface instanceof ParameterizedType) {
                ParameterizedType pt = (ParameterizedType) genericInterface;
                if(intrface.isAssignableFrom((Class)pt.getRawType())) {
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
     * @param parameterizedType - either a generic superclass or a generic interface
     * @param typeVariable      - a type variable corresponding to e.g. &lt;T&gt;
     * @return a class or type variable corresponding to the parameterized type's
     *         parameter type for the given type variable
     */
    @SuppressWarnings("rawtypes")
    private static Type getParameterType(ParameterizedType parameterizedType, TypeVariable typeVariable) {
       Type rawType = parameterizedType.getRawType();
       if(rawType instanceof Class) {
           TypeVariable[] typeParameters = ((Class) rawType).getTypeParameters();
           Type[] actualTypeArguments = parameterizedType.getActualTypeArguments();
           if(typeParameters.length == actualTypeArguments.length) {
               for(int i=0; i<typeParameters.length; i++) {
                   TypeVariable tp = typeParameters[i];
                   if(tp.getName().equals(typeVariable.getName())) {
                       Type actualTypeArgument = actualTypeArguments[i];
                       return actualTypeArgument;
                   }
               }
           }
        }
        return null;
    }

    @SuppressWarnings("rawtypes")
    private static class MutableTypeVariable{
        private TypeVariable var;
        public MutableTypeVariable(TypeVariable var) {
            this.var = var;
        }
    }
    

}
