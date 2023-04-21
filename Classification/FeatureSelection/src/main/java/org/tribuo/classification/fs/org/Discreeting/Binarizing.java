package FS_Wrapper_Approaches.org.Discreeting;

import static org.apache.commons.math3.special.Erf.erf;

/**
 * This interface includes a static method that is utilized to convert continuous value to binary one
 */
public interface Binarizing {
    /**
     * This method used to convert continuous values to binary ones
     * @param TF is the type (id) of the transfer function
     * @param Value is the continuous value to be converted
     * @return return the converted value based on the selected function
     */
    static int discreteValue(TransferFunction TF, double Value) {
        return switch (TF) {
            case TFunction_V1 -> Math.abs(erf(Math.sqrt(Math.PI) / 2 * Value)) >= 0.5 ? 1 : 0;
            case TFunction_V2 -> Math.abs(Math.tan(Value)) >= 0.5 ? 1 : 0;
            case TFunction_V3 -> Math.abs(Value / Math.abs(1 + Math.pow(Value, 2))) >= 0.5 ? 1 : 0;
            case TFunction_V4 -> Math.abs(2 / Math.PI * Math.atan(Math.PI / 2 * Value)) >= 0.5 ? 1 : 0;
        };
    }
}
