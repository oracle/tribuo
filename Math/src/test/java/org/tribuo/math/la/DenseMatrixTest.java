/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.math.la;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.tribuo.math.la.DenseVectorTest.makeMalformedProto;

import java.util.Optional;
import java.util.Random;

import org.junit.jupiter.api.Test;
import org.tribuo.math.protos.TensorProto;

/**
 * Matrices used -
 * A,B,C = 4x4
 * D = 4x7
 * E = 7x3
 * F = 3x4
 * AD,BD,CD = 4x7
 * DE = 4x3
 * EF = 7x4
 * FA,FB,FC = 3x4
 * FD = 3x7
 * DEF = 4x4
 */
public class DenseMatrixTest {

    public static double[][] identityArr(int size) {
        double[][] values = new double[size][size];

        for (int i = 0; i < size; i++) {
            values[i][i] = 1.0;
        }

        return values;
    }

    public static DenseMatrix identity(int size) {
        return new DenseMatrix(identityArr(size));
    }

    // a 4x4 matrix
    public static DenseMatrix generateA() {
        double[][] values = new double[4][4];

        values[0][0] = 1.0;
        values[0][1] = 2.0;
        values[0][2] = 3.0;
        values[0][3] = 4.0;
        values[1][0] = 5.0;
        values[1][1] = 6.0;
        values[1][2] = 7.0;
        values[1][3] = 8.0;
        values[2][0] = 9.0;
        values[2][1] = 10.0;
        values[2][2] = 11.0;
        values[2][3] = 12.0;
        values[3][0] = 13.0;
        values[3][1] = 14.0;
        values[3][2] = 15.0;
        values[3][3] = 16.0;

        return new DenseMatrix(values);
    }

    // a 4x4 matrix
    public static DenseMatrix generateB() {
        double[][] values = new double[4][4];

        values[0][0] = 1.0;
        values[0][1] = -2.0;
        values[0][2] = 3.0;
        values[0][3] = -4.0;
        values[1][0] = 5.0;
        values[1][1] = -6.0;
        values[1][2] = 7.0;
        values[1][3] = -8.0;
        values[2][0] = 9.0;
        values[2][1] = -10.0;
        values[2][2] = 11.0;
        values[2][3] = -12.0;
        values[3][0] = 13.0;
        values[3][1] = -14.0;
        values[3][2] = 15.0;
        values[3][3] = -16.0;

        return new DenseMatrix(values);
    }

    // a 4x4 matrix
    public static DenseMatrix generateC() {
        double[][] values = new double[4][4];

        values[0][0] = 9;
        values[0][1] = 24;
        values[0][2] = 125;
        values[0][3] = 1;
        values[1][0] = 321;
        values[1][1] = 526;
        values[1][2] = 32;
        values[1][3] = 20;
        values[2][0] = 1235;
        values[2][1] = 892;
        values[2][2] = 159;
        values[2][3] = 732;
        values[3][0] = 2;
        values[3][1] = 5;
        values[3][2] = 8;
        values[3][3] = 4;

        return new DenseMatrix(values);
    }

    // a 4x7 matrix
    public static DenseMatrix generateD() {
        double[][] values = new double[4][7];

        values[0][0] = 1;
        values[0][1] = 2;
        values[0][2] = 3;
        values[0][3] = 4;
        values[0][4] = 5;
        values[0][5] = 6;
        values[0][6] = 7;
        values[1][0] = 8;
        values[1][1] = 9;
        values[1][2] = 10;
        values[1][3] = 11;
        values[1][4] = 12;
        values[1][5] = 13;
        values[1][6] = 14;
        values[2][0] = 15;
        values[2][1] = 16;
        values[2][2] = 17;
        values[2][3] = 18;
        values[2][4] = 19;
        values[2][5] = 20;
        values[2][6] = 21;
        values[3][0] = 22;
        values[3][1] = 23;
        values[3][2] = 24;
        values[3][3] = 25;
        values[3][4] = 26;
        values[3][5] = 27;
        values[3][6] = 28;

        return new DenseMatrix(values);
    }

    // a 7x3 matrix
    public static DenseMatrix generateE() {
        double[][] values = new double[7][3];

        values[0][0] = 21;
        values[0][1] = 20;
        values[0][2] = 19;
        values[1][0] = 18;
        values[1][1] = -17;
        values[1][2] = 16;
        values[2][0] = 15;
        values[2][1] = 14;
        values[2][2] = 13;
        values[3][0] = -12;
        values[3][1] = -11;
        values[3][2] = -10;
        values[4][0] = -9;
        values[4][1] = -8;
        values[4][2] = -7;
        values[5][0] = 6;
        values[5][1] = 5;
        values[5][2] = 4;
        values[6][0] = -3;
        values[6][1] = 2;
        values[6][2] = -1;

        return new DenseMatrix(values);
    }

    // a 3x4 matrix
    public static DenseMatrix generateF() {
        double[][] values = new double[3][4];

        values[0][0] = 265;
        values[0][1] = 35;
        values[0][2] = 32438;
        values[0][3] = 234577;
        values[1][0] = 201;
        values[1][1] = 3.54;
        values[1][2] = 354;
        values[1][3] = 873;
        values[2][0] = 612;
        values[2][1] = 896;
        values[2][2] = 978;
        values[2][3] = 12;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateSymmetric() {
        double[][] values = new double[3][3];

        values[0][0] = 4;
        values[0][1] = 12;
        values[0][2] = -16;
        values[1][0] = 12;
        values[1][1] = 37;
        values[1][2] = -43;
        values[2][0] = -16;
        values[2][1] = -43;
        values[2][2] = 98;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateCholOutput() {
        double[][] values = new double[3][3];

        values[0][0] = 2;
        values[0][1] = 0;
        values[0][2] = 0;
        values[1][0] = 6;
        values[1][1] = 1;
        values[1][2] = 0;
        values[2][0] = -8;
        values[2][1] = 5;
        values[2][2] = 3;

        return new DenseMatrix(values);
    }

    public static DenseMatrix.LUFactorization generateLUOutput() {
        double[][] lValues = new double[3][3];

        lValues[0][0] = 1;
        lValues[0][1] = 0;
        lValues[0][2] = 0;
        lValues[1][0] = -0.75;
        lValues[1][1] = 1;
        lValues[1][2] = 0;
        lValues[2][0] = -0.25;
        lValues[2][1] = 0.263157894736842;
        lValues[2][2] = 1;

        DenseMatrix l = new DenseMatrix(lValues);

        double[][] uValues = new double[3][3];

        uValues[0][0] = -16;
        uValues[0][1] = -43;
        uValues[0][2] = 98;
        uValues[1][0] = 0;
        uValues[1][1] = 4.75;
        uValues[1][2] = 30.5;
        uValues[2][0] = 0;
        uValues[2][1] = 0;
        uValues[2][2] = 0.473684210526317;

        DenseMatrix u = new DenseMatrix(uValues);

        int[] permutation = new int[]{2,1,0};

        DenseMatrix.LUFactorization lu = new DenseMatrix.LUFactorization(l,u,permutation,true);

        return lu;
    }

    public static DenseVector generateVector() {
        double[] values = new double[4];

        values[0] = 1;
        values[1] = 2;
        values[2] = 3;
        values[3] = 4;

        return new DenseVector(values);
    }

    public static DenseMatrix generateOneDimMatrix() {
        double[][] values = new double[4][1];

        values[0][0] = 1;
        values[1][0] = 2;
        values[2][0] = 3;
        values[3][0] = 4;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateAA() {
        double[][] values = new double[4][4];

        values[0][0] = 90;
        values[0][1] = 100;
        values[0][2] = 110;
        values[0][3] = 120;
        values[1][0] = 202;
        values[1][1] = 228;
        values[1][2] = 254;
        values[1][3] = 280;
        values[2][0] = 314;
        values[2][1] = 356;
        values[2][2] = 398;
        values[2][3] = 440;
        values[3][0] = 426;
        values[3][1] = 484;
        values[3][2] = 542;
        values[3][3] = 600;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateAB() {
        double[][] values = new double[4][4];

        values[0][0] = 90;
        values[0][1] = -100;
        values[0][2] = 110;
        values[0][3] = -120;
        values[1][0] = 202;
        values[1][1] = -228;
        values[1][2] = 254;
        values[1][3] = -280;
        values[2][0] = 314;
        values[2][1] = -356;
        values[2][2] = 398;
        values[2][3] = -440;
        values[3][0] = 426;
        values[3][1] = -484;
        values[3][2] = 542;
        values[3][3] = -600;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateAC() {
        double[][] values = new double[4][4];

        values[0][0] = 4364;
        values[0][1] = 3772;
        values[0][2] = 698;
        values[0][3] = 2253;
        values[1][0] = 10632;
        values[1][1] = 9560;
        values[1][2] = 1994;
        values[1][3] = 5281;
        values[2][0] = 16900;
        values[2][1] = 15348;
        values[2][2] = 3290;
        values[2][3] = 8309;
        values[3][0] = 23168;
        values[3][1] = 21136;
        values[3][2] = 4586;
        values[3][3] = 11337;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateBA() {
        double[][] values = new double[4][4];

        values[0][0] = -34;
        values[0][1] = -36;
        values[0][2] = -38;
        values[0][3] = -40;
        values[1][0] = -66;
        values[1][1] = -68;
        values[1][2] = -70;
        values[1][3] = -72;
        values[2][0] = -98;
        values[2][1] = -100;
        values[2][2] = -102;
        values[2][3] = -104;
        values[3][0] = -130;
        values[3][1] = -132;
        values[3][2] = -134;
        values[3][3] = -136;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateBB() {
        double[][] values = new double[4][4];

        values[0][0] = -34;
        values[0][1] = 36;
        values[0][2] = -38;
        values[0][3] = 40;
        values[1][0] = -66;
        values[1][1] = 68;
        values[1][2] = -70;
        values[1][3] = 72;
        values[2][0] = -98;
        values[2][1] = 100;
        values[2][2] = -102;
        values[2][3] = 104;
        values[3][0] = -130;
        values[3][1] = 132;
        values[3][2] = -134;
        values[3][3] = 136;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateBC() {
        double[][] values = new double[4][4];

        values[0][0] = 3064;
        values[0][1] = 1628;
        values[0][2] = 506;
        values[0][3] = 2141;
        values[1][0] = 6748;
        values[1][1] = 3168;
        values[1][2] = 1482;
        values[1][3] = 4977;
        values[2][0] = 10432;
        values[2][1] = 4708;
        values[2][2] = 2458;
        values[2][3] = 7813;
        values[3][0] = 14116;
        values[3][1] = 6248;
        values[3][2] = 3434;
        values[3][3] = 10649;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateCA() {
        double[][] values = new double[4][4];

        values[0][0] = 1267;
        values[0][1] = 1426;
        values[0][2] = 1585;
        values[0][3] = 1744;
        values[1][0] = 3499;
        values[1][1] = 4398;
        values[1][2] = 5297;
        values[1][3] = 6196;
        values[2][0] = 16642;
        values[2][1] = 19660;
        values[2][2] = 22678;
        values[2][3] = 25696;
        values[3][0] = 151;
        values[3][1] = 170;
        values[3][2] = 189;
        values[3][3] = 208;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateCB() {
        double[][] values = new double[4][4];

        values[0][0] = 1267;
        values[0][1] = -1426;
        values[0][2] = 1585;
        values[0][3] = -1744;
        values[1][0] = 3499;
        values[1][1] = -4398;
        values[1][2] = 5297;
        values[1][3] = -6196;
        values[2][0] = 16642;
        values[2][1] = -19660;
        values[2][2] = 22678;
        values[2][3] = -25696;
        values[3][0] = 151;
        values[3][1] = -170;
        values[3][2] = 189;
        values[3][3] = -208;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateCC() {
        double[][] values = new double[4][4];

        values[0][0] = 162162;
        values[0][1] = 124345;
        values[0][2] = 21776;
        values[0][3] = 91993;
        values[1][0] = 211295;
        values[1][1] = 313024;
        values[1][2] = 62205;
        values[1][3] = 34345;
        values[2][0] = 495276;
        values[2][1] = 644320;
        values[2][2] = 214056;
        values[2][3] = 138391;
        values[3][0] = 11511;
        values[3][1] = 9834;
        values[3][2] = 1714;
        values[3][3] = 5974;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateAD() {
        double[][] values = new double[4][7];

        values[0][0] = 150;
        values[0][1] = 160;
        values[0][2] = 170;
        values[0][3] = 180;
        values[0][4] = 190;
        values[0][5] = 200;
        values[0][6] = 210;
        values[1][0] = 334;
        values[1][1] = 360;
        values[1][2] = 386;
        values[1][3] = 412;
        values[1][4] = 438;
        values[1][5] = 464;
        values[1][6] = 490;
        values[2][0] = 518;
        values[2][1] = 560;
        values[2][2] = 602;
        values[2][3] = 644;
        values[2][4] = 686;
        values[2][5] = 728;
        values[2][6] = 770;
        values[3][0] = 702;
        values[3][1] = 760;
        values[3][2] = 818;
        values[3][3] = 876;
        values[3][4] = 934;
        values[3][5] = 992;
        values[3][6] = 1050;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateBD() {
        double[][] values = new double[4][7];

        values[0][0] = -58;
        values[0][1] = -60;
        values[0][2] = -62;
        values[0][3] = -64;
        values[0][4] = -66;
        values[0][5] = -68;
        values[0][6] = -70;
        values[1][0] = -114;
        values[1][1] = -116;
        values[1][2] = -118;
        values[1][3] = -120;
        values[1][4] = -122;
        values[1][5] = -124;
        values[1][6] = -126;
        values[2][0] = -170;
        values[2][1] = -172;
        values[2][2] = -174;
        values[2][3] = -176;
        values[2][4] = -178;
        values[2][5] = -180;
        values[2][6] = -182;
        values[3][0] = -226;
        values[3][1] = -228;
        values[3][2] = -230;
        values[3][3] = -232;
        values[3][4] = -234;
        values[3][5] = -236;
        values[3][6] = -238;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateCD() {
        double[][] values = new double[4][7];

        values[0][0] = 2098;
        values[0][1] = 2257;
        values[0][2] = 2416;
        values[0][3] = 2575;
        values[0][4] = 2734;
        values[0][5] = 2893;
        values[0][6] = 3052;
        values[1][0] = 5449;
        values[1][1] = 6348;
        values[1][2] = 7247;
        values[1][3] = 8146;
        values[1][4] = 9045;
        values[1][5] = 9944;
        values[1][6] = 10843;
        values[2][0] = 26860;
        values[2][1] = 29878;
        values[2][2] = 32896;
        values[2][3] = 35914;
        values[2][4] = 38932;
        values[2][5] = 41950;
        values[2][6] = 44968;
        values[3][0] = 250;
        values[3][1] = 269;
        values[3][2] = 288;
        values[3][3] = 307;
        values[3][4] = 326;
        values[3][5] = 345;
        values[3][6] = 364;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateFA() {
        double[][] values = new double[3][4];

        values[0][0] = 3341883.00;
        values[0][1] = 3609198.00;
        values[0][2] = 3876513.00;
        values[0][3] = 4143828.00;
        values[1][0] =   14753.70;
        values[1][1] =   16185.24;
        values[1][2] =   17616.78;
        values[1][3] =   19048.32;
        values[2][0] =   14050.00;
        values[2][1] =   16548.00;
        values[2][2] =   19046.00;
        values[2][3] =   21544.00;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateFB() {
        double[][] values = new double[3][4];

        values[0][0] = 3341883.00;
        values[0][1] = -3609198.00;
        values[0][2] = 3876513.00;
        values[0][3] = -4143828.00;
        values[1][0] =   14753.70;
        values[1][1] =   -16185.24;
        values[1][2] =   17616.78;
        values[1][3] =   -19048.32;
        values[2][0] =   14050.00;
        values[2][1] =   -16548.00;
        values[2][2] =   19046.00;
        values[2][3] =   -21544.00;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateFC() {
        double[][] values = new double[3][4];

        values[0][0] = 40543704.00;
        values[0][1] = 30132351.00;
        values[0][2] =  7068503.00;
        values[0][3] = 24683889.00;
        values[1][0] =   441881.34;
        values[1][1] =   326819.04;
        values[1][2] =    88508.28;
        values[1][3] =   262891.80;
        values[2][0] =  1500978.00;
        values[2][1] =  1358420.00;
        values[2][2] =   260770.00;
        values[2][3] =   734476.00;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateDE() {
        double[][] values = new double[4][3];

        values[0][0] =  24;
        values[0][1] = -12;
        values[0][2] =  32;
        values[1][0] = 276;
        values[1][1] =  23;
        values[1][2] = 270;
        values[2][0] = 528;
        values[2][1] =  58;
        values[2][2] = 508;
        values[3][0] = 780;
        values[3][1] =  93;
        values[3][2] = 746;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateFD() {
        double[][] values = new double[3][7];

        values[0][0] = 5647809.00;
        values[0][1] = 5915124.00;
        values[0][2] = 6182439.00;
        values[0][3] = 6449754.00;
        values[0][4] = 6717069.00;
        values[0][5] = 6984384.00;
        values[0][6] = 7251699.00;
        values[1][0] =   24745.32;
        values[1][1] =   26176.86;
        values[1][2] =   27608.40;
        values[1][3] =   29039.94;
        values[1][4] =   30471.48;
        values[1][5] =   31903.02;
        values[1][6] =   33334.56;
        values[2][0] =   22714.00;
        values[2][1] =   25212.00;
        values[2][2] =   27710.00;
        values[2][3] =   30208.00;
        values[2][4] =   32706.00;
        values[2][5] =   35204.00;
        values[2][6] =   37702.00;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateEF() {
        double[][] values = new double[7][4];

        values[0][0] =    21213.00;
        values[0][1] =    17829.80;
        values[0][2] =   706860.00;
        values[0][3] =  4943805.00;
        values[1][0] =    11145.00;
        values[1][1] =    14905.82;
        values[1][2] =   593514.00;
        values[1][3] =  4207737.00;
        values[2][0] =    14745.00;
        values[2][1] =    12222.56;
        values[2][2] =   504240.00;
        values[2][3] =  3531033.00;
        values[3][0] =   -11511.00;
        values[3][1] =    -9418.94;
        values[3][2] =  -402930.00;
        values[3][3] = -2824647.00;
        values[4][0] =    -8277.00;
        values[4][1] =    -6615.32;
        values[4][2] =  -301620.00;
        values[4][3] = -2118261.00;
        values[5][0] =     5043.00;
        values[5][1] =     3811.70;
        values[5][2] =   200310.00;
        values[5][3] =  1411875.00;
        values[6][0] =    -1005.00;
        values[6][1] =     -993.92;
        values[6][2] =   -97584.00;
        values[6][3] =  -701997.00;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateDEF() {
        double[][] values = new double[4][4];

        values[0][0] =     23532.00;
        values[0][1] =     29469.52;
        values[0][2] =    805560.00;
        values[0][3] =   5619756.00;
        values[1][0] =    243003.00;
        values[1][1] =    251661.42;
        values[1][2] =   9225090.00;
        values[1][3] =  64766571.00;
        values[2][0] =    462474.00;
        values[2][1] =    473853.32;
        values[2][2] =  17644620.00;
        values[2][3] = 123913386.00;
        values[3][0] =    681945.00;
        values[3][1] =    696045.22;
        values[3][2] =  26064150.00;
        values[3][3] = 183060201.00;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateATA() {
        double[][] values = new double[4][4];

        //Note indices, which are different to the rest
        values[0][0] = 276;
        values[1][0] = 304;
        values[2][0] = 332;
        values[3][0] = 360;
        values[0][1] = 304;
        values[1][1] = 336;
        values[2][1] = 368;
        values[3][1] = 400;
        values[0][2] = 332;
        values[1][2] = 368;
        values[2][2] = 404;
        values[3][2] = 440;
        values[0][3] = 360;
        values[1][3] = 400;
        values[2][3] = 440;
        values[3][3] = 480;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateATB() {
        double[][] values = new double[4][4];

        //Note indices, which are different to the rest
        values[0][0] = 276;
        values[1][0] = 304;
        values[2][0] = 332;
        values[3][0] = 360;
        values[0][1] = -304;
        values[1][1] = -336;
        values[2][1] = -368;
        values[3][1] = -400;
        values[0][2] = 332;
        values[1][2] = 368;
        values[2][2] = 404;
        values[3][2] = 440;
        values[0][3] = -360;
        values[1][3] = -400;
        values[2][3] = -440;
        values[3][3] = -480;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateATC() {
        double[][] values = new double[4][4];

        //Note indices, which are different to the rest
        values[0][0] = 12755;
        values[1][0] = 14322;
        values[2][0] = 15889;
        values[3][0] = 17456;
        values[0][1] = 10747;
        values[1][1] = 12194;
        values[2][1] = 13641;
        values[3][1] = 15088;
        values[0][2] =  1820;
        values[1][2] =  2144;
        values[2][2] =  2468;
        values[3][2] =  2792;
        values[0][3] =  6741;
        values[1][3] =  7498;
        values[2][3] =  8255;
        values[3][3] =  9012;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateBTA() {
        double[][] values = new double[4][4];

        //Note indices, which are different to the rest
        values[0][0] =  276;
        values[1][0] = -304;
        values[2][0] =  332;
        values[3][0] = -360;
        values[0][1] =  304;
        values[1][1] = -336;
        values[2][1] =  368;
        values[3][1] = -400;
        values[0][2] =  332;
        values[1][2] = -368;
        values[2][2] =  404;
        values[3][2] = -440;
        values[0][3] =  360;
        values[1][3] = -400;
        values[2][3] =  440;
        values[3][3] = -480;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateBTB() {
        double[][] values = new double[4][4];

        //Note indices, which are different to the rest
        values[0][0] = 276 ;
        values[1][0] = -304;
        values[2][0] = 332 ;
        values[3][0] = -360;
        values[0][1] = -304;
        values[1][1] = 336 ;
        values[2][1] = -368;
        values[3][1] = 400 ;
        values[0][2] = 332 ;
        values[1][2] = -368;
        values[2][2] = 404 ;
        values[3][2] = -440;
        values[0][3] = -360;
        values[1][3] = 400 ;
        values[2][3] = -440;
        values[3][3] =  480;

        return new DenseMatrix(values);
    }

    /**
     * Note, returns a matrix not bitcoin.
     * @return A dense matrix representing B transpose times C.
     */
    public static DenseMatrix generateBTC() {
        double[][] values = new double[4][4];

        //Note indices, which are different to the rest
        values[0][0] =  12755;
        values[1][0] = -14322;
        values[2][0] =  15889;
        values[3][0] = -17456;
        values[0][1] =  10747;
        values[1][1] = -12194;
        values[2][1] =  13641;
        values[3][1] = -15088;
        values[0][2] =  1820 ;
        values[1][2] = -2144 ;
        values[2][2] =  2468 ;
        values[3][2] = -2792 ;
        values[0][3] =  6741 ;
        values[1][3] = -7498 ;
        values[2][3] =  8255 ;
        values[3][3] = -9012 ;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateCTA() {
        double[][] values = new double[4][4];

        //Note indices, which are different to the rest
        values[0][0] = 12755 ;
        values[1][0] = 10747 ;
        values[2][0] = 1820  ;
        values[3][0] = 6741  ;
        values[0][1] = 14322 ;
        values[1][1] = 12194 ;
        values[2][1] = 2144  ;
        values[3][1] = 7498  ;
        values[0][2] = 15889 ;
        values[1][2] = 13641 ;
        values[2][2] = 2468  ;
        values[3][2] = 8255  ;
        values[0][3] = 17456 ;
        values[1][3] = 15088 ;
        values[2][3] = 2792  ;
        values[3][3] = 9012  ;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateCTB() {
        double[][] values = new double[4][4];

        //Note indices, which are different to the rest
        values[0][0] =  12755;
        values[1][0] =  10747;
        values[2][0] =  1820 ;
        values[3][0] =  6741 ;
        values[0][1] = -14322;
        values[1][1] = -12194;
        values[2][1] = -2144 ;
        values[3][1] = -7498 ;
        values[0][2] =  15889;
        values[1][2] =  13641;
        values[2][2] =  2468 ;
        values[3][2] =  8255 ;
        values[0][3] = -17456;
        values[1][3] = -15088;
        values[2][3] = -2792 ;
        values[3][3] = -9012 ;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateCTC() {
        double[][] values = new double[4][4];

        //Note indices, which are different to the rest
        values[0][0] = 1628351;
        values[1][0] = 1270692;
        values[2][0] = 207778 ;
        values[3][0] = 910457 ;
        values[0][1] = 1270692;
        values[1][1] = 1072941;
        values[2][1] = 161700 ;
        values[3][1] = 663508 ;
        values[0][2] = 207778 ;
        values[1][2] = 161700 ;
        values[2][2] = 41994  ;
        values[3][2] = 117185 ;
        values[0][3] = 910457 ;
        values[1][3] = 663508 ;
        values[2][3] = 117185 ;
        values[3][3] = 536241 ;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateAAT() {
        double[][] values = new double[4][4];

        //Note indices, which are different to the rest
        values[0][0] = 30 ;
        values[1][0] = 70 ;
        values[2][0] = 110;
        values[3][0] = 150;
        values[0][1] = 70 ;
        values[1][1] = 174;
        values[2][1] = 278;
        values[3][1] = 382;
        values[0][2] = 110;
        values[1][2] = 278;
        values[2][2] = 446;
        values[3][2] = 614;
        values[0][3] = 150;
        values[1][3] = 382;
        values[2][3] = 614;
        values[3][3] = 846;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateABT() {
        double[][] values = new double[4][4];

        //Note indices, which are different to the rest
        values[0][0] = -10;
        values[1][0] = -18;
        values[2][0] = -26;
        values[3][0] = -34;
        values[0][1] = -18;
        values[1][1] = -26;
        values[2][1] = -34;
        values[3][1] = -42;
        values[0][2] = -26;
        values[1][2] = -34;
        values[2][2] = -42;
        values[3][2] = -50;
        values[0][3] = -34;
        values[1][3] = -42;
        values[2][3] = -50;
        values[3][3] = -58;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateACT() {
        double[][] values = new double[4][4];

        //Note indices, which are different to the rest
        values[0][0] = 436  ;
        values[1][0] = 1072 ;
        values[2][0] = 1708 ;
        values[3][0] = 2344 ;
        values[0][1] = 1549 ;
        values[1][1] = 5145 ;
        values[2][1] = 8741 ;
        values[3][1] = 12337;
        values[0][2] = 6424 ;
        values[1][2] = 18496;
        values[2][2] = 30568;
        values[3][2] = 42640;
        values[0][3] = 52   ;
        values[1][3] = 128  ;
        values[2][3] = 204  ;
        values[3][3] = 280  ;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateBAT() {
        double[][] values = new double[4][4];

        //Note indices, which are different to the rest
        values[0][0] = -10;
        values[1][0] = -18;
        values[2][0] = -26;
        values[3][0] = -34;
        values[0][1] = -18;
        values[1][1] = -26;
        values[2][1] = -34;
        values[3][1] = -42;
        values[0][2] = -26;
        values[1][2] = -34;
        values[2][2] = -42;
        values[3][2] = -50;
        values[0][3] = -34;
        values[1][3] = -42;
        values[2][3] = -50;
        values[3][3] = -58;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateBBT() {
        double[][] values = new double[4][4];

        //Note indices, which are different to the rest
        values[0][0] = 30 ;
        values[1][0] = 70 ;
        values[2][0] = 110;
        values[3][0] = 150;
        values[0][1] = 70 ;
        values[1][1] = 174;
        values[2][1] = 278;
        values[3][1] = 382;
        values[0][2] = 110;
        values[1][2] = 278;
        values[2][2] = 446;
        values[3][2] = 614;
        values[0][3] = 150;
        values[1][3] = 382;
        values[2][3] = 614;
        values[3][3] = 846;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateBCT() {
        double[][] values = new double[4][4];

        //Note indices, which are different to the rest
        values[0][0] = 332  ;
        values[1][0] = 768  ;
        values[2][0] = 1204 ;
        values[3][0] = 1640 ;
        values[0][1] = -715 ;
        values[1][1] = -1487;
        values[2][1] = -2259;
        values[3][1] = -3031;
        values[0][2] = -3000;
        values[1][2] = -3920;
        values[2][2] = -4840;
        values[3][2] = -5760;
        values[0][3] = 0    ;
        values[1][3] = 4    ;
        values[2][3] = 8    ;
        values[3][3] = 12   ;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateCAT() {
        double[][] values = new double[4][4];

        //Note indices, which are different to the rest
        values[0][0] = 436  ;
        values[1][0] = 1549 ;
        values[2][0] = 6424 ;
        values[3][0] = 52   ;
        values[0][1] = 1072 ;
        values[1][1] = 5145 ;
        values[2][1] = 18496;
        values[3][1] = 128  ;
        values[0][2] = 1708 ;
        values[1][2] = 8741 ;
        values[2][2] = 30568;
        values[3][2] = 204  ;
        values[0][3] = 2344 ;
        values[1][3] = 12337;
        values[2][3] = 42640;
        values[3][3] = 280  ;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateCBT() {
        double[][] values = new double[4][4];

        //Note indices, which are different to the rest
        values[0][0] = 332  ;
        values[1][0] = -715 ;
        values[2][0] = -3000;
        values[3][0] = 0    ;
        values[0][1] = 768  ;
        values[1][1] = -1487;
        values[2][1] = -3920;
        values[3][1] = 4    ;
        values[0][2] = 1204 ;
        values[1][2] = -2259;
        values[2][2] = -4840;
        values[3][2] = 8    ;
        values[0][3] = 1640 ;
        values[1][3] = -3031;
        values[2][3] = -5760;
        values[3][3] = 12   ;

        return new DenseMatrix(values);
    }

    public static DenseMatrix generateCCT() {
        double[][] values = new double[4][4];

        //Note indices, which are different to the rest
        values[0][0] = 16283  ;
        values[1][0] = 19533  ;
        values[2][0] = 53130  ;
        values[3][0] = 1142   ;
        values[0][1] = 19533  ;
        values[1][1] = 381141 ;
        values[2][1] = 885355 ;
        values[3][1] = 3608   ;
        values[0][2] = 53130  ;
        values[1][2] = 885355 ;
        values[2][2] = 2881994;
        values[3][2] = 11130  ;
        values[0][3] = 1142   ;
        values[1][3] = 3608   ;
        values[2][3] = 11130  ;
        values[3][3] = 109    ;

        return new DenseMatrix(values);
    }

    @Test
    public void identityTest() {
        DenseMatrix a = generateA();
        DenseMatrix b = generateB();
        DenseMatrix c = generateC();
        DenseMatrix d = generateD();
        DenseMatrix e = generateE();
        DenseMatrix f = generateF();
        DenseMatrix eye = identity(10);
        DenseMatrix eye4 = identity(4);
        DenseMatrix eye3 = identity(3);
        DenseMatrix eye7 = identity(7);

        // Identity matrix tests
        assertEquals(eye,eye.matrixMultiply(eye));
        assertEquals(eye4,eye4.matrixMultiply(eye4));
        assertEquals(eye3,eye3.matrixMultiply(eye3));
        assertEquals(eye7,eye7.matrixMultiply(eye7));

        // 4x4 tests
        assertEquals(a,a.matrixMultiply(eye4));
        assertEquals(b,b.matrixMultiply(eye4));
        assertEquals(c,c.matrixMultiply(eye4));
        assertEquals(a,eye4.matrixMultiply(a));
        assertEquals(b,eye4.matrixMultiply(b));
        assertEquals(c,eye4.matrixMultiply(c));

        // 4x7 tests
        assertEquals(d,d.matrixMultiply(eye7));
        assertEquals(d,eye4.matrixMultiply(d));

        // 7x3 tests
        assertEquals(e,e.matrixMultiply(eye3));
        assertEquals(e,eye7.matrixMultiply(e));

        // 3x4 tests
        assertEquals(f,f.matrixMultiply(eye4));
        assertEquals(f,eye3.matrixMultiply(f));
    }

    @Test
    public void squareMatrixMultiplyTest() {
        DenseMatrix a = generateA();
        DenseMatrix b = generateB();
        DenseMatrix c = generateC();

        DenseMatrix aa = generateAA();
        DenseMatrix ab = generateAB();
        DenseMatrix ac = generateAC();
        assertEquals(aa,a.matrixMultiply(a));
        assertEquals(ab,a.matrixMultiply(b));
        assertEquals(ac,a.matrixMultiply(c));

        DenseMatrix ba = generateBA();
        DenseMatrix bb = generateBB();
        DenseMatrix bc = generateBC();
        assertEquals(ba,b.matrixMultiply(a));
        assertEquals(bb,b.matrixMultiply(b));
        assertEquals(bc,b.matrixMultiply(c));

        DenseMatrix ca = generateCA();
        DenseMatrix cb = generateCB();
        DenseMatrix cc = generateCC();
        assertEquals(ca,c.matrixMultiply(a));
        assertEquals(cb,c.matrixMultiply(b));
        assertEquals(cc,c.matrixMultiply(c));
    }

    @Test
    public void matrixMultiplyTest() {
        //4x4 matrices
        DenseMatrix a = generateA();
        DenseMatrix b = generateB();
        DenseMatrix c = generateC();

        //4x7 matrix
        DenseMatrix d = generateD();

        //7x3 matrix
        DenseMatrix e = generateE();

        //3x4 matrix
        DenseMatrix f = generateF();

        //4x7 output
        DenseMatrix ad = generateAD();
        assertEquals(ad,a.matrixMultiply(d));
        DenseMatrix bd = generateBD();
        assertEquals(bd,b.matrixMultiply(d));
        DenseMatrix cd = generateCD();
        assertEquals(cd,c.matrixMultiply(d));

        //3x4 output
        DenseMatrix fa = generateFA();
        assertEquals(fa,f.matrixMultiply(a));
        DenseMatrix fb = generateFB();
        assertEquals(fb,f.matrixMultiply(b));
        DenseMatrix fc = generateFC();
        assertEquals(fc,f.matrixMultiply(c));

        //4x3 output
        DenseMatrix de = generateDE();
        assertEquals(de,d.matrixMultiply(e));

        //3x7 output
        DenseMatrix fd = generateFD();
        assertEquals(fd,f.matrixMultiply(d));

        //7x4 output
        DenseMatrix ef = generateEF();
        assertEquals(ef,e.matrixMultiply(f));

        //4x4 output
        DenseMatrix def = generateDEF();
        assertEquals(def,d.matrixMultiply(e).matrixMultiply(f));
    }

    @Test
    public void matrixMatrixBothTransposeTest() {
        //4x4 matrices
        DenseMatrix a = generateA();
        DenseMatrix b = generateB();
        DenseMatrix c = generateC();

        DenseMatrix aa = generateAA();
        assertEquals(aa.transpose(), a.matrixMultiply(a,true,true));
        DenseMatrix ab = generateAB();
        assertEquals(ab.transpose(), b.matrixMultiply(a,true,true));
        DenseMatrix ac = generateAC();
        assertEquals(ac.transpose(), c.matrixMultiply(a,true,true));

        DenseMatrix ba = generateBA();
        assertEquals(ba.transpose(), a.matrixMultiply(b,true,true));
        DenseMatrix bb = generateBB();
        assertEquals(bb.transpose(), b.matrixMultiply(b,true,true));
        DenseMatrix bc = generateBC();
        assertEquals(bc.transpose(), c.matrixMultiply(b,true,true));

        DenseMatrix ca = generateCA();
        assertEquals(ca.transpose(), a.matrixMultiply(c,true,true));
        DenseMatrix cb = generateCB();
        assertEquals(cb.transpose(), b.matrixMultiply(c,true,true));
        DenseMatrix cc = generateCC();
        assertEquals(cc.transpose(), c.matrixMultiply(c,true,true));
    }

    @Test
    public void matrixMatrixThisTransposeTest() {
        //4x4 matrices
        DenseMatrix a = generateA();
        DenseMatrix b = generateB();
        DenseMatrix c = generateC();

        DenseMatrix aTa = generateATA();
        assertEquals(aTa,a.matrixMultiply(a,true,false));
        DenseMatrix aTb = generateATB();
        assertEquals(aTb,a.matrixMultiply(b,true,false));
        DenseMatrix aTc = generateATC();
        assertEquals(aTc,a.matrixMultiply(c,true,false));

        DenseMatrix bTa = generateBTA();
        assertEquals(bTa,b.matrixMultiply(a,true,false));
        DenseMatrix bTb = generateBTB();
        assertEquals(bTb,b.matrixMultiply(b,true,false));
        DenseMatrix bTc = generateBTC();
        assertEquals(bTc,b.matrixMultiply(c,true,false));

        DenseMatrix cTa = generateCTA();
        assertEquals(cTa,c.matrixMultiply(a,true,false));
        DenseMatrix cTb = generateCTB();
        assertEquals(cTb,c.matrixMultiply(b,true,false));
        DenseMatrix cTc = generateCTC();
        assertEquals(cTc,c.matrixMultiply(c,true,false));
    }

    @Test
    public void matrixMatrixOtherTransposeTest() {
        //4x4 matrices
        DenseMatrix a = generateA();
        DenseMatrix b = generateB();
        DenseMatrix c = generateC();

        DenseMatrix aaT = generateAAT();
        assertEquals(aaT,a.matrixMultiply(a,false,true));
        DenseMatrix abT = generateABT();
        assertEquals(abT,a.matrixMultiply(b,false,true));
        DenseMatrix acT = generateACT();
        assertEquals(acT,a.matrixMultiply(c,false,true));

        DenseMatrix baT = generateBAT();
        assertEquals(baT,b.matrixMultiply(a,false,true));
        DenseMatrix bbT = generateBBT();
        assertEquals(bbT,b.matrixMultiply(b,false,true));
        DenseMatrix bcT = generateBCT();
        assertEquals(bcT,b.matrixMultiply(c,false,true));

        DenseMatrix caT = generateCAT();
        assertEquals(caT,c.matrixMultiply(a,false,true));
        DenseMatrix cbT = generateCBT();
        assertEquals(cbT,c.matrixMultiply(b,false,true));
        DenseMatrix ccT = generateCCT();
        assertEquals(ccT,c.matrixMultiply(c,false,true));
    }

    @Test
    public void matrixVectorTest() {
        DenseMatrix a = generateA();
        DenseMatrix b = generateB();
        DenseMatrix c = generateC();

        DenseVector vector = generateVector();
        DenseMatrix oneDimMatrix = generateOneDimMatrix();

        SGDVector matrixMatrixOutput;
        SGDVector matrixVectorOutput;

        matrixMatrixOutput = a.matrixMultiply(oneDimMatrix).getColumn(0);
        matrixVectorOutput = a.leftMultiply(vector);
        assertEquals(matrixMatrixOutput,matrixVectorOutput);

        matrixMatrixOutput = b.matrixMultiply(oneDimMatrix).getColumn(0);
        matrixVectorOutput = b.leftMultiply(vector);
        assertEquals(matrixMatrixOutput,matrixVectorOutput);

        matrixMatrixOutput = c.matrixMultiply(oneDimMatrix).getColumn(0);
        matrixVectorOutput = c.leftMultiply(vector);
        assertEquals(matrixMatrixOutput,matrixVectorOutput);
    }

    @Test
    public void serializationTest() {
        DenseMatrix a = generateA();
        TensorProto proto = a.serialize();
        Tensor deser = Tensor.deserialize(proto);
        assertEquals(a,deser);
    }

    @Test
    public void serializationValidationTest() {
        String className = DenseMatrix.class.getName();
        TensorProto invalidShape = makeMalformedProto(className, new int[]{-1}, new double[1]);
        try {
            Tensor deser = Tensor.deserialize(invalidShape);
            fail("Should have thrown ISE");
        } catch (IllegalStateException e) {
            //pass
        }
        invalidShape = makeMalformedProto(className, new int[]{3}, new double[1]);
        try {
            Tensor deser = Tensor.deserialize(invalidShape);
            fail("Should have thrown ISE");
        } catch (IllegalStateException e) {
            //pass
        }
        TensorProto elementMismatch = makeMalformedProto(className, new int[]{5,4}, new double[1]);
        try {
            Tensor deser = Tensor.deserialize(elementMismatch);
            fail("Should have thrown ISE");
        } catch (IllegalStateException e) {
            //pass
        }
    }

    @Test
    public void symmetricTest() {
        assertFalse(generateA().isSymmetric());
        assertFalse(generateB().isSymmetric());
        assertTrue(generateSymmetric().isSymmetric());
    }

    @Test
    public void choleskyTest() {
        DenseMatrix symmetric = generateSymmetric();
        assertTrue(symmetric.isSymmetric());
        Optional<DenseMatrix.CholeskyFactorization> cholOpt = symmetric.choleskyFactorization();
        assertTrue(cholOpt.isPresent());

        // check pre-computed output
        DenseMatrix.CholeskyFactorization chol = cholOpt.get();
        assertEquals(generateCholOutput(),chol.lMatrix());

        // check factorization
        Matrix computedSymmetric = chol.lMatrix().matrixMultiply(chol.lMatrix(),false,true);
        assertEquals(symmetric,computedSymmetric);

        // test factorization
        testFactorization(symmetric, chol, 1e-13);
    }

    
    @Test
    public void choleskyTest2() {
        DenseMatrix a = new DenseMatrix(new double[][] {new double[] {8,2,3}, new double[] {2,9,3}, new double[] {3,3,6}});
        assertTrue(a.isSymmetric());
        Optional<DenseMatrix.CholeskyFactorization> cholOpt = a.choleskyFactorization();
        assertTrue(cholOpt.isPresent());
        DenseMatrix c = cholOpt.get().lMatrix();
        assertEquals(new DenseMatrix(new double[][]{new double[]{2.8284271247461903, 0.0, 0.0}, new double[]{0.7071067811865475, 2.9154759474226504, 0.0}, new double[]{1.0606601717798212, 0.7717436331412897, 2.0686739145418453}}),c);
        // check factorization
        assertEquals(a,c.matrixMultiply(c,false,true));

        a = generateSymmetricPositiveDefinite(5, new Random(1234));
        assertTrue(a.isSymmetric());
        assertTrue(a.isSquare());
        c = a.choleskyFactorization().get().lMatrix();
        assertEquals(a, c.matrixMultiply(c, false, true));
        assertEquals(new DenseMatrix(new double[][]{new double[]{14.94547893340321, 0.0, 0.0, 0.0, 0.0}, new double[]{1.0804097707164124, 8.928639781892967, 0.0, 0.0, 0.0}, new double[]{0.9664077730739391, 0.6309279673272034, 3.0262982901105855, 0.0, 0.0}, new double[]{0.7832294202894065, 1.4377257847560159, 2.2605017656943716, 15.960534656196803, 0.0}, new double[]{0.7843088590247412, 0.04136969152859865, 3.556219834851627, 0.5877350448006833, 16.75596860731699}}), c);
        
        a = generateSymmetricPositiveDefinite(20, new Random(42));
        assertTrue(a.isSymmetric());
        c = a.choleskyFactorization().get().lMatrix();
        assertEquals(a, c.matrixMultiply(c, false, true));
        assertEquals(new DenseMatrix(new double[][]{new double[]{16.75440746002779, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{1.6018808759506862, 13.63602457544494, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.7516483220309336, 1.190157675401124, 5.4036024131544655, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{1.368025817917537, 2.270165339191156, 1.764426591618153, 18.147655254377153, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{1.8923058726188353, 1.0123840323556053, 3.872869835424954, 0.3826508633553371, 17.135975549880694, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{1.4334670987046003, 1.831123954664325, 3.4125124344865716, 0.41829925313682753, -0.47377519666787493, 17.834272696939415, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.9955273667573433, 1.5888606812196417, 1.2931406114626933, 0.7507404663781049, 1.1230328218627708, 0.6982985586204641, 11.729209946273539, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{1.0566045539820537, 1.323704443126565, 2.3405076341478823, 0.8909192341770037, 0.5588640878069887, 0.42167743837570937, 0.6918655860556959, 12.604895816109266, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{1.390901857759653, 1.5244450976350756, 2.1296282171996017, 0.5175951429570781, 0.40949856268727963, 0.29710908111391915, 1.0551485878373852, 0.21347459054055823, 12.185375044419702, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{1.568660236011373, 1.6539549798204274, 2.7054801249873144, 0.5875503505223718, 0.36058080347341825, 0.9631435157481918, 1.8664848238242542, 0.3528938775896363, 1.0221291272724007, 18.468466444374396, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{1.6747699323818752, 0.9348877369145657, 2.26838953173407, 0.9708056620594314, 0.4648508465963835, 0.4502237005191294, 0.8494944520263819, 1.0245845314750173, 0.9046030178733844, -0.25418850237135154, 16.042987693353492, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.8869180378613991, 0.7737834087685098, 1.4084786652563897, 0.12734000384887173, 0.502736132499496, 0.07316975918621323, 0.971795330127886, 0.22343852351493915, 0.895107379408903, 0.7852625675942757, 0.5856430554056011, 5.226898701059288, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{1.566488974573314, 2.295179090448591, 1.7394746979213342, 0.40853611940626017, 0.582994247658075, 0.7148554103841811, 1.474587799832604, 0.3318139748198812, 1.24859872007004, 0.19733009851870595, 0.5557654573528521, -0.19658626542639288, 16.240209978936896, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{1.1818432493780875, 1.5652452724346404, 3.2857719301330177, 0.9635319388163569, 0.7563151193955898, 0.9287100018154593, 1.1770478441746075, 0.02087442066230491, 0.6706469709304964, 0.2620284517297045, 0.3304383096053042, 2.185137328699002, 0.9479299608640565, 13.840735361297632, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.5789967411201067, 0.9988471923692965, 1.183196829816785, 0.7521697338655636, 0.018668631393969338, 0.5116517301295441, 0.8139711973181936, -0.2094555759887365, 0.39243306130843325, 0.07411895260819369, -0.06757487751692495, 0.29838420849840613, 0.8241581418664375, -0.01620837071104333, 3.2260845557553037, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{1.0104885322567119, 0.9325402930773361, 1.301762888169249, 0.6129186638967622, 0.8976470672020842, 0.3659205085404095, 0.34379283852757003, -0.04079047659367302, 0.43869995114135396, 0.943916377651184, 0.8766579013250082, 1.3128333521551454, 0.37860345884081986, 0.20891699349297296, 1.3466852460402794, 4.431218108245882, 0.0, 0.0, 0.0, 0.0}, new double[]{0.5492881425825883, 1.7200875435056948, 2.6137768716353555, 1.5272404478083912, -0.35532261666960424, 0.7701932271918118, 0.9672343431260562, -0.24698254339887663, 0.6459562358661791, 0.4510727698685872, 0.9702553782551973, 1.424541341254133, 0.4390031008245334, -0.12801559327632064, 2.832720413681546, 0.30850316580459985, 15.2646653384555, 0.0, 0.0, 0.0}, new double[]{1.657019097342754, 1.5610725306412903, 2.363618503717435, 0.8735325067275853, -0.3123830847695933, 0.5853338418588566, 1.4777076819228194, 0.31199933818155606, 0.808967278455822, 0.3956563813584885, 0.5128113149029574, 1.6343353507832152, 0.5845773670802912, -0.44100182190227216, 0.9841283372747659, 0.679495505925546, -0.07558768037348314, 13.611207913547869, 0.0, 0.0}, new double[]{0.8955649049493016, 0.887165103491017, 0.9108045971202429, 0.7785116617751562, 0.2633420108871542, 0.2362331319670587, 0.6655798737488293, 0.8378945325807816, 0.5670482817021022, 0.7767713529738539, 0.7850772851108367, 1.7543617362601482, 0.7582218118032898, 0.22930366774009395, -0.0926515221296133, 0.9599536741417869, 0.1721095656171293, 0.42092281314426966, 11.464109239515773, 0.0}, new double[]{1.2970751226801458, 0.7674164191430094, 1.005264076968263, 1.6432539998603195, 0.7770414629193831, 0.9415726468857499, 0.711083760232971, 0.9179014895793605, 0.32172296524409544, 0.006783534749445878, 1.058649212086019, 2.2596039666799954, 0.09793459293296733, 0.16032662213243493, 2.9599257579515035, 1.9884554445584544, 0.45819043454085545, 0.0034148678716191716, 0.4436061788222598, 16.119007983152684}}), c);
    }

    //another library used this trick to make sure the matrix is positive definite
    public static DenseMatrix generateSymmetricPositiveDefinite(int n, Random rng) {
        double[][] values = new double[n][n];
        for(int i=0; i<n; i++) {
            for(int j=0; j<n; j++) {
                values[i][j] = rng.nextDouble();
            }
            values[i][i] = 20*(0.1+values[i][i]);
        }
        DenseMatrix m = new DenseMatrix(values);
        return m.matrixMultiply(m, true, false);
    }

    public static DenseMatrix generateSquareRandom(int n, Random rng) {
        double[][] values = new double[n][n];
        for(int i=0; i<n; i++) {
            for(int j=0; j<n; j++) {
                values[i][j] = rng.nextDouble();
            }
        }
        return new DenseMatrix(values);
    }

    //see https://github.com/scipy/scipy/blob/main/scipy/linalg/tests/test_decomp_cholesky.py
    public static DenseMatrix generateCholeskyTestMatrix() {
        double[][] values = new double[3][3];
        values[0][0] = 8;
        values[0][1] = 2;
        values[0][2] = 3;
        values[1][0] = 2;
        values[1][1] = 9;
        values[1][2] = 3;
        values[2][0] = 3;
        values[2][1] = 3;
        values[2][2] = 6;
        return new DenseMatrix(values);
    }


    @Test
    public void luTest() {
        DenseMatrix symmetric = generateSymmetric();
        Optional<DenseMatrix.LUFactorization> luOpt = symmetric.luFactorization();
        assertTrue(luOpt.isPresent());
        DenseMatrix.LUFactorization lu = luOpt.get();

        // check pre-computed output
        DenseMatrix.LUFactorization output = generateLUOutput();
        assertEquals(output.lower(),luOpt.get().lower());
        assertEquals(output.upper(),luOpt.get().upper());
        assertArrayEquals(output.permutationArr(),luOpt.get().permutationArr());
        assertEquals(output.oddSwaps(),luOpt.get().oddSwaps());

        // check factorization
        Matrix computedSymmetric = lu.lower().matrixMultiply(lu.upper());
        assertEquals(lu.permutationMatrix().matrixMultiply(symmetric),computedSymmetric);

        // test factorization
        testFactorization(symmetric, lu, 1e-13);

        //lets try a couple of non-symmetrical matrices
        DenseMatrix a = generateSquareRandom(10, new Random(42));
        assertFalse(a.isSymmetric());
        luOpt = a.luFactorization();
        assertTrue(luOpt.isPresent());
        lu = luOpt.get();
        lu = a.luFactorization().get();
        Matrix computed = lu.lower().matrixMultiply(lu.upper());
        assertEquals(lu.permutationMatrix().matrixMultiply(a),computed);

        a = generateSquareRandom(20, new Random(42));
        assertFalse(a.isSymmetric());
        luOpt = a.luFactorization();
        assertTrue(luOpt.isPresent());
        lu = luOpt.get();
        lu = a.luFactorization().get();
        computed = lu.lower().matrixMultiply(lu.upper());
        assertEquals(lu.permutationMatrix().matrixMultiply(a),computed);

        //an example computed with another library
        a = new DenseMatrix(new double[][] {new double[] {0.44670904, 0.44742455, 0.45204733},
                                            new double[] {0.71710816, 0.14136726, 0.18301841},
                                            new double[] {0.40983909, 0.07235836, 0.95855327}});
        DenseVector b = new DenseVector(new double[] {0.63392567, 0.93362273, 0.86074978});
        DenseMatrix.LUFactorization lu_a = a.luFactorization().get();
        DenseVector x2 = lu_a.solve(b);
        assertEquals(new DenseVector(new double[] {1.2466263014829564,-0.2127572718468386,0.38102040828578143}), x2);
    }
    
    @Test
    public void luTest10() {
        DenseMatrix a = generateSquareRandom(10, new Random(42));
        assertFalse(a.isSymmetric());
        Optional<DenseMatrix.LUFactorization> luOpt = a.luFactorization();
        luOpt = a.luFactorization();
        assertTrue(luOpt.isPresent());
        DenseMatrix.LUFactorization lu = luOpt.get();
        lu = luOpt.get();
        lu = a.luFactorization().get();
        Matrix computed = lu.lower().matrixMultiply(lu.upper());
        assertEquals(lu.permutationMatrix().matrixMultiply(a),computed);

        DenseMatrix upper = new DenseMatrix(new double[][]{new double[]{0.919327782868717, 0.436490974423287, 0.749906181255448, 0.386566874359349, 0.177378477909378, 0.594349910889684, 0.209767568866332, 0.825965871887821, 0.172217937687852, 0.587427381786296}, new double[]{0.0, 0.6046335192903662, -0.2315554768998931, 0.6366476011582105, 0.36981144543030414, -0.02265336333957818, 0.08894501826509618, -0.2421177129596924, 0.7778246727901739, 0.1868970417505812}, new double[]{0.0, 0.0, 0.582161900514335, 0.5515716470908145, 0.5050928223956577, 0.06080618000984278, 0.24126113352929207, 0.028311954869486233, 0.2856241540086502, 0.14599033226603875}, new double[]{0.0, 0.0, 0.0, 0.9201513995278638, 1.0765116824099081, 0.11333278778720035, 1.0435545524490883, -0.12398318208866327, 1.0420632762843565, 0.6298691400964507}, new double[]{0.0, 0.0, 0.0, 0.0, 1.1359028355590344, -0.2293887981353377, 1.423647757155546, -0.4798724675894213, 0.7684694590742605, 0.1621140233457365}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.6387337970258129, -0.429595684654273, 0.041738035239056726, -0.2567574094585644, 0.31066510702634487}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8464215324300727, -0.3385041472325138, 0.5745459425968332, 0.22845107403060855}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0382120340233407, -0.2084728664560298, -0.04915633819475923}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6513748657999127, -0.28770234443194553}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4330588997846473}});
        assertEquals(upper, lu.upper());
        DenseMatrix lower = new DenseMatrix(new double[][]{new double[]{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.5101798611502932, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.5227455598663895, 0.10499381292122886, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.5409679787159667, -0.3453554379860352, -0.7715373629846236, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.9724962222164031, 0.28835849803934743, 0.5385521322941219, -0.8471242288141386, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.7914083459574576, 0.5586538967647579, -0.2669415253271348, -0.2578728317292549, 0.6435470362577285, 1.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.8172062465284433, 0.35449109373629945, 0.08465209264493595, 0.17847876514502334, -0.42215148181041795, -0.3788241883348112, 1.0, 0.0, 0.0, 0.0}, new double[]{0.45791252529909143, 0.716128188680375, 0.8971569861596805, -0.8829557983548465, 0.6364353827507656, 0.4799824825668223, 0.8472910553690725, 1.0, 0.0, 0.0}, new double[]{0.3428020747767634, 0.48476626101806697, -0.08308375409609205, 0.38019693902353446, -0.38431061971684766, 0.7615607650134028, 0.6171134330987622, -0.029517947598312787, 1.0, 0.0}, new double[]{0.7584332185567152, 0.9552320656695076, -0.26009512153465214, 0.05570230015791643, 0.1862446242150791, 0.13540648537444386, -0.16638626226876788, 0.17955644250561797, 0.30334486487915147, 1.0}});
        assertEquals(lower, lu.lower());
    }

    @Test
    public void luTest20() {
        DenseMatrix a = generateSquareRandom(20, new Random(42));
        DenseMatrix.LUFactorization lu = a.luFactorization().get();
        DenseMatrix upper = new DenseMatrix(new double[][]{new double[]{0.91629377516759, 0.202182492969044, 0.56299024004504, 0.561700943667366, 0.080282170345142, 0.416882590906358, 0.560143976505208, 0.100264341467102, 0.610836098745395, 0.920378070753754, 0.03370946135387, 0.179426442633273, 0.997460814518753, 0.741524133735247, 0.063185128546488, 0.318886141572087, 0.631989300813935, 0.727637943878689, 0.028750514440684, 0.812558114652981}, new double[]{0.0, 0.9052275706085295, 0.7311124942979998, 0.17095063660702303, -0.010669320462571885, 0.6812034755884623, 0.3489823054201352, 0.8042763478267299, 0.3953375940092203, 0.4047899102009168, 0.5460004302339985, 0.06049073424230716, 0.2813126916984501, 0.6667691965772725, 0.6070155383157478, 0.28437108835221503, 0.7089863201825783, 0.6096031527180594, 0.25133910642700047, 0.7119862310531005}, new double[]{0.0, 0.0, -0.8418476964092433, 0.2391655310590597, 0.5757389140414536, -0.42180230458380397, -0.4116354501152575, -0.20697949944360516, -0.15932781555894426, -0.9116064716607113, 0.01642904477034135, -0.15965565256018974, -0.957082051438826, -0.6363568431353184, 0.10085272088659036, -0.0839926518392104, -0.1318782939936175, -0.677359059991486, 0.41479172496712496, -0.44148925159663444}, new double[]{0.0, 0.0, 0.0, 0.679310460740777, 0.30536174388152537, 0.09289890466190279, -0.10901446406792045, 0.6577949827161325, 0.06645844700549122, -0.0020572824488075403, 0.7739619189359576, 0.46160778934827934, -0.20418276695810306, 0.47354251316234486, 0.5206282501079079, 0.8807407912012298, 0.6373368056049682, -0.04212319842740797, 0.5317367928049365, 0.5154946648904738}, new double[]{0.0, 0.0, 0.0, 0.0, 1.004270276574797, -0.41062361258634067, 0.06470483365678648, 0.395375206053751, -0.30879568797903967, -0.8577130542425583, 0.6319041693434939, 0.5732188590079953, -1.2735986438387392, 0.009356547995667985, 0.24898513869846228, 1.0291215968438152, -0.32167040809750125, -1.0019321717583807, 0.34318287299865397, -0.22115588557937782}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.6885304612825144, -0.09878565829551755, 0.11888667536611253, 0.026275391516468516, 0.7797339853049778, 0.7905988710596875, 0.4176897754831432, 0.8325929643882475, 0.13011386047293466, -0.06655370067747732, 0.35944616851482125, -0.07971923964435537, 0.7362704520604607, -0.08009767172937574, 0.2407018974393958}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5635665117604685, -0.6249197850949855, 0.23031320412001716, 0.11247725802992603, -1.2988300632624379, 0.11026088327531813, 0.14024854093374872, -0.3875080862753112, -0.5058502641980529, -0.33893480877287796, -0.23032588232457918, 0.12379631467206165, -0.007968186522634757, -0.972676934413537}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.4070639645891962, -0.12418858944571715, 0.6910032948976171, -1.4404891543646792, 0.2622833083003355, 0.8587138026959753, -0.21585181843661005, -0.3106664976261241, -1.495423265502768, -0.5942855879266988, 0.09640037702399482, -0.10120550590895772, -1.213780372497036}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6802013374354048, 0.1058075299127755, 1.2379538951016724, 0.8253372563137479, 0.28039553297872594, -0.4691976313711233, -0.14590375747978418, 0.2553780343530524, -0.751028592571138, 0.08371269681886573, -0.44019720447366306, 0.03428601335235071}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.8019709962475664, 0.9482737524483995, -0.12829271956674393, 0.19730015198640594, -0.7755174705264541, -0.29453892749883254, -0.6304960591161828, -0.9226950539319441, -0.23156393988772728, -0.7638380971401721, -0.0526118887077868}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.6349095289112598, 0.9880461582144417, 0.8770199686641961, -0.732470256163468, -0.30405294149385953, 0.3323507498525623, -0.8535567865401182, 0.4991743682669355, -0.6767957549007376, 0.015790757312618864}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.8234358031976985, 0.33820564979310785, 0.8532516474938868, 0.7042310546951711, -0.06423357030263827, 0.28005420631706696, 0.2396254014654069, 0.43221953335735164, 0.528719611205809}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7145335303026192, -0.27218542943627877, 0.3313284160982715, -1.2236380259408746, -0.3517536922205687, 0.45984472966121825, 0.314122632468761, -0.6695645773752903}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.3445205726164846, -1.7642384709526822, -0.11149215277908996, -0.7150759280650528, -0.3805758372617428, -0.5346470409282629, -0.2360072475107876}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.603793754175576, -0.7227592385899313, 0.08933690369120117, -0.15647605160230427, -0.6287687377835115, 0.4898330713252278}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.9136570013250238, -0.14654528737983064, -0.044723655495255255, -0.5334655967012899, -0.037340679446577874}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.711745128807139, -0.5079629246932265, 0.8235155496618926, -1.0723815155444778}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.40246596787399197, 0.5083203142718309, -0.24587939289488858}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1995894850960833, -0.09077276170080262}, new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.19718923178294}});
        assertEquals(upper, lu.upper());
        DenseMatrix lower = new DenseMatrix(new double[][]{new double[]{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.3079779583562422, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.7609445225601399, 0.8337847234567108, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.4473141936816384, -0.09494029705026269, 0.1637102253065407, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.975716325972954, 0.44360588537892465, -0.12162520386418879, -0.7621284376905496, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.7940288363301351, 0.5774070069363221, 0.6657504787296222, -0.6283738341691794, 0.4147748029319753, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.47430585732298797, 0.5007279777683581, 0.06697485794746916, 0.5292140082116717, 0.36161236775140626, 0.5908255058790153, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.4197464440148081, 0.844017790432058, 0.4062134998961642, 0.4941441960718356, 0.5766279197366718, 0.2355875305580612, -0.5508060434484056, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.6794156391168887, 0.819068864976729, 0.5181261419263038, 0.4877148304343442, -0.35273920138322346, -0.701843245096469, 0.7092975753114856, -0.2773619520959566, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.5754309984653508, 0.33770322283741516, -0.46620789973611665, 0.993377069976633, 0.7480965060063309, 0.5362313482958221, 0.6321596725234732, 0.4967542962016245, -0.543153895575124, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.5118691552060198, 0.7996126782295147, 0.8573434443825495, 0.3011988820979851, -0.15716572853459956, -0.3026791986783505, -0.006724822479335297, 0.3140663634975763, -0.5196348554321051, -0.5161094309260015, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.5550385681628209, 0.9241643397204394, 0.8565946072603277, -0.04554456415982458, 0.4289197325443804, 0.3930243241875422, 0.8354331063835333, 0.10937713856509156, -0.10748457309978182, 0.07292951498907436, 0.4648891734874218, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.924877489856665, 0.16516986494967228, 0.25925481629627706, -0.8204474315440505, 0.6289897594567927, 0.16257602317783265, -0.17892272534318476, -0.398486864932312, -0.24193475415361745, -0.9495701198542961, 0.6442702974197089, -0.2821881020966079, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.791721578036539, 0.7012628674246133, 0.16059038996068944, 0.3640215888193717, 0.7221996807408587, -0.505654501540419, -0.11534483823016972, 0.4294290147986364, 0.42714956440474094, -0.44132530245664997, 0.09598532383257474, 0.9776071987597791, 0.7023757563795182, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.43385229365210265, 0.5366798318339504, 0.44956224031251935, 0.557601842211255, 0.20012354981960018, 0.417780155890305, 0.5891940018045582, 0.10518293529451214, -0.3286571126891387, -0.488165630534509, 0.5645239802111713, 0.3264587527258916, 0.0049180337098236335, 0.26081377697536984, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.7164868938471124, 0.057444333550601474, 0.5215596397477621, -0.5119247376365421, 0.2409973255461667, 0.3429726755092168, 0.5648373399165897, -0.4890784878527986, 0.46028006537941496, -0.9483895811414086, 0.4263033727785258, 0.46854884411888414, -0.22411197565282925, -0.22181876013963414, -0.7237871594156134, 1.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.8199121582268207, 0.44769787996680405, 0.24813891263154456, 0.22976829035756546, -0.24162333204388178, -0.4426186743049232, 0.5347906413658639, -0.25441672637636326, -0.15102889633903993, -0.34133850661373516, 0.8367235419100765, 0.4416918512441593, -0.3901336283827199, 0.022682755644211405, 0.6304792606838273, 0.22895645987172347, 1.0, 0.0, 0.0, 0.0}, new double[]{0.541059443066477, 0.0734295519671491, -0.16515613607865418, 0.6066543955194328, 0.5177988165076132, 0.24672481378747482, 0.3747518123138858, -0.15685824721396155, -0.2163979041035694, 0.19841180164613537, 0.0034976161062877254, 0.07306529002913684, 0.5540708129952783, 0.21256664205840442, -0.23302696186542915, -0.02299628311671909, -0.6312978639861092, 1.0, 0.0, 0.0}, new double[]{0.563819407121102, 0.4653910509604774, 0.46763282136793294, -0.5144092914239631, 0.5443932918982667, 0.4497527019423619, 0.21657484340462652, -0.09875033890514759, -0.23064080504375747, -0.39510415944934346, 0.4928250345738006, 0.6090843004243195, -0.2972883410070384, 0.24627922745765815, 0.3764349274440289, 0.18953240519184184, -0.032537487902659624, -0.6731924063648309, 1.0, 0.0}, new double[]{0.037674681576550156, 0.9632634503348367, 0.23042129274228612, 0.9604648378866348, -0.3934188736330582, -0.5349584958362864, 0.22053698758116233, 0.6042169929804793, -0.16605117995562096, 0.11964967085605736, 0.8115608242264145, 0.34289275272594144, -0.961334398130144, -0.05509181439223857, 0.32182043765652824, 0.050943936755478154, 0.23846240734990845, 0.2641474881442394, 0.08178900340243297, 1.0}});
        assertEquals(lower, lu.lower());
    }

    
    @Test
    public void eigenTest() {
        DenseMatrix symmetric = generateSymmetric();

        Optional<DenseMatrix.EigenDecomposition> eigOpt = symmetric.eigenDecomposition();
        assertTrue(eigOpt.isPresent());
        DenseMatrix.EigenDecomposition eig = eigOpt.get();

        // check factorization
        Matrix computed = eig.eigenvectors().matrixMultiply(DenseSparseMatrix.createDiagonal(eig.eigenvalues())).matrixMultiply(eig.eigenvectors(),false,true);
        for (int i =0 ; i < symmetric.dim1; i++) {
            assertArrayEquals(symmetric.getRow(i).toArray(), computed.getRow(i).toArray(), 1e-12);
        }

        // check decomposition
        for (int i = 0; i < symmetric.dim1; i++) {
            // assert A.x = \lambda.x
            double eigenValue = eig.eigenvalues().get(i);
            DenseVector eigenVector = eig.getEigenVector(i);
            double[] output = symmetric.leftMultiply(eigenVector).toArray();
            double[] expected = eigenVector.scale(eigenValue).toArray();
            assertArrayEquals(expected,output,1e-12);
        }

        // test factorization
        testFactorization(symmetric, eig, 1e-10);
    }

    private static void testFactorization(DenseMatrix matrix, Matrix.Factorization factorization, double tolerance) {
        // check solve vector method
        DenseVector y = new DenseVector(new double[]{5, 6, 34});
        SGDVector b = factorization.solve(y);
        DenseVector output = matrix.rightMultiply(b);
        assertArrayEquals(y.toArray(), output.toArray(), tolerance);

        // check inverse method
        Matrix inv = factorization.inverse();
        double[][] identityArr = identityArr(3);
        DenseMatrix outputMatrix = matrix.matrixMultiply(inv);
        for (int i = 0; i < identityArr.length; i++) {
            assertArrayEquals(identityArr[i], outputMatrix.getRow(i).toArray(), tolerance);
        }
    }

    @Test
    public void eigenTest3() {
        DenseMatrix a = new DenseMatrix(new double[][] {new double[] {1.0, 2.0, 3.0}, new double[] {2.0, 1.0, 4.0}, new double[] {3.0, 4.0, 5.0}});
        DenseMatrix.EigenDecomposition eig = a.eigenDecomposition().get();
        assertArrayEquals(new double[] {9.079525367450728,-0.5925456084234676,-1.4869797590272613}, eig.eigenvalues().toArray());
        assertEigenvectorsEquals(new DenseMatrix(new double[][]{ new double[]{0.407376105324850,-0.905523895664965, 0.118622018150518},
                                                                 new double[]{0.484198771392754, 0.104025841064094,-0.868752078658035},
                                                                 new double[]{0.774336011426632, 0.411345473745188, 0.480831199733589}}), eig.eigenvectors());
    }

    @Test
    public void eigenTest8() {
        DenseMatrix a = generateSymmetricPositiveDefinite(8, new Random(123));
        DenseMatrix.EigenDecomposition eig = a.eigenDecomposition().get();
        assertArrayEquals(new double[]{459.76211240290263, 293.2666844177924, 216.59270408767136, 170.21352490625335, 166.78909016040728, 157.9347692793759, 143.28347730878713, 6.388464263025822}, eig.eigenvalues().toArray(), 1e-12);
        double[][] values = new double[][]{new double[]{0.13866815658629814, -0.8252358100347458, -0.5322979910418807, -0.06874839874131271, 0.05232354573084099, -0.06491424767108263, 0.06112114654771209, 0.03165594801425184}, new double[]{0.10040552986430423, -0.3376082404349135, 0.4375635736101318, -0.11901054624642458, -0.2335674596850905, 0.780423576489013, -0.08180549203857021, 0.002675631411587839}, new double[]{0.07581067854928174, -0.21073685231366956, 0.4962334686590166, -0.6395694525012203, -0.13654456793617276, -0.5017239717927986, 0.15195512526320729, 0.032928059562896524}, new double[]{0.05952863799372147, -0.053657044549330915, 0.03753211518153525, 0.024900409997130044, 0.011629984966546214, -0.046407158488371125, -0.04938822374115087, -0.9933883909062082}, new double[]{0.10148264893220066, -0.2702761476343119, 0.3993474902510659, 0.7416253629655215, -0.2078352622158741, -0.2721705343847198, 0.29563328477316864, 0.04994140515356563}, new double[]{0.07317536757652769, -0.1638812735247339, 0.19609570362266984, 0.14340131278754917, 0.14775784823370017, -0.22004923286199535, -0.9120630704169633, 0.08159498307657408}, new double[]{0.9707144758110352, 0.21915710575752975, -0.08314349908025911, -0.019147960448430115, -0.018914352152997034, 0.00948566942422636, 0.015953634844472334, 0.041253296979782646}, new double[]{0.058813248463141024, -0.09890060504236657, 0.2696217133439837, 0.02241144059690243, 0.9265760530592435, 0.10127275489512025, 0.2110211084939485, 0.015240404666016118}};
        assertEigenvectorsEquals(new DenseMatrix(values), eig.eigenvectors());
    }

    @Test
    public void eigenTest20() {
        DenseMatrix a = generateSymmetricPositiveDefinite(20, new Random(42));
        DenseMatrix.EigenDecomposition eig = a.eigenDecomposition().get();
        for (int i = 0; i < a.dim1; i++) {
            // assert A.x = \lambda.x
            double eigenValue = eig.eigenvalues().get(i);
            DenseVector eigenVector = eig.getEigenVector(i);
            double[] outpt = a.leftMultiply(eigenVector).toArray();
            double[] expected = eigenVector.scale(eigenValue).toArray();
            assertArrayEquals(expected,outpt,1e-12);
        }
        assertArrayEquals(new double[]{(607.0122035912394), (342.23158891508194), (319.4166697533473), (314.3320479150682), (277.0466868867828), (272.8066860877825), (258.25981917694133), (247.40723877187338), (229.2357092782707), (198.22007297732208), (194.0112606156516), (168.66008287924282), (154.86848777133343), (142.54341246616536), (130.89687348036423), (125.88587299805424), (30.17405842512715), (21.994327309233963), (18.44990986009159), (8.326134535013239)}, eig.eigenvalues().toArray(), 1e-12);
        double[][] values = new double[][]{new double[]{-0.26715610385542954, -0.01390478703198786, 0.1854110021241962, -0.1854662839160773, -0.16229872855019126, -0.15697766285531115, -0.7864295335461396, 0.16376563037553812, -0.32631164748717006, -0.13389230710603137, 0.1805076758436477, 0.05177064282900672, 0.02963293191284903, 0.031168113842195717, -0.01300327521337336, 0.025592540782747024, 0.030883988897009274, 0.0038945894037057784, 0.008438603064424666, -0.0015324664887223986}, new double[]{-0.21646536805125438, -0.013881077062626332, -0.01884624743066648, -0.007719582182124222, -0.15538791543439368, 0.04399647639053526, 0.01326559640729243, 0.04899467891124426, -0.08245118897883152, 0.19811793138915004, -0.1439106414069904, -0.7424814246902505, -0.48003665485603947, 0.2438294393508589, 0.0562373627082841, -0.08755214897319864, 0.0567217067754628, -0.03615656676219461, 0.015398062015907793, -0.01902975054850682}, new double[]{-0.1066921727252573, 0.00752781363333148, 0.01026475624523088, -0.031229671834569855, 0.0012686806850796797, 0.021181797375686102, 0.02021525202531208, -0.02837784114210916, -0.019720486930612822, 0.05227451308196039, -0.012161251995479868, -0.02483240450746341, 0.016905728457830504, -0.004130760055860076, -0.01904130742958451, -0.04416933391011066, -0.4074617231152516, 0.8676002055968268, 0.17512107945483427, -0.17288787786512805}, new double[]{-0.33920371511087805, -0.46691663525020594, -0.09031745999231369, 0.6321417470530342, -0.020930272891643172, 0.43592985933677975, -0.1452977571167301, 0.004938893921918626, 0.12863584565123845, -0.14245594866051303, 0.03359117860557403, 0.0840023224659054, -0.02300917557737052, -0.01623752607387114, -0.011098541422544728, -0.0033447671673163286, 0.007461812930786365, -0.013528489929949457, 0.02808146342169226, -0.024086648088416315}, new double[]{-0.259951086830275, -0.04997448086930181, 0.6491032077238915, -0.32175219248118364, 0.38418523432722484, 0.3963171032227332, 0.18407833127986634, -0.08492585414572532, -0.08656096384235985, -0.14763563022387532, -0.09947539484383917, 0.05987364971422366, -0.05398743934010043, 0.04417092562419047, 0.029588033891042777, -0.065271962860663, 0.07067220805307384, -0.02878732331263646, 0.01801166565448024, 0.019677109132111653}, new double[]{-0.3160522302492655, 0.1425457927728826, -0.6945965698158629, -0.4412619380518735, 0.25596051294629746, 0.30979369884113817, -0.08332532009062198, -0.049365521208594, 0.0653886986505201, -0.14860384242615965, -0.014372554107587941, 0.03861282802085877, -0.014269848155164286, -0.015802221885064142, 0.029659945096216545, -0.004361717142820119, 0.022282695747436652, -0.03838055234237704, 0.012884385334153743, -0.014256056678976633}, new double[]{-0.18106927259731728, 0.028748825007578232, 0.04337755020686359, -0.021238387457528943, -0.03847239877274511, 0.011325817965529623, 0.06589137341041316, -0.00047616346151320524, -0.00965129771873849, 0.14532583048801603, -0.10248137823354767, -0.036481870869998274, -8.539265679888359e-05, -0.11504227993486503, -0.17514409047608884, 0.9351610289393385, 0.049619156156582724, 0.021736153494524058, -0.018600022686212497, -0.05780589091592112}, new double[]{-0.15591504015606542, -0.03953307882230713, 0.03301774964306644, -0.021098871447422937, 0.01782536479691528, -0.0035645450588703362, -0.02719654267535835, -0.036528215244615365, 0.10382343154094024, 0.08032026638691166, -0.12974686778592456, -0.44442381390623015, 0.8179812555384617, 0.0383787019925684, -0.22912576306606652, -0.1088944425937944, 0.019102109645195625, -0.045001745479443675, -0.03140350863718813, 0.017645434601391112}, new double[]{-0.1754400162655663, 0.01711452604601059, 0.03965138273897645, -0.030375332105871844, -0.09746918771877507, -0.029798619073213253, 0.018230685423737642, -0.023368576984577207, -0.026283884585206595, 0.1526486186907142, -0.06565525432286132, -0.08417847146265778, -0.10532587646752077, -0.9236993369000231, -0.06003477384997813, -0.20743602856009283, 0.08907302697233448, -0.008322587362255062, -0.022097477444714853, -0.01838009751552626}, new double[]{-0.3429134201976233, 0.795859467843955, 0.0903851269467892, 0.42246126697206393, 0.11051462059291979, -0.08292327038947497, 0.008279250728023886, -0.013891522199128592, 0.10806934045589865, -0.11719353164345055, 0.05599440099466047, 0.03489598840817147, -0.01960623645570973, 0.05539124877786237, -0.040244824792618886, -0.05714197672159361, 0.06762349645206606, 0.0057802638559934115, 0.013432301682123959, 0.01154118392104854}, new double[]{-0.2410770916621225, -0.18877168116635618, 0.05365413952935207, -0.1097097232975626, -0.08063525762356431, -0.40326918223039254, -0.04471858180417749, -0.7253429627986044, 0.37956281631354716, -0.1374996707499872, 0.07298202701198002, 0.026706571009866727, -0.12001278348116623, 0.06892953229099162, -0.04307738212372077, -0.008074525719773884, 0.05648994551633868, 0.0029953794744891665, 0.014857944097380246, 0.023918496469108614}, new double[]{-0.10327213057795478, 0.018867378998055854, 0.031802369587523194, -0.004460665569963404, 0.011462906328509775, -0.04813267037004528, 0.01314423728214291, -0.014666799062726915, -0.01235191188621735, 0.08459453112224846, -0.006622633724362204, 0.009127643356117878, 0.005184779688786686, -0.04825899579638822, 0.03829236978987259, 0.026064083810448932, -0.674375822967597, -0.4376357897852843, 0.569335977575442, 0.05218083896969409}, new double[]{-0.2606735378193528, -0.003758233708587899, 0.05419125719059475, -0.2248451448955618, -0.65236316674233, 0.008383753171361743, 0.35640003262099895, 0.3618580018584963, 0.2158028483568182, -0.3457362410337, 0.05596705513192054, 0.11503286856883942, 0.04174780431192078, 0.05793052973521669, -0.04438856158062291, -0.046584819472861555, 0.0130568010141935, -0.019017023547427202, 0.032082592808699316, -0.04077916554753696}, new double[]{-0.22805448182235571, -0.018505801239088612, 0.02003690469103533, -0.08667775157176884, -0.046736601319101165, 0.10017515395417752, 0.10835233294222103, 0.02107044575192801, 0.06062673398714634, 0.6992915201740049, 0.5898469641747583, 0.18523582853613746, 0.0381212371745189, 0.13432133571942126, -0.05566996531799591, -0.09171998076376336, 0.09231646788289002, -0.019494880786832258, -0.021149822314972977, 0.007601678513472485}, new double[]{-0.08432944212460246, -0.018377177274517965, -0.020684304470545103, 0.0005125472758766888, -0.024502940800675967, -0.0013914454767143362, 0.03059283209237869, 0.034702131584703415, -0.022772161689467423, 0.0011657672355060958, -0.0148812638752305, -0.002598381845480636, -0.030052756685455772, -0.017280138843647857, -0.01591330228238729, 0.03539313031899835, -0.1933885285046404, 0.13359082373558595, -0.2278019764038912, 0.9374353567274204}, new double[]{-0.1164379288702311, 0.004490960089059218, 0.03185588853917711, -0.0021963858491475747, 0.0213529609396527, -0.03376520623626567, 0.012872677649063656, -0.007495895385014294, 0.004820917932928765, 0.011213906726055664, -0.002696263721788023, 0.013711583567772936, -0.030878577335419606, 0.0026984846273555066, 0.036668884296995795, -0.018462398128311096, -0.535096101610618, -0.16706542306989128, -0.7661932201791547, -0.2816083020712854}, new double[]{-0.23436896326134618, -0.1041723056459737, -0.1690436052266547, 0.12406587574627122, -0.02373498749671781, -0.2050166538149868, 0.3889604947272082, -0.21274058157455994, -0.777598272090964, -0.13129020674110803, 0.08456306139940428, 0.0518245364849499, 0.10362567787979178, 0.04667351366681789, -0.015198233337077252, -0.035249809931340914, 0.06103039843549032, -0.021591485525223473, 0.006679244668796843, -0.02736066086138905}, new double[]{-0.20359229956041464, -0.009877311296230795, -0.034642523916877166, -0.0046217003018542355, -0.12837293581337775, -0.08612684845090862, -0.07579519209698454, 0.013989794218879463, -0.03824305574811956, 0.38749003306445007, -0.7307276143578229, 0.41715892719054265, -0.011133220543453202, 0.17935057457453865, -0.06921439184757223, -0.1561046472327393, 0.07774146868774824, -0.015492185546862169, -0.0029185387766898505, -0.007448075235953968}, new double[]{-0.13527490691546717, -0.002810832081829183, 0.023038953528946464, 0.018015972940972555, -0.04320812223613813, -0.06861644689187882, 0.01776645097549279, -0.0009270836701030808, 0.05221468669802647, 0.08548070887491509, -0.0479150112971065, -0.02858434807813084, 0.21904454141202281, -0.05448484093769856, 0.942966943751532, 0.11783109434024673, 0.07141093203398967, 0.03875927438266021, -0.004812479447754362, 0.012435093376353554}, new double[]{-0.24147461042496196, -0.2738472465888555, -0.03891671477744992, 0.023818955522287515, 0.50802230435199, -0.5400427606883157, 0.11889559592832631, 0.5037751747256916, 0.17583773856009446, -0.04341988104559377, 0.022034850426857166, -0.009191750056862291, -0.07652751987417403, -0.00019623152598396564, -0.04280071917815721, -0.01851662493389959, 0.06051047939350801, 0.024914400680777896, 0.029690601833618248, -0.019701443530991528}};
        assertEigenvectorsEquals(new DenseMatrix(values), eig.eigenvectors());
    }

    /**
     * if you generate the expected values using another library, the eigenvectors (i.e. the columns)
     * may have the opposite sign as what is produced by DenseMatrix.  So, we will negate a column if
     * the first value of the expected eigenvector is the negative of the first value of the actual
     * eigenvector.   
     * @param expectedDM
     * @param actualDM
     */
    private void assertEigenvectorsEquals(DenseMatrix expectedDM, DenseMatrix actualDM) {
        assertEquals(expectedDM.dim1, actualDM.dim1, "dim1 differs");
        assertEquals(expectedDM.dim2, actualDM.dim2, "dim2 differs");
        assertArrayEquals(expectedDM.getShape(), actualDM.getShape(), "shape differs");
        
        // loop over column indices (i.e. we are going to test eigenvectors one at a time) 
        // and see if we need to negate the actual eigenvector if it differs by sign from the actual
        for(int j=0; j<expectedDM.dim2; j++) {
            boolean negate = false;
            //if their signs are different, then their sum will come out to near zero
            if(Math.abs(expectedDM.get(0,j) + actualDM.get(0,j)) < 1e-12) {
                negate = true;
            }

            for(int i=0; i<actualDM.dim1; i++) {
                if(negate) {
                    assertEquals(expectedDM.get(i,j), -actualDM.get(i,j), 1e-12, "matrix differs at ("+i+", "+j+")");
                } else {
                    assertEquals(expectedDM.get(i,j), actualDM.get(i,j), 1e-12, "matrix differs at ("+i+", "+j+")");
                }
            }
        }
    }

    @Test
    public void setColumnTest() {
        //create a 2x3 matrix
        DenseMatrix a = new DenseMatrix(new double[][] {new double[] {1.0, 2.0, 3.0}, new double[] {4.0, 5.0, 6.0}});
        assertEquals(2, a.getDimension1Size());
        assertEquals(3, a.getDimension2Size());
        a.setColumn(2, new DenseVector(new double[] {7.0, 8.0}));
        assertEquals(7.0, a.get(0,2));
        assertEquals(8.0, a.get(1,2));
        double[][] d = a.toArray();
        assertArrayEquals(new double[] {1.0, 2.0, 7.0}, d[0]);
        assertArrayEquals(new double[] {4.0, 5.0, 8.0}, d[1]);
    }
    
    @Test
    public void selectColumnsTest() {
        DenseMatrix a = generateSquareRandom(8, new Random(42));
        DenseMatrix columns = a.selectColumns(new int[] {0,5,7});
        assertEquals(8, columns.getShape()[0]);
        assertEquals(3, columns.getShape()[1]);
        assertEquals(a.getColumn(0), columns.getColumn(0));
        assertEquals(a.getColumn(5), columns.getColumn(1));
        assertEquals(a.getColumn(7), columns.getColumn(2));
    }
    
    public static String printMatrixPythonFriendly(DenseMatrix dm) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < dm.dim1; i++) {
            sb.append("[");
            for (int j = 0; j < dm.dim2; j++) {
                if (dm.get(i, j) < 0.0) {
                    sb.append(String.format("%.15f", dm.get(i, j)));
                } else {
                    sb.append(String.format(" %.15f", dm.get(i, j)));
                }
                sb.append(",");
            }
            sb.deleteCharAt(sb.length() - 1);
            sb.append("],\n");
        }
        sb.deleteCharAt(sb.length() - 1);
        sb.deleteCharAt(sb.length() - 1);
        sb.append("]");
        return sb.toString();
    }

}
