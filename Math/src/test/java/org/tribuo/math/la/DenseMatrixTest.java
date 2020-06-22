/*
 * Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
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

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

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

    public static DenseMatrix identity(int size) {
        double[][] values = new double[size][size];

        for (int i = 0; i < size; i++) {
            values[i][i] = 1.0;
        }

        return new DenseMatrix(values);
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

}
