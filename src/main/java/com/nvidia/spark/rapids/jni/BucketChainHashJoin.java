/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.GatherMap;

public class BucketChainHashJoin {
    public static GatherMap[] innerJoinGatherMaps(Table leftTable, Table rightTable, boolean compareNullsEqual ){
        if (leftTable.getNumberOfColumns() != rightTable.getNumberOfColumns()) {
            throw new IllegalArgumentException("Column count mismatch, this: " + leftTable.getNumberOfColumns() +
                    "rightKeys: " + rightTable.getNumberOfColumns());
        }
        long[] gatherMapData =
                innerJoinGatherMaps(leftTable.getNativeView(), rightTable.getNativeView(), compareNullsEqual);
        return buildJoinGatherMaps(gatherMapData);
    }

    private static GatherMap[] buildJoinGatherMaps(long[] gatherMapData) {
        long bufferSize = gatherMapData[0];
        long leftAddr = gatherMapData[1];
        long leftHandle = gatherMapData[2];
        long rightAddr = gatherMapData[3];
        long rightHandle = gatherMapData[4];
        GatherMap[] maps = new GatherMap[2];
        maps[0] = new GatherMap(DeviceMemoryBuffer.fromRmm(leftAddr, bufferSize, leftHandle));
        maps[1] = new GatherMap(DeviceMemoryBuffer.fromRmm(rightAddr, bufferSize, rightHandle));
        return maps;
    }

    private static native long[] innerJoinGatherMaps(long leftKeys, long rightKeys,
                                                     boolean compareNullsEqual) throws CudfException;
}
