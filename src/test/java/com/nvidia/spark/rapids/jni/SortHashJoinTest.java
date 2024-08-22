package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.GatherMap;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;
public class SortHashJoinTest {
    @Test
    void testSortHashJoin() {
        Table.TestBuilder tb1 = new Table.TestBuilder();
        tb1.column(1, 2, 3, 4, 5);

        Table.TestBuilder tb2 = new Table.TestBuilder();
        tb2.column(1, 2, 3, 4, 5);

        Table left = tb1.build();
        Table right = tb2.build();

        GatherMap[] map = BucketChainHashJoin.innerJoinGatherMaps(left, right,  true);
    }
}
