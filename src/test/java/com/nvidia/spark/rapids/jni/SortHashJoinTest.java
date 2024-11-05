package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.*;
import org.junit.jupiter.api.Test;

import java.io.File;

public class SortHashJoinTest {

//    @Test
//    void testSortHashJoin() {
//        Table.TestBuilder tb1 = new Table.TestBuilder();
//        tb1.column(1, 9, 2, 3, 4, 5, 9999);
//        tb1.column(1, 9, 2, 3, 4, 3, 9);
//        tb1.column(10, 90, 2, 3, 4, 3, 9);
//
//        Table.TestBuilder tb2 = new Table.TestBuilder();
//        tb2.column(1, 9, 2, 3, 6, 8, 4, 12, 14, 19, 99999);
//        tb2.column(1, 9, 2, 3, 6, 8, 777, 12, 14, 19, 9);
//        tb2.column(10, 90, 2, 3, 6, 8, 777, 12, 14, 19, 9);
//
//        Table left = tb1.build();
//        Table right = tb2.build();
//
//        // Measure execution time
//        long startTime = System.nanoTime();
//        GatherMap[] map = BucketChainHashJoin.innerJoinGatherMaps(left, right, true);
//        long endTime = System.nanoTime();
//
//        // Calculate execution time
//        long durationNanos = endTime - startTime;
//        double durationMillis = durationNanos / 1_000_000.0;
//
//        // Print metrics
//        System.out.println("Execution time: " + durationMillis + " ms");
//
//        //System.out.println(map[0].getRowCount());
//        System.out.println("gather map");
//        ColumnView colV1 = map[0].toColumnView(0L, (int) map[0].getRowCount());
////        HostColumnVector HCV = colV1.copyToHost();
////        System.out.println(HCV.toString());
////
//        for(int i = 0; i < (int) map[0].getRowCount(); i++){
//            System.out.print(colV1.getScalarElement(i).getInt() + " ");
//        }
//
//        System.out.println();
//
//        ColumnView colV2 = map[1].toColumnView(0L, (int) map[1].getRowCount());
//
//        for(int i = 0; i < (int) map[1].getRowCount(); i++){
//            System.out.print(colV2.getScalarElement(i).getInt() + " ");
//        }
//
//        System.out.println("\nthis is table 1 gathered");
//
//        Table result = BucketChainHashJoin.gather(left, colV1);
//
//        for (int i = 0; i < result.getNumberOfColumns(); i++) {
//            ColumnVector column = result.getColumn(i);
//            System.out.print("Column " + i + ": ");
//            for (int j = 0; j < column.getRowCount(); j++) {
//                System.out.print(column.getScalarElement(j).getInt() + " "); // Adjust the get method based on data type}
//            }
//        }
//
//        System.out.println("\nthis is table 2");
//        Table result2 = BucketChainHashJoin.gather(right, colV2);
//
//        for (int i = 0; i < result2.getNumberOfColumns(); i++) {
//            ColumnVector column = result2.getColumn(i);
//            System.out.print("Column " + i + ": ");
//            for (int j = 0; j < column.getRowCount(); j++) {
//                System.out.print(column.getScalarElement(j).getInt() + " "); // Adjust the get method based on data type}
//            }
//        }
//
//
//    }
//    @Test
//    void testCUDFHASHJoin() {
//        Table.TestBuilder tb1 = new Table.TestBuilder();
//        tb1.column(1, 9, 2, 3, 4, 3,99999);
//        tb1.column(1, 9, 2, 3, 4, 3, 9);
//        tb1.column(10, 90, 2, 3, 4, 3, 9);
//
//        Table.TestBuilder tb2 = new Table.TestBuilder();
//        tb2.column(1, 9, 2, 3, 6, 8, 4, 12, 14, 19, 99999);
//        tb2.column(1, 9, 2, 3, 6, 8, 777, 12, 14, 19, 9);
//        tb2.column(10, 90, 2, 3, 6, 8, 777, 12, 14, 19, 9);
//
//
//        Table left = tb1.build();
//        Table right = tb2.build();
//
//        // Measure execution time
//        long startTime = System.nanoTime();
//        GatherMap[] map = left.innerJoinGatherMaps(right, true);
//        long endTime = System.nanoTime();
//
//        // Calculate execution time
//        long durationNanos = endTime - startTime;
//        double durationMillis = durationNanos / 1_000_000.0;
//
//        // Print metrics
//        System.out.println("Execution time: " + durationMillis + " ms");
//
//        ColumnView colV1 = map[0].toColumnView(0L, (int) map[0].getRowCount());
//        for(int i = 0; i < (int) map[0].getRowCount(); i++){
//            System.out.print(colV1.getScalarElement(i).getInt() + " ");
//        }
//
//        System.out.println();
//
//        ColumnView colV2 = map[1].toColumnView(0L, (int) map[1].getRowCount());
//
//        for(int i = 0; i < (int) map[1].getRowCount(); i++){
//            System.out.print(colV2.getScalarElement(i).getInt() + " ");
//        }
//
//        System.out.println();
//    }
//
//    @Test
//    void readCSV() {
//        Schema schema = Schema.builder()
//                .column(DType.INT32, "key")
//                .column(DType.INT32, "col1")
//                .build();
//        CSVOptions opts = CSVOptions.builder()
//                .includeColumn("key")
//                .includeColumn("col1")
//                .hasHeader()
//                .build();
//        File file = new File("/home/fejiang/IdeaProjects/csv/tabler.csv");
//        Table table = Table.readCSV(schema, opts, file);
//        System.out.println(table.getRowCount());
//    }
//
//    @Test
//    void readCSVCudfJoin() {
//        Schema schema = Schema.builder()
//                .column(DType.INT32, "key")
//                .column(DType.INT32, "col1")
//                .build();
//        CSVOptions opts = CSVOptions.builder()
//                .includeColumn("key")
//                //.includeColumn("col1")
//                .hasHeader()
//                .build();
//        File file1 = new File("/home/fejiang/IdeaProjects/csv/tabler.csv");
//        Table table1 = Table.readCSV(schema, opts, file1);
//        System.out.println(table1.getRowCount());
//        Table table3 = Table.concatenate(table1, table1);
//
//        File file2 = new File("/home/fejiang/Documents/tabler4.csv");
//        Table table2 = Table.readCSV(schema, opts, file2);
//        Table table4 = Table.concatenate(table2, table2, table2);
//        System.out.println(table2.getRowCount());
//
//        // Measure execution time
//        long startTime = System.nanoTime();
//        GatherMap[] map = table1.innerJoinGatherMaps(table2, true);
//        long endTime = System.nanoTime();
//
//        // Calculate execution time
//        long durationNanos = endTime - startTime;
//        double durationMillis = durationNanos / 1_000_000.0;
//
//        // Print metrics
//        System.out.println("Execution time: " + durationMillis + " ms");
//        System.out.println(map[0].getRowCount());
//    }
//
//    @Test
//    void readCSVPartitionHashJoin() {
//
//        Schema schema = Schema.builder()
//                .column(DType.INT32, "key")
//                .column(DType.INT32, "col1")
//                .build();
//        CSVOptions opts = CSVOptions.builder()
//                .includeColumn("key")
//                //.includeColumn("col1")
//                .hasHeader()
//                .build();
//        File file1 = new File("/home/fejiang/IdeaProjects/csv/tabler.csv");
//        Table table1 = Table.readCSV(schema, opts, file1);
//        System.out.println(table1.getRowCount());
//        Table table3 = Table.concatenate(table1, table1);
//
//        File file2 = new File("/home/fejiang/Documents/tabler4.csv");
//        Table table2 = Table.readCSV(schema, opts, file2);
//        Table table4 = Table.concatenate(table2, table2);
//        System.out.println(table2.getRowCount());
//
//        // Measure execution time
//        long startTime = System.nanoTime();
//        GatherMap[] map = BucketChainHashJoin.innerJoinGatherMaps(table1, table2, true);
//        long endTime = System.nanoTime();
//
//        // Calculate execution time
//        long durationNanos = endTime - startTime;
//        double durationMillis = durationNanos / 1_000_000.0;
//
//        // Print metrics
//        System.out.println("Execution time: " + durationMillis + " ms");
//
//        System.out.println(map[0].getRowCount());
//    }

}
