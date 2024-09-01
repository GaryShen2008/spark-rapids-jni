package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.shadow.com.univocity.parsers.csv.CsvWriter;

import java.io.File;
import java.io.IOException;

public class SortHashJoinTest {

    @Test
    void testSortHashJoin() {
        Table.TestBuilder tb1 = new Table.TestBuilder();
        tb1.column(1, 2, 3, 4);

        Table.TestBuilder tb2 = new Table.TestBuilder();
        tb2.column(1, 2, 3, 6, 8, 3, 3, 3, 2);

        Table left = tb1.build();
        Table right = tb2.build();

        // Measure execution time
        long startTime = System.nanoTime();
        GatherMap[] map = BucketChainHashJoin.innerJoinGatherMaps(left, right, true);
        long endTime = System.nanoTime();

        // Calculate execution time
        long durationNanos = endTime - startTime;
        double durationMillis = durationNanos / 1_000_000.0;

        // Print metrics
        System.out.println("Execution time: " + durationMillis + " ms");

    }

    @Test
    void testCUDFHASHJoin() {
        Table.TestBuilder tb1 = new Table.TestBuilder();
        tb1.column(1, 2, 3, 4);

        Table.TestBuilder tb2 = new Table.TestBuilder();
        tb2.column(1, 2, 3, 6, 8, 3, 3, 3, 2);

        Table left = tb1.build();
        Table right = tb2.build();

        // Measure execution time
        long startTime = System.nanoTime();
        GatherMap[] map = left.innerJoinGatherMaps(right, true);
        long endTime = System.nanoTime();

        // Calculate execution time
        long durationNanos = endTime - startTime;
        double durationMillis = durationNanos / 1_000_000.0;

        // Print metrics
        System.out.println("Execution time: " + durationMillis + " ms");
    }

    @Test
    void readCSV() {
        Schema schema = Schema.builder()
                .column(DType.INT32, "key")
                .column(DType.INT32, "col1")
                .build();
        CSVOptions opts = CSVOptions.builder()
                .includeColumn("key")
                .includeColumn("col1")
                .hasHeader()
                .build();
        File file = new File("/home/fejiang/IdeaProjects/csv/tabler.csv");
        Table table = Table.readCSV(schema, opts, file);
        System.out.println(table.getRowCount());
    }

    @Test
    void readCSVCudfJoin() {
        Schema schema = Schema.builder()
                .column(DType.INT32, "key")
                .column(DType.INT32, "col1")
                .build();
        CSVOptions opts = CSVOptions.builder()
                .includeColumn("key")
                //.includeColumn("col1")
                .hasHeader()
                .build();
        File file1 = new File("/home/fejiang/IdeaProjects/csv/tabler.csv");
        Table table1 = Table.readCSV(schema, opts, file1);
        System.out.println(table1.getRowCount());
        Table table3 = Table.concatenate(table1, table1);

        File file2 = new File("/home/fejiang/IdeaProjects/csv/tables.csv");
        Table table2 = Table.readCSV(schema, opts, file2);
        System.out.println(table2.getRowCount());
        Table table4 = Table.concatenate(table2, table2);

        // Measure execution time
        long startTime = System.nanoTime();
        GatherMap[] map = table3.innerJoinGatherMaps(table4, true);
        long endTime = System.nanoTime();

        // Calculate execution time
        long durationNanos = endTime - startTime;
        double durationMillis = durationNanos / 1_000_000.0;

        // Print metrics
        System.out.println("Execution time: " + durationMillis + " ms");
        System.out.println(map[0].getRowCount());
    }

    @Test
    void readCSVPartitionHashJoin() {

        Schema schema = Schema.builder()
                .column(DType.INT32, "key")
                .column(DType.INT32, "col1")
                .build();
        CSVOptions opts = CSVOptions.builder()
                .includeColumn("key")
                //.includeColumn("col1")
                .hasHeader()
                .build();
        File file1 = new File("/home/fejiang/IdeaProjects/csv/tabler.csv");
        Table table1 = Table.readCSV(schema, opts, file1);
        System.out.println(table1.getRowCount());
        Table table3 = Table.concatenate(table1, table1);

        File file2 = new File("/home/fejiang/IdeaProjects/csv/tables.csv");
        Table table2 = Table.readCSV(schema, opts, file2);
        Table table4 = Table.concatenate(table2, table2);
        System.out.println(table2.getRowCount());

        // Measure execution time
        long startTime = System.nanoTime();
        GatherMap[] map = BucketChainHashJoin.innerJoinGatherMaps(table3, table4, true);
        long endTime = System.nanoTime();

        // Calculate execution time
        long durationNanos = endTime - startTime;
        double durationMillis = durationNanos / 1_000_000.0;

        // Print metrics
        System.out.println("Execution time: " + durationMillis + " ms");

        System.out.println(map[0].getRowCount());
    }

}
