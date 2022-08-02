package com.nvidia.spark.rapids.jni;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Base64;

import org.junit.jupiter.api.Test;

import ai.rapids.cudf.BufferType;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Table;


public class CudaIPCTest {

  void createCppProcess(byte[] handle) {
    ProcessBuilder builder = new ProcessBuilder("/home/bobwang/github/learning/cpp/cmake-build-debug/cuda.main", new String(handle));
    try {
      Process p = builder.start();
      BufferedReader br=new BufferedReader(
          new InputStreamReader(
              p.getInputStream()));
      String line;
      while((line=br.readLine())!=null){
        System.out.println(line);
      }
    } catch (IOException e) {
      System.out.println("00000 " + e);
      throw new RuntimeException(e);
    }
  }

  @Test
  void test() throws InterruptedException, IOException {

    Table table = Table.readParquet(new File("/home/bobwang/tmp/cuda_ipc/person.parquet"));
    ColumnVector cv;
    cv = table.getColumn(0);
    HostColumnVector hcv = cv.copyToHost();
    for (int i = 0; i < hcv.getRowCount(); i++) {
      System.out.print(hcv.getInt(i) + " ");
    }
    System.out.println();
//    cv = ColumnVector.fromInts(11, 32, 3, 4, 5, 6);
    System.out.println("ptr is: " + String.format("0x%08x", cv.getDeviceBufferFor(BufferType.DATA).getAddress()));
    byte[] handle = CudaIPC.getCudaIpcMemHandle(cv.getDeviceBufferFor(BufferType.DATA).getAddress());
    for (int i = 0; i < handle.length; i++) {
      System.out.print(handle[i] + " ");
    }
    System.out.println();

    File file = new File("/tmp/cuda.ipc");
    if (file.exists()) {
      file.delete();
    }

    FileOutputStream fileOutputStream = new FileOutputStream(file);
    fileOutputStream.write(handle);
    fileOutputStream.close();

    File file1 = new File("/tmp/ipc_handle_encoded.txt");
    if (file1.exists()) {
      file1.delete();
    }

    FileOutputStream fileOutputStream1 = new FileOutputStream(file1);
    fileOutputStream1.write("hello".getBytes());
    long ptr = cv.getDeviceBufferFor(BufferType.DATA).getAddress();

    FileWriter fileWriter = new FileWriter(file1);
    String cudaIpcInfo = Base64.getEncoder().encodeToString(handle);
    fileWriter.write(cudaIpcInfo);
    fileWriter.flush();
    fileWriter.close();
    System.out.println("ptr is " + ptr);
    Thread.sleep(1000000000);
  }
}
