/*
 * To change this license header, choose License Headers in Project Properties. To change this
 * template file, choose Tools | Templates and open the template in the editor.
 */
package eu.project.rapid.gvirtusfe;

import java.io.IOException;

/**
 *
 * @author cferraro
 */
public class CudaDr_device {

  public CudaDr_device() {}

  public int cuDeviceGet(Frontend fe, Result res, int devID) throws IOException {

    Buffer b = new Buffer();
    b.AddPointer(0);
    b.AddInt(devID);
    String outputbuffer = "";
    StringBuilder output = new StringBuilder();
    fe.Execute("cuDeviceGet", b, res);
    int sizeType = res.getInput_stream().readByte();
    for (int i = 0; i < 7; i++)
      res.getInput_stream().readByte();
    for (int i = 0; i < sizeType; i++) {
      if (i == 0 || i == 1) {
        byte bb = res.getInput_stream().readByte();
        outputbuffer += Integer.toHexString(bb & 0xFF);
      } else
        res.getInput_stream().readByte();
    }
    StringBuilder out2 = new StringBuilder();
    if (outputbuffer.length() > 2) {
      for (int i = 0; i < outputbuffer.length() - 1; i += 2) {
        String str = outputbuffer.substring(i, i + 2);
        out2.insert(0, str);
      }
      outputbuffer = String.valueOf(Integer.parseInt(out2.toString(), 16));
    }
    return Integer.valueOf(outputbuffer);

  }


  public String cuDeviceGetName(Frontend fe, Result res, int len, int dev) throws IOException {

    Buffer b = new Buffer();
    b.AddByte(1);
    for (int i = 0; i < 8; i++)
      b.AddByte(0);
    b.AddByte(1);
    for (int i = 0; i < 7; i++)
      b.AddByte(0);
    b.AddInt(len);
    b.AddInt(dev);


    String outbuffer = "";
    StringBuilder output = new StringBuilder();
    fe.Execute("cuDeviceGetName", b, res);
    int sizeType = res.getInput_stream().readByte();

    for (int i = 0; i < 7; i++)
      res.getInput_stream().readByte();
    res.getInput_stream().readByte();

    for (int i = 0; i < 7; i++)
      res.getInput_stream().readByte();


    for (int i = 0; i < sizeType; i++) {
      byte bit = res.getInput_stream().readByte();
      outbuffer += Integer.toHexString(bit);
    }
    for (int i = 0; i < outbuffer.length() - 1; i += 2) {
      String str = outbuffer.substring(i, i + 2);
      output.append((char) Integer.parseInt(str, 16));

    }
    return output.toString();

  }


  public int cuDeviceGetCount(Frontend fe, Result res) throws IOException {

    Buffer b = new Buffer();
    b.AddPointer(0);
    String outputbuffer = "";
    fe.Execute("cuDeviceGetCount", b, res);
    int sizeType = res.getInput_stream().readByte();
    for (int i = 0; i < 7; i++)
      res.getInput_stream().readByte();
    for (int i = 0; i < sizeType; i++) {
      if (i == 0) {
        byte bb = res.getInput_stream().readByte();
        outputbuffer += Integer.toHexString(bb & 0xFF);
      } else
        res.getInput_stream().readByte();
    }
    StringBuilder out2 = new StringBuilder();
    if (outputbuffer.length() > 2) {
      for (int i = 0; i < outputbuffer.length() - 1; i += 2) {
        String str = outputbuffer.substring(i, i + 2);
        out2.insert(0, str);
      }
      outputbuffer = String.valueOf(Integer.parseInt(out2.toString(), 16));
    }

    return Integer.valueOf(outputbuffer);

  }

  public int[] cuDeviceComputeCapability(Frontend fe, Result res, int device) throws IOException {

    Buffer b = new Buffer();
    b.AddPointer(0);
    b.AddPointer(0);
    b.AddInt(device);
    String outputbuffer = "";
    fe.Execute("cuDeviceComputeCapability", b, res);
    int sizeType = res.getInput_stream().readByte();
    for (int i = 0; i < 7; i++)
      res.getInput_stream().readByte();
    for (int i = 0; i < sizeType; i++) {
      if (i == 0) {
        byte bb = res.getInput_stream().readByte();
        outputbuffer += Integer.toHexString(bb & 0xFF);
      } else
        res.getInput_stream().readByte();
    }
    StringBuilder out2 = new StringBuilder();
    if (outputbuffer.length() > 2) {
      for (int i = 0; i < outputbuffer.length() - 1; i += 2) {
        String str = outputbuffer.substring(i, i + 2);
        out2.insert(0, str);
      }
      outputbuffer = String.valueOf(Integer.parseInt(out2.toString(), 16));
    }

    int[] majorminor = new int[2];

    majorminor[0] = Integer.valueOf(outputbuffer);
    outputbuffer = "";
    sizeType = res.getInput_stream().readByte();
    for (int i = 0; i < 7; i++)
      res.getInput_stream().readByte();
    for (int i = 0; i < sizeType; i++) {
      if (i == 0) {
        byte bb = res.getInput_stream().readByte();
        outputbuffer += Integer.toHexString(bb & 0xFF);
      } else
        res.getInput_stream().readByte();
    }
    StringBuilder out3 = new StringBuilder();
    if (outputbuffer.length() > 2) {
      for (int i = 0; i < outputbuffer.length() - 1; i += 2) {
        String str = outputbuffer.substring(i, i + 2);
        out3.insert(0, str);
      }
      outputbuffer = String.valueOf(Integer.parseInt(out3.toString(), 16));
    }
    majorminor[1] = Integer.valueOf(outputbuffer);
    return majorminor;

  }

  public int cuDeviceGetAttribute(Frontend fe, Result res, int attribute, int device)
      throws IOException {
    Buffer b = new Buffer();
    b.AddPointer(0);
    b.AddInt(attribute);
    b.AddInt(device);
    String outputbuffer = "";
    fe.Execute("cuDeviceGetAttribute", b, res);
    int sizeType = res.getInput_stream().readByte();
    for (int i = 0; i < 7; i++)
      res.getInput_stream().readByte();
    for (int i = 0; i < sizeType; i++) {
      if (i == 0) {
        byte bb = res.getInput_stream().readByte();
        outputbuffer += Integer.toHexString(bb & 0xFF);
      } else
        res.getInput_stream().readByte();
    }
    StringBuilder out2 = new StringBuilder();
    if (outputbuffer.length() > 2) {
      for (int i = 0; i < outputbuffer.length() - 1; i += 2) {
        String str = outputbuffer.substring(i, i + 2);
        out2.insert(0, str);
      }
      outputbuffer = String.valueOf(Integer.parseInt(out2.toString(), 16));
    }

    return Integer.valueOf(outputbuffer);

  }

  public long cuDeviceTotalMem(Frontend fe, Result res, int dev) throws IOException {

    Buffer b = new Buffer();
    b.AddByte(8);
    for (int i = 0; i < 16; i++)
      b.AddByte(0);
    b.AddInt(dev);
    fe.Execute("cuDeviceTotalMem", b, res);
    for (int i = 0; i < 8; i++)
      res.getInput_stream().readByte();
    long x = getLong(res);
    return x;

  }

  private long getLong(Result res) throws IOException {

    StringBuilder output = new StringBuilder();
    for (int i = 0; i < 8; i++) {
      byte bit = res.getInput_stream().readByte();
      int a = bit & 0xFF;
      if (a == 0) {
        output.insert(0, Integer.toHexString(a));
        output.insert(0, Integer.toHexString(a));
      } else {
        output.insert(0, Integer.toHexString(a));
      }
    }
    return Long.parseLong(output.toString(), 16);
  }
}
