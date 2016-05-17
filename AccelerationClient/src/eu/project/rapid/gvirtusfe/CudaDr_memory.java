/*
 * To change this license header, choose License Headers in Project Properties. To change this
 * template file, choose Tools | Templates and open the template in the editor.
 */
package eu.project.rapid.gvirtusfe;

import java.io.IOException;

import eu.project.rapid.ac.utils.Utils;

/**
 *
 * @author cferraro
 */
public class CudaDr_memory {

  public CudaDr_memory() {}

  public String cuMemAlloc(Frontend fe, Result res, long size) throws IOException {

    Buffer b = new Buffer();
    byte[] bits = this.longToByteArray(size);
    for (int i = 0; i < bits.length; i++) {
      b.AddByte(bits[i] & 0xFF);
    }
    String pointer = "";
    fe.Execute("cuMemAlloc", b, res);
    pointer = getHex(res, 8);
    return pointer;
  }


  private String getHex(Result res, int size) throws IOException {

    byte[] array = new byte[size];
    for (int i = 0; i < size; i++) {
      byte bit = res.getInput_stream().readByte();
      array[i] = bit;
    }
    // String hex = DatatypeConverter.printHexBinary(array);
    String hex = Utils.bytesToHex(array);
    return hex;
  }


  public void cuMemcpyHtoD(Frontend fe, Result res, String dst, float[] src, int count)
      throws IOException {

    Buffer b = new Buffer();
    byte[] bits = this.longToByteArray(count);
    for (int i = 0; i < bits.length; i++) {
      b.AddByte(bits[i] & 0xFF);
    }
    b.Add(dst);
    for (int i = 0; i < bits.length; i++) {
      b.AddByte(bits[i] & 0xFF);
    }
    b.Add(src);
    fe.Execute("cuMemcpyHtoD", b, res);

  }


  public float[] cuMemcpyDtoH(Frontend fe, Result res, String srcDevice, long ByteCount)
      throws IOException {

    Buffer b = new Buffer();
    b.Add(srcDevice);

    byte[] bits = this.longToByteArray(ByteCount);
    for (int i = 0; i < bits.length; i++) {
      b.AddByte(bits[i] & 0xFF);
    }
    fe.Execute("cuMemcpyDtoH", b, res);
    for (int i = 0; i <= 7; i++) {
      byte bb = res.getInput_stream().readByte();
    }
    int sizeType = 98304;
    float[] result = new float[sizeType / 4];
    for (int i = 0; i < sizeType / 4; i++) {
      result[i] = getFloat(res);
    }

    return result;

  }

  public void cuMemFree(Frontend fe, Result res, String ptr) throws IOException {
    Buffer b = new Buffer();
    b.Add(ptr);
    fe.Execute("cuMemFree", b, res);

  }


  private float getFloat(Result res) throws IOException {

    byte bytes[] = new byte[4];
    for (int i = 3; i >= 0; i--) {
      bytes[i] = res.getInput_stream().readByte();
    }
    // Sokol
    // String output = javax.xml.bind.DatatypeConverter.printHexBinary(bytes);
    String output = Utils.bytesToHex(bytes);
    Long i = Long.parseLong(output, 16);
    Float f = Float.intBitsToFloat(i.intValue());
    return f;
  }

  public byte[] longToByteArray(long value) {
    return new byte[] {(byte) value, (byte) (value >> 8), (byte) (value >> 16),
        (byte) (value >> 24), (byte) (value >> 32), (byte) (value >> 40), (byte) (value >> 48),
        (byte) (value >> 56)

    };
  }


}
