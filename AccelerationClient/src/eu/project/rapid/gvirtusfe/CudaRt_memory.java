/*
 * To change this license header, choose License Headers in Project Properties. To change this
 * template file, choose Tools | Templates and open the template in the editor.
 */
package eu.project.rapid.gvirtusfe;

import java.io.IOException;

import eu.project.rapid.ac.DFE;
import eu.project.rapid.ac.utils.Utils;

/**
 *
 * @author cferraro
 */
public class CudaRt_memory {

  private GVirtusFrontend gvfe;

  public CudaRt_memory(DFE dfe) {
    this.gvfe = dfe.getGvirtusFrontend();
  }

  public String cudaMalloc(Result res, long size) throws IOException {

    Buffer b = new Buffer();
    b.Add((int) size);
    String pointer = "";
    gvfe.Execute("cudaMalloc", b, res);
    pointer = getHex(res, 8);
    return pointer;
  }

  public void cudaMemcpy(Result res, String dst, float[] src, int count, int kind)
      throws IOException {

    Buffer b = new Buffer();
    b.Add(dst);
    b.Add(count * 4);
    b.Add(src);
    b.Add(count * 4);
    b.AddInt(kind);
    gvfe.Execute("cudaMemcpy", b, res);
  }

  private String getHex(Result res, int size) throws IOException {

    byte[] array = new byte[size];
    for (int i = 0; i < size; i++) {
      byte bit = res.getInput_stream().readByte();
      array[i] = bit;
    }

    // Sokol: DatatypeConverter does not exist in Android.
    // String hex = DatatypeConverter.printHexBinary(array);
    String hex = Utils.bytesToHex(array);

    System.out.println(hex);
    return hex;
  }
}
