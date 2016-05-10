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
public class CudaDr_context {

  GVirtusFrontend gvfe;

  public CudaDr_context(DFE dfe) {
    this.gvfe = dfe.getGvirtusFrontend();
  }

  public String cuCtxCreate(Result res, int flags, int dev) throws IOException {

    Buffer b = new Buffer();
    b.AddInt(flags);
    b.AddInt(dev);
    String outbuffer = "";
    gvfe.Execute("cuCtxCreate", b, res);
    return getHex(res, 8);
  }

  public int cuCtxDestroy(Result res, String ctx) throws IOException {

    Buffer b = new Buffer();
    b.Add(ctx);
    gvfe.Execute("cuCtxDestroy", b, res);
    return 0;
  }

  private String getHex(Result res, int size) throws IOException {

    byte[] array = new byte[size];
    for (int i = 0; i < size; i++) {
      byte bit = res.getInput_stream().readByte();
      array[i] = bit;
    }

    // Sokol: DatatypeConverter does not exist on Android
    // String hex = DatatypeConverter.printHexBinary(array);
    String hex = Utils.bytesToHex(array);

    System.out.println(hex);
    return hex;
  }

}
