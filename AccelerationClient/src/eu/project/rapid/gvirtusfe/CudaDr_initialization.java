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
public class CudaDr_initialization {

  public CudaDr_initialization() {}

  public int cuInit(Frontend fe, Result res, int flags) throws IOException {
    Buffer b = new Buffer();
    b.AddInt(flags);
    fe.Execute("cuInit", b, res);
    return 0;
  }


}
