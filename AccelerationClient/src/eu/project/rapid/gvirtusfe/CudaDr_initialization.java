/*
 * To change this license header, choose License Headers in Project Properties. To change this
 * template file, choose Tools | Templates and open the template in the editor.
 */
package eu.project.rapid.gvirtusfe;

import java.io.IOException;

import eu.project.rapid.ac.DFE;

/**
 *
 * @author cferraro
 */
public class CudaDr_initialization {

  GVirtusFrontend gvfe;

  public CudaDr_initialization(DFE dfe) {
    this.gvfe = dfe.getGvirtusFrontend();
  }

  public int cuInit(Result res, int flags) throws IOException {
    Buffer b = new Buffer();
    b.AddInt(flags);
    gvfe.Execute("cuInit", b, res);
    return 0;
  }


}
