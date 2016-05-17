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
public class CudaDr_execution {

  public CudaDr_execution() {}

  public void cuParamSetv(Frontend fe, Result res, String hfunc, int offset, String ptr,
      int numbytes) throws IOException {

    Buffer b = new Buffer();
    b.AddInt(offset);
    b.AddInt(numbytes);
    long sizeofp = 8;
    b.Add(sizeofp);
    b.Add(ptr);
    b.Add(hfunc);
    fe.Execute("cuParamSetv", b, res);

  }

  public void cuParamSeti(Frontend FE, Result res, String hfunc, int offset, int value)
      throws IOException {

    Buffer b = new Buffer();
    b.AddInt(offset);
    b.AddInt(value);
    b.Add(hfunc);
    FE.Execute("cuParamSeti", b, res);
  }

  public void cuParamSetSize(Frontend FE, Result res, String hfunc, int numbytes)
      throws IOException {

    Buffer b = new Buffer();
    b.AddInt(numbytes);
    b.Add(hfunc);
    FE.Execute("cuParamSetSize", b, res);
  }

  public void cuFuncSetBlockShape(Frontend FE, Result res, String hfunc, int x, int y, int z)
      throws IOException {

    Buffer b = new Buffer();
    b.AddInt(x);
    b.AddInt(y);
    b.AddInt(z);
    b.Add(hfunc);
    FE.Execute("cuFuncSetBlockShape", b, res);
  }

  public void cuFuncSetSharedSize(Frontend FE, Result res, String hfunc, int bytes)
      throws IOException {

    Buffer b = new Buffer();
    byte[] bits = this.intToByteArray(bytes);
    for (int i = 0; i < bits.length; i++) {
      b.AddByte(bits[i] & 0xFF);
    }
    b.Add(hfunc);
    FE.Execute("cuFuncSetSharedSize", b, res);
  }

  public void cuLaunchGrid(Frontend FE, Result res, String hfunc, int grid_width, int grid_height)
      throws IOException {

    Buffer b = new Buffer();
    b.AddInt(grid_width);
    b.AddInt(grid_height);
    b.Add(hfunc);
    FE.Execute("cuLaunchGrid", b, res);
  }

  public byte[] intToByteArray(int value) {
    return new byte[] {(byte) value, (byte) (value >> 8), (byte) (value >> 16),
        (byte) (value >> 24)};
  }
}
