package eu.project.rapid.gvirtusfe;

import java.io.IOException;

import eu.project.rapid.ac.utils.Utils;


/**
 *
 * @author cferraro
 */
public class CudaDr_module {

  public CudaDr_module() {}

  public String cuModuleGetFunction(Frontend FE, Result res, String cmodule, String str)
      throws IOException {

    Buffer b = new Buffer();
    str = str + "\0";
    long size = str.length();
    byte[] bits = this.longToByteArray(size);

    for (int i = 0; i < bits.length; i++) {
      b.AddByte(bits[i] & 0xFF);
    }
    for (int i = 0; i < bits.length; i++) {
      b.AddByte(bits[i] & 0xFF);
    }
    for (int i = 0; i < size; i++) {
      b.AddByte(str.charAt(i));
    }

    b.Add(cmodule);

    FE.Execute("cuModuleGetFunction", b, res);
    String pointer = "";
    pointer = getHex(res, 8);
    for (int i = 0; i < res.getSizebuffer() - 8; i++) {
      res.getInput_stream().readByte();
    }

    return pointer;

  }

  public String cuModuleLoadDataEx(Frontend FE, Result res, String ptxSource, int jitNumOptions,
      int[] jitOptions, long jitOptVals0, char[] jitOptVals1, long jitOptVals2) throws IOException {
    Buffer b = new Buffer();
    b.AddInt(jitNumOptions);
    b.Add(jitOptions);
    // addStringForArgument
    ptxSource = ptxSource + "\0";
    long sizePtxSource = ptxSource.length();
    long size = sizePtxSource;
    byte[] bits = this.longToByteArray(size);

    for (int i = 0; i < bits.length; i++) {
      b.AddByte(bits[i] & 0xFF);
    }
    for (int i = 0; i < bits.length; i++) {
      b.AddByte(bits[i] & 0xFF);
    }
    for (int i = 0; i < sizePtxSource; i++)
      b.AddByte(ptxSource.charAt(i));
    b.Add(8);
    long OptVals0 = jitOptVals0;
    byte[] bit = this.longToByteArray(OptVals0);
    for (int i = 0; i < bit.length; i++) {
      b.AddByte(bit[i] & 0xFF);
    }
    b.Add(8);
    b.AddByte(160);
    b.AddByte(159);
    b.AddByte(236);
    b.AddByte(1);
    b.AddByte(0);
    b.AddByte(0);
    b.AddByte(0);
    b.AddByte(0);

    b.Add(8);
    long OptVals2 = jitOptVals2;
    byte[] bit2 = this.longToByteArray(OptVals2);
    for (int i = 0; i < bit.length; i++) {
      b.AddByte(bit2[i] & 0xFF);
    }

    FE.Execute("cuModuleLoadDataEx", b, res);
    String pointer = "";
    pointer = getHex(res, 8);
    for (int i = 0; i < res.getSizebuffer() - 8; i++)
      res.getInput_stream().readByte();

    return pointer;
  }


  private String getHex(Result res, int size) throws IOException {

    byte[] array = new byte[size];
    for (int i = 0; i < size; i++) {
      byte bit = res.getInput_stream().readByte();
      array[i] = bit;
    }
    // Sokol
    // String hex = DatatypeConverter.printHexBinary(array);
    String hex = Utils.bytesToHex(array);
    return hex;
  }

  public byte[] longToByteArray(long value) {
    return new byte[] {(byte) value, (byte) (value >> 8), (byte) (value >> 16),
        (byte) (value >> 24), (byte) (value >> 32), (byte) (value >> 40), (byte) (value >> 48),
        (byte) (value >> 56)

    };
  }
}
