/*
 * To change this license header, choose License Headers in Project Properties. To change this
 * template file, choose Tools | Templates and open the template in the editor.
 */
package eu.project.rapid.gvirtusfe;

import eu.project.rapid.ac.utils.Utils;

/**
 *
 * @author cferraro
 */
public class Buffer {

  String mpBuffer;
  int mLenght;
  int mBackOffset;
  int mOffset;

  public Buffer() {
    mpBuffer = "";
    mLenght = 0;
    mBackOffset = 0;
    mOffset = 0;
  }



  public void AddPointerNull() {
    byte[] bites = {(byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0};
    // Sokol: DatatypeConverter does not exist in Android.
    // mpBuffer += javax.xml.bind.DatatypeConverter.printHexBinary(bites);
    mpBuffer += Utils.bytesToHex(bites);
    mLenght += Long.SIZE / 8;
    mBackOffset = mLenght;
  }

  public void Add(int item) {
    byte[] bites =
        {(byte) item, (byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0};
    mpBuffer += Utils.bytesToHex(bites);
    mLenght += Integer.SIZE / 8;
    mBackOffset = mLenght;
  }

  public void Add(long item) {
    byte[] bites =
        {(byte) item, (byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0};
    mpBuffer += Utils.bytesToHex(bites);
    mLenght += Long.SIZE / 8;
    mBackOffset = mLenght;
  }

  public void Add(String item) {
    // Sokol
    // byte[] bites = DatatypeConverter.parseHexBinary(item);
    byte[] bites = Utils.hexToBytes(item);
    mpBuffer += Utils.bytesToHex(bites);
    mLenght += bites.length;
    mBackOffset = mLenght;
  }

  public void Add(float[] item) {
    // byte bites[] = new byte[item.length*4];
    for (int i = 0; i < item.length; i++) {
      String s = String.format("%8s", Integer.toHexString(Float.floatToRawIntBits(item[i])))
          .replace(' ', '0');
      StringBuilder out2 = new StringBuilder();
      String ss = "";
      for (int j = s.length() - 1; j > 0; j -= 2) {
        String str = s.substring(j - 1, j + 1);
        out2.append(str);
        ss = out2.toString();
      }
      mpBuffer += ss;
    }
    // mpBuffer+=Utils.bytesToHex(bites);
    mLenght += Long.SIZE / 8 * item.length;
    mBackOffset = mLenght;
  }

  public void Add(int[] item) {
    // byte bites[] = new byte[item.length*4];
    Add(item.length * 4);


    for (int i = 0; i < item.length; i++) {
      AddInt(item[i]);
    }
    mLenght += Integer.SIZE / 8 * item.length;
    mBackOffset = mLenght;
  }

  public void AddInt(int item) {
    byte[] bites = {(byte) item, (byte) 0, (byte) 0, (byte) 0};
    mpBuffer += Utils.bytesToHex(bites);
    mLenght += Integer.SIZE / 8;
    mBackOffset = mLenght;
  }

  public void AddPointer(int item) {
    byte[] bites = {(byte) item, (byte) 0, (byte) 0, (byte) 0};
    int size = (Integer.SIZE / 8);
    this.Add(size);
    mpBuffer += Utils.bytesToHex(bites);
    mLenght += size;
    mBackOffset = mLenght;
  }

  public String GetString() {
    return mpBuffer;
  }

  public long Size() {
    return mpBuffer.length();
  }

  void AddStruct(CudaRt_device.cudaDeviceProp struct) {
    byte[] bites = new byte[640];
    bites[0] = (byte) 0x78;
    bites[1] = (byte) 0x02;
    for (int i = 2; i < 640; i++) {
      bites[i] = (byte) 0;

    }
    mpBuffer += Utils.bytesToHex(bites);
    mLenght += bites.length;
    mBackOffset = mLenght;


  }

  void AddByte(int i) {
    byte[] bites = new byte[1];
    bites[0] = (byte) i;
    mpBuffer += Utils.bytesToHex(bites);
    mLenght += bites.length;
    mBackOffset = mLenght;
  }

}
