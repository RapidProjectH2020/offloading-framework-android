/*
 * To change this license header, choose License Headers in Project Properties. To change this
 * template file, choose Tools | Templates and open the template in the editor.
 */
package eu.project.rapid.gvirtusfe;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Socket;

import android.util.Log;
import eu.project.rapid.ac.utils.Utils;

/**
 *
 * @author cferraro
 */
public final class Frontend {

  private static final String TAG = Frontend.class.getName();

  String serverIpAddress;
  int port;
  Socket socket, clientSocket;
  DataOutputStream outputStream, clientOutputStream;
  DataInputStream in, clientIn;

  public Frontend(String url, int port) {

    this.serverIpAddress = url;
    this.port = port;
    try {
      this.socket = new Socket();
      socket.connect(new InetSocketAddress(this.serverIpAddress, this.port), 10 * 1000);
      this.outputStream = new DataOutputStream(this.socket.getOutputStream());
      this.in = new DataInputStream(this.socket.getInputStream());
    } catch (IOException e) {
      Log.e(TAG, "Could not connect to GVirtuS backend server!" + e);
    }
  }

  public int Execute(String routine, Buffer input_buffer, Result res) throws IOException {

    for (int i = 0; i < routine.length(); i++)
      this.outputStream.writeByte(routine.charAt(i));
    this.outputStream.writeByte(0);
    long size = input_buffer.Size() / 2;
    byte[] bits = this.longToByteArray(size);
    for (int i = 0; i < bits.length; i++) {
      this.outputStream.write(bits[i] & 0xFF);
    }

    // byte[] bytes2 = DatatypeConverter.parseHexBinary(input_buffer.GetString());
    byte[] bytes2 = Utils.hexToBytes(input_buffer.GetString());

    for (int i = 0; i < bytes2.length; i++) {
      this.outputStream.write(bytes2[i] & 0xFF);
    }
    int message = this.in.readByte();
    this.in.readByte();
    this.in.readByte();
    this.in.readByte();
    res.setExit_code(message);
    int sizes = (int) this.in.readByte();
    res.setSizeBuffer(sizes);
    for (int i = 0; i < 7; i++)
      this.in.readByte();
    res.setInput_stream(this.in);
    return 0;
  }

  //
  // public int ExecuteMultiThread(String routine,Buffer input_buffer,Result res) throws
  // IOException{
  //
  // for (int i =0; i< routine.length();i++)this.outputStream.writeByte(routine.charAt(i));
  // this.outputStream.writeByte(0);
  // this.clientSocket = new Socket(this.serverIpAddress,9998);
  // this.clientIn = new DataInputStream(this.clientSocket.getInputStream());
  // this.clientOutputStream = new DataOutputStream(this.clientSocket.getOutputStream());
  // long size = input_buffer.Size()/2;
  // byte[] bits = this.longToByteArray(size);
  // for (int i =0; i< bits.length;i++){
  // this.clientOutputStream.write(bits[i] & 0xFF);
  // }
  //
  // byte[] bytes2 = DatatypeConverter.parseHexBinary(input_buffer.GetString());
  //
  // for (int i =0; i< bytes2.length;i++){
  // this.clientOutputStream.write(bytes2[i] & 0xFF);
  // }
  //
  // int message = this.clientIn.readByte();
  // this.clientIn.readByte();
  // this.clientIn.readByte();
  // this.clientIn.readByte();
  // res.setExit_code(message);
  // long size_buffer = this.clientIn.readByte();
  // for (int i =0 ; i< 7; i++) this.clientIn.readByte();
  // res.setInput_stream(this.clientIn);
  // return 0;
  // }
  //
  public void writeLong(DataOutputStream os, long l) throws IOException {
    os.write((byte) l);
    os.write((byte) (l >> 56));
    os.write((byte) (l >> 48));
    os.write((byte) (l >> 40));
    os.write((byte) (l >> 32));
    os.write((byte) (l >> 24));
    os.write((byte) (l >> 16));
    os.write((byte) (l >> 8));
  }

  public void writeChar(DataOutputStream os, char l) throws IOException {
    os.write((byte) l);
    os.write((byte) (l >> 56));
    os.write((byte) (l >> 48));
    os.write((byte) (l >> 40));
    os.write((byte) (l >> 32));
    os.write((byte) (l >> 24));
    os.write((byte) (l >> 16));
    os.write((byte) (l >> 8));
  }

  public char readChar(DataInputStream os) throws IOException {
    int x;
    x = os.readByte();
    x = x >> 56;
    x = os.readByte();
    x = x >> 48;
    x = os.readByte();
    x = x >> 40;
    x = os.readByte();
    x = x >> 32;
    x = os.readByte();
    x = x >> 24;
    x = os.readByte();
    x = x >> 16;
    x = os.readByte();
    x = x >> 8;
    x = os.readByte();
    return (char) x;

  }

  public void writeInt(DataOutputStream os, int l) throws IOException {
    os.write((byte) l);
    os.write((byte) (l >> 24));
    os.write((byte) (l >> 16));
    os.write((byte) (l >> 8));
  }

  public void writeHex(DataOutputStream os, long x) throws IOException {
    String hex = Integer.toHexString((int) (x));
    StringBuilder out2 = new StringBuilder();
    int scarto = 0;
    if (hex.length() > 2) {
      for (int i = hex.length() - 1; i > 0; i -= 2) {
        String str = hex.substring(i - 1, i + 1);
        out2.insert(0, str);
        os.write((byte) Integer.parseInt(out2.toString(), 16));
        scarto += 2;
      }
      if (scarto != hex.length()) {
        os.write((byte) Integer.parseInt(hex.substring(0, 1), 16));
      }
    }
    os.write((byte) (0));
    os.write((byte) (0));
    os.write((byte) (0));
    os.write((byte) (0));
    os.write((byte) (0));
    os.write((byte) (0));
  }

  public byte[] longToByteArray(long value) {
    return new byte[] {(byte) value, (byte) (value >> 8), (byte) (value >> 16),
        (byte) (value >> 24), (byte) (value >> 32), (byte) (value >> 40), (byte) (value >> 48),
        (byte) (value >> 56)


    };
  }

}
