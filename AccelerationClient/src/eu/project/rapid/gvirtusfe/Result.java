/*
 * To change this license header, choose License Headers in Project Properties. To change this
 * template file, choose Tools | Templates and open the template in the editor.
 */
package eu.project.rapid.gvirtusfe;

import java.io.DataInputStream;

/**
 *
 * @author cferraro
 */
public class Result {
  int exit_code;
  String output_buffer;
  DataInputStream input_stream;
  int sizebuffer;


  public DataInputStream getInput_stream() {
    return input_stream;
  }

  public void setInput_stream(DataInputStream input_stream) {
    this.input_stream = input_stream;
  }

  public String getOutput_buffer() {
    return output_buffer;
  }

  public void setOutput_buffer(String output_buffer) {
    this.output_buffer = output_buffer;
  }

  public Result() {
    this.output_buffer = "";
  }

  public int getExit_code() {
    return exit_code;
  }

  public void setExit_code(int exit_code) {
    this.exit_code = exit_code;
  }

  void setSizeBuffer(int sizes) {
    this.sizebuffer = sizes;
  }

  public int getSizebuffer() {
    return sizebuffer;
  }

}
