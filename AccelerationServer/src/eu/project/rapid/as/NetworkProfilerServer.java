/*******************************************************************************
 * Copyright (C) 2015, 2016 RAPID EU Project
 *
 * This library is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either version
 * 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with this library;
 * if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301 USA
 *******************************************************************************/
package eu.project.rapid.as;

import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Random;

import android.util.Log;
import eu.project.rapid.common.Configuration;
import eu.project.rapid.common.RapidMessages;
import eu.project.rapid.common.RapidUtils;

/**
 * Listen for phone connections for measuring the data rate. The phone will send/receive some data
 * for 3 seconds.
 *
 */
public class NetworkProfilerServer implements Runnable {

  private static final String TAG = "NetworkProfilerServer";
  public static final int uid = android.os.Process.myUid();

  private Configuration config;
  private ServerSocket serverSocket;

  private long totalBytesRead;
  private long totalTimeBytesRead;
  private static final int BUFFER_SIZE = 10 * 1024;
  private byte[] buffer;

  public NetworkProfilerServer(Configuration config) {
    this.config = config;
    buffer = new byte[BUFFER_SIZE];
  }

  @Override
  public void run() {
    try {
      serverSocket = new ServerSocket(config.getClonePortBandwidthTest());
      while (true) {
        Socket clientSocket = serverSocket.accept();
        new Thread(new ClientThread(clientSocket)).start();
      }
    } catch (IOException e) {
      Log.e(TAG,
          "Could not start server on port " + config.getClonePortBandwidthTest() + " (" + e + ")");
    }
  }

  private class ClientThread implements Runnable {

    private Socket clientSocket;

    public ClientThread(Socket clientSocket) {
      this.clientSocket = clientSocket;
    }

    @Override
    public void run() {
      int request = 0;

      InputStream is = null;
      OutputStream os = null;
      DataOutputStream dos = null;
      try {
        is = clientSocket.getInputStream();
        os = clientSocket.getOutputStream();
        dos = new DataOutputStream(os);

        while (request != -1) {
          request = is.read();

          switch (request) {

            case RapidMessages.UPLOAD_FILE:
              new Thread(new Runnable() {

                @Override
                public void run() {
                  long t0 = System.nanoTime();
                  long elapsed = 0;
                  while (elapsed < 3000) {
                    try {
                      Thread.sleep(3000 - elapsed);
                    } catch (InterruptedException e1) {
                    } finally {
                      elapsed = (System.nanoTime() - t0) / 1000000;
                    }
                  }
                  RapidUtils.closeQuietly(clientSocket);
                }

              }).start();

              totalTimeBytesRead = System.nanoTime();
              totalBytesRead = 0;
              while (true) {
                totalBytesRead += is.read(buffer);
                os.write(1);
              }

            case RapidMessages.UPLOAD_FILE_RESULT:
              dos.writeLong(totalBytesRead);
              dos.writeLong(totalTimeBytesRead);
              dos.flush();
              break;

            case RapidMessages.DOWNLOAD_FILE:
              new Random().nextBytes(buffer);
              // Used for measuring the dlRate on the phone
              while (true) {
                os.write(buffer);
                is.read();
              }
          }
        }

      } catch (IOException e) {
      } finally {
        Log.i(TAG, "Client finished bandwidth measurement: " + request);

        if (request == RapidMessages.UPLOAD_FILE)
          totalTimeBytesRead = System.nanoTime() - totalTimeBytesRead;

        RapidUtils.closeQuietly(os);
        RapidUtils.closeQuietly(dos);
        RapidUtils.closeQuietly(is);
      }
    }
  }
}
