/*******************************************************************************
 * Copyright (C) 2015, 2016 RAPID EU Project
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *******************************************************************************/
package eu.project.rapid.as;

import java.io.IOException;
import java.net.Socket;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;

import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLServerSocket;
import javax.net.ssl.SSLServerSocketFactory;

import android.content.Context;
import android.util.Log;
import eu.project.rapid.common.Configuration;

/**
 * The thread that listens for new clients (phones or other clones) to connect using SSL.
 *
 */
public class CloneSslThread implements Runnable {

  private static final String TAG = "CloneSslThread";

  private Context context;
  private eu.project.rapid.common.Configuration config; // The configurations read from the file or
                                                        // got form the Manager

  public CloneSslThread(Context context, Configuration config) {
    this.context = context;
    this.config = config;
  }

  @Override
  public void run() {

    Log.i(TAG, "CloneSslThread started");

    try {
      SSLContext sslContext = SSLContext.getInstance("SSL");
      sslContext.init(config.getKmf().getKeyManagers(), null, null);

      SSLServerSocketFactory factory = (SSLServerSocketFactory) sslContext.getServerSocketFactory();
      Log.i(TAG, "factory created");

      SSLServerSocket serverSocket =
          (SSLServerSocket) factory.createServerSocket(config.getSslClonePort());
      Log.i(TAG, "server socket created");

      // If we want also the client to authenticate himself
      // serverSocket.setNeedClientAuth(true); // default is false

      while (true) {
        // Log.i(TAG, "Saved session IDs: ");
        // Enumeration<byte[]> sessionIDs = sslContext.getServerSessionContext().getIds();
        // while(sessionIDs.hasMoreElements()) {
        // Log.i(TAG, "ID: " + RapidUtils.bytesToHex(sessionIDs.nextElement()));
        // }
        // Log.i(TAG, "");
        //
        Socket clientSocket = serverSocket.accept();
        Log.i(TAG, "New client connected using SSL");
        new ClientHandler(clientSocket, context, config);
      }

    } catch (IOException e1) {
      e1.printStackTrace();
    } catch (NoSuchAlgorithmException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } catch (KeyManagementException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }
}
