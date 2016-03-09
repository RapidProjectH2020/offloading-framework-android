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
import java.net.ServerSocket;
import java.net.Socket;

import android.content.Context;
import android.util.Log;
import eu.project.rapid.common.Configuration;

/**
 * The thread that listens for new clients (phones or other clones) to connect in clear.
 *
 */
public class CloneThread implements Runnable {

  private static final String TAG = "CloneThread";

  private Context context;
  private Configuration config; // The configurations read from the file or got from the Manager

  public CloneThread(Context context, eu.project.rapid.common.Configuration config) {
    this.context = context;
    this.config = config;
  }

  @Override
  public void run() {

    try {
      ServerSocket serverSocket = new ServerSocket(config.getClonePort());
      Log.i(TAG, "CloneThread started on port " + config.getClonePort());
      while (true) {
        Socket clientSocket = serverSocket.accept();
        Log.i(TAG, "New client connected in clear");
        new ClientHandler(clientSocket, context, config);
      }
    } catch (IOException e) {
      Log.e(TAG, "IOException: " + e.getMessage());
      e.printStackTrace();
    }
  }
}
