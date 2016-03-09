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

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.lang.reflect.Array;
import java.net.InetSocketAddress;
import java.net.Socket;

import android.util.Log;
import eu.project.rapid.ac.DynamicObjectInputStream;
import eu.project.rapid.ac.ResultContainer;
import eu.project.rapid.common.Clone;
import eu.project.rapid.common.Configuration;
import eu.project.rapid.common.RapidMessages;
import eu.project.rapid.common.RapidUtils;

/**
 * The thread taking care of communication with the clone helpers
 *
 */
public class CloneHelperThread extends Thread {

  private String TAG = "ServerHelper-";
  private Configuration config;
  private Clone clone;
  private Socket mSocket;
  private OutputStream mOutStream;
  private InputStream mInStream;
  private ObjectOutputStream mObjOutStream;
  private DynamicObjectInputStream mObjInStream;

  // This id is assigned to the clone helper by the main clone.
  // It is needed for splitting the input when parallelizing a certain method (see for example
  // virusScanning).
  // To not be confused with the id that the clone has read from the config file.
  private int cloneHelperId;

  public CloneHelperThread(Configuration config, int cloneHelperId, Clone clone) {
    this.config = config;
    this.clone = clone;
    this.cloneHelperId = cloneHelperId;
    TAG = TAG + this.cloneHelperId;
  }

  @Override
  public void run() {

    try {

      // Try to connect to the clone helper.
      // If it is not possible to connect stop running.
      if (!establishConnection()) {
        // Try to close created sockets
        closeConnection();
        return;
      }

      // Send the cloneId to this clone.
      mOutStream.write(RapidMessages.CLONE_ID_SEND);
      mOutStream.write(cloneHelperId);

      while (true) {

        synchronized (ClientHandler.nrClonesReady) {
          Log.d(TAG, "Server Helpers started so far: " + ClientHandler.nrClonesReady.addAndGet(1));
          if (ClientHandler.nrClonesReady.get() >= ClientHandler.numberOfCloneHelpers)
            ClientHandler.nrClonesReady.notifyAll();
        }

        /**
         * wait() until the main server wakes up the thread then do something depending on the
         * request
         */
        synchronized (ClientHandler.pausedHelper) {
          while (ClientHandler.pausedHelper[cloneHelperId]) {
            try {
              ClientHandler.pausedHelper.wait();
            } catch (InterruptedException e) {
            }
          }

          ClientHandler.pausedHelper[cloneHelperId] = true;
        }

        Log.d(TAG, "Sending command: " + ClientHandler.requestFromMainServer);

        switch (ClientHandler.requestFromMainServer) {

          case RapidMessages.PING:
            pingOtherServer();
            break;

          case RapidMessages.APK_REGISTER:
            mOutStream.write(RapidMessages.APK_REGISTER);
            mObjOutStream.writeObject(ClientHandler.appName);
            mObjOutStream.writeInt(ClientHandler.appLength);
            mObjOutStream.flush();

            int response = mInStream.read();

            if (response == RapidMessages.APK_REQUEST) {
              // Send the APK file if needed
              Log.d(TAG, "Sending apk to the clone " + clone.getIp());

              File apkFile = new File(ClientHandler.apkFilePath);
              FileInputStream fin = new FileInputStream(apkFile);
              BufferedInputStream bis = new BufferedInputStream(fin);
              int BUFFER_SIZE = 8192;
              byte[] tempArray = new byte[BUFFER_SIZE];
              int read = 0;
              int totalRead = 0;
              Log.d(TAG, "Sending apk");
              while ((read = bis.read(tempArray, 0, tempArray.length)) > -1) {
                totalRead += read;
                mObjOutStream.write(tempArray, 0, read);
                Log.d(TAG, "Sent " + totalRead + " of " + apkFile.length() + " bytes");
              }
              mObjOutStream.flush();
              bis.close();
            } else if (response == RapidMessages.APK_PRESENT) {
              Log.d(TAG, "Application already registered on clone " + clone.getIp());
            }
            break;

          case RapidMessages.EXECUTE:
            Log.d(TAG, "Asking clone " + clone.getIp() + " to parallelize the execution");

            mOutStream.write(RapidMessages.EXECUTE);

            // Send the number of clones needed.
            // Since this is a helper clone, only one clone should be requested.
            mObjOutStream.writeInt(1);
            mObjOutStream.writeObject(ClientHandler.objToExecute);
            mObjOutStream.writeObject(ClientHandler.methodName);
            mObjOutStream.writeObject(ClientHandler.pTypes);
            mObjOutStream.writeObject(ClientHandler.pValues);
            mObjOutStream.flush();

            /**
             * This is the response from the clone helper, which is a partial result of the method
             * execution. This partial result is stored in an array, and will be later composed with
             * the other partial results of the other clones to obtain the total desired result to
             * be sent back to the phone.
             */
            Object cloneResult = mObjInStream.readObject();

            ResultContainer container = (ResultContainer) cloneResult;

            Log.d(TAG,
                "Received response from clone ip: " + clone.getIp() + " port: " + clone.getPort());
            Log.d(TAG, "Writing in responsesFromServer in position: " + cloneHelperId);
            synchronized (ClientHandler.responsesFromServers) {
              Array.set(ClientHandler.responsesFromServers, cloneHelperId,
                  container.functionResult);
            }
            break;

          case -1:
            closeConnection();
            return;
        }
      }
    } catch (IOException e) {
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } finally {
      closeConnection();
    }
  }

  private boolean establishConnection() {
    try {

      Log.d(TAG, "Trying to connect to clone " + clone.getIp() + ":" + clone.getPort());

      mSocket = new Socket();
      mSocket.connect(new InetSocketAddress(clone.getIp(), clone.getPort()), 10 * 1000);

      mOutStream = mSocket.getOutputStream();
      mInStream = mSocket.getInputStream();
      mObjOutStream = new ObjectOutputStream(mOutStream);
      mObjInStream = new DynamicObjectInputStream(mInStream);

      Log.d(TAG, "Connection established whith clone " + clone.getIp());

      return true;
    } catch (Exception e) {
      Log.e(TAG, "Exception not caught properly - " + e);
      return false;
    } catch (Error e) {
      Log.e(TAG, "Error not caught properly - " + e);
      return false;
    }
  }

  private void pingOtherServer() {
    try {
      // Send a message to the Server Helper (other server)
      Log.d(TAG, "PING other server");
      mOutStream.write(eu.project.rapid.common.RapidMessages.PING);

      // Read and display the response message sent by server helper
      int response = mInStream.read();

      if (response == RapidMessages.PONG)
        Log.d(TAG, "PONG from other server: " + clone.getIp() + ":" + clone.getPort());
      else {
        Log.d(TAG, "Bad Response to Ping - " + response);
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private void closeConnection() {
    RapidUtils.closeQuietly(mObjOutStream);
    RapidUtils.closeQuietly(mObjInStream);
    RapidUtils.closeQuietly(mSocket);
  }
}


