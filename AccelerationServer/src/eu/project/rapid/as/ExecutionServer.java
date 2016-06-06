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

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.net.DatagramPacket;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.MulticastSocket;
import java.net.Socket;
import java.net.UnknownHostException;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.Security;
import java.security.UnrecoverableKeyException;
import java.security.cert.Certificate;
import java.security.cert.CertificateException;
import java.util.Random;

import javax.net.ssl.KeyManagerFactory;

import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.os.Handler;
import android.os.IBinder;
import android.util.Log;
import android.util.SparseArray;
import eu.project.rapid.ac.d2d.D2DMessage;
import eu.project.rapid.ac.utils.Constants;
import eu.project.rapid.ac.utils.Utils;
import eu.project.rapid.common.Configuration;
import eu.project.rapid.common.RapidMessages;
import eu.project.rapid.common.RapidUtils;

/**
 * Execution server which waits for incoming connections and starts a separate thread for each of
 * them, leaving the ClientHandler to actually deal with the clients
 * 
 */
public class ExecutionServer extends Service {

  private Context context;

  // When doing tests about send/receive data
  // To avoid creating the objects in the real deployment
  private static final boolean TESTING_UL_DL_RATE = false;
  public static SparseArray<byte[]> bytesToSend;

  private static final String TAG = "ExecutionServer";
  private Configuration config;

  public static boolean asServiceRunning = true;
  private boolean managerThreadFinished = false;
  private boolean okTalkingToManager = false;

  private Handler mBroadcastHandler;
  private Runnable mBroadcastRunnable;

  static {
    Security.insertProviderAt(new org.spongycastle.jce.provider.BouncyCastleProvider(), 1);
  }

  /** Called when the service is first created. */
  @Override
  public void onCreate() {
    super.onCreate();
    Log.d(TAG, "Server created");

    if (TESTING_UL_DL_RATE) {
      bytesToSend = new SparseArray<byte[]>();
      byte[] b1 = new byte[1024];
      byte[] b2 = new byte[1024 * 1024];
      new Random().nextBytes(b1);
      new Random().nextBytes(b2);
      bytesToSend.append(1024, b1);
      bytesToSend.append(1024 * 1024, b2);
    }
  }

  @Override
  public void onDestroy() {
    super.onDestroy();
    Log.d(TAG, "Server destroyed");
  }

  @Override
  public int onStartCommand(Intent intent, int flags, int startId) {
    // Create server socket
    Log.d(TAG, "Start server socket");
    context = this.getApplicationContext();
    if (context == null) {
      Log.e(TAG, "Context is null!!!");
      stopSelf();
    }

    // Create a special file on the clone that methods can use to check
    // if are being executed on the clone or on the phone.
    // This can be of help to advanced developers.
    createOffloadedFile();

    // Delete the file containing the cloneHelperId assigned to this clone
    // (if such file does not exist do nothing).
    Utils.deleteCloneHelperId();

    try {
      config = new Configuration(Constants.CLONE_CONFIG_FILE);
      config.parseConfigFile();
    } catch (FileNotFoundException e1) {
      Log.e(TAG, "Configuration file not found on the clone: " + Constants.CLONE_CONFIG_FILE);
      Log.e(TAG, "Continuinig with default values.");
      config = new Configuration();
    }

    try {
      Log.i(TAG, "KeyStore default type: " + KeyStore.getDefaultType());
      KeyStore keyStore = KeyStore.getInstance("BKS");
      keyStore.load(new FileInputStream(Constants.SSL_KEYSTORE),
          Constants.SSL_DEFAULT_PASSW.toCharArray());
      KeyManagerFactory kmf =
          KeyManagerFactory.getInstance(KeyManagerFactory.getDefaultAlgorithm());
      kmf.init(keyStore, Constants.SSL_DEFAULT_PASSW.toCharArray());

      PrivateKey privateKey = (PrivateKey) keyStore.getKey(Constants.SSL_CERT_ALIAS,
          Constants.SSL_DEFAULT_PASSW.toCharArray());
      Certificate cert = keyStore.getCertificate(Constants.SSL_CERT_ALIAS);
      PublicKey publicKey = cert.getPublicKey();

      config.setCryptoInitialized(true);
      config.setKmf(kmf);
      config.setPublicKey(publicKey);
      config.setPrivateKey(privateKey);

      Log.i(TAG, "Certificate: " + cert.toString());
      Log.i(TAG, "PrivateKey algorithm: " + privateKey.getAlgorithm());
      Log.i(TAG, "PublicKey algorithm: " + publicKey.getAlgorithm());

    } catch (IOException e) {
      Log.e(TAG, "Crypto not initialized: " + e);
    } catch (KeyStoreException e) {
      Log.e(TAG, "Crypto not initialized: " + e);
    } catch (NoSuchAlgorithmException e) {
      Log.e(TAG, "Crypto not initialized: " + e);
    } catch (CertificateException e) {
      Log.e(TAG, "Crypto not initialized: " + e);
    } catch (UnrecoverableKeyException e) {
      Log.e(TAG, "Crypto not initialized: " + e);
    }

    // Connect to the manager to register and get the configuration details
    Thread t = new Thread(new ConnectToManager());
    t.start();

    // Start the D2D handler that broadcasts ping messages with a certain frequency
    startD2DThread();

    return START_STICKY;
  }


  /**
   * Create a sentinel file on the clone in order to let the method know it is being executed on the
   * clone.
   */
  private void createOffloadedFile() {
    try {
      File f = new File(Constants.RAPID_FOLDER);
      if (!f.exists()) {
        f.mkdir();
      }
      f = new File(Constants.FILE_OFFLOADED);
      f.createNewFile();
    } catch (FileNotFoundException e) {
      Log.e(TAG, "Could not create offloaded file: " + e);
    } catch (IOException e) {
      Log.e(TAG, "Could not create offloaded file: " + e);
    }
  }

  @Override
  public IBinder onBind(Intent intent) {
    return null;
  }

  /**
   * Read the config file to get the IP and port for the Manager.
   * 
   * @throws IOException
   * @throws UnknownHostException
   */
  private class ConnectToManager implements Runnable {

    @Override
    public void run() {

      // Before proceeding wait until the network interface is up and correctly configured
      boolean hostMachineReachable = false;
      do {
        try {
          // The manager runs on the host machine, so checking if we can ping the ManagerIp in
          // reality we are checking if we can ping the host machine.
          // We should definitely be able to do that, otherwise this clone is useless if not
          // connected to the network.
          InetAddress hostMachineAddress = InetAddress.getByName(config.getManagerIp());
          try {
            Log.i(TAG,
                "Trying to ping the host machine " + hostMachineAddress.getHostAddress() + "...");
            hostMachineReachable = hostMachineAddress.isReachable(5000);
          } catch (IOException e) {
            Log.w(TAG, "Error while trying to ping the host machine: " + e);
          }
        } catch (UnknownHostException e1) {
          Log.e(TAG, "Error while getting hostname: " + e1);
        }
      } while (!hostMachineReachable);
      Log.i(TAG, "Host machine replied to ping. Network interface is up and running.");

      Socket dirManagerSocket = new Socket();
      ObjectOutputStream oos = null;
      ObjectInputStream ois = null;

      try {
        Log.d(TAG,
            "Connecting to Manager: " + config.getManagerIp() + ":" + config.getManagerPort());

        dirManagerSocket.connect(
            new InetSocketAddress(config.getManagerIp(), config.getManagerPort()), 3 * 1000);

        OutputStream os = dirManagerSocket.getOutputStream();
        InputStream is = dirManagerSocket.getInputStream();

        os.write(eu.project.rapid.common.RapidMessages.CLONE_CONNECTION);

        oos = new ObjectOutputStream(os);
        ois = new ObjectInputStream(is);

        // Send the name, id, and the public key to the Manager
        os.write(RapidMessages.CLONE_AUTHENTICATION);
        oos.writeObject(config.getCloneName());
        oos.writeInt(config.getCloneId());
        oos.writeBoolean(config.isCryptoInitialized());
        if (config.isCryptoInitialized()) {
          oos.writeObject(config.getPublicKey());
        }
        oos.flush();

        // Get the port where the clone should listen for connections
        config.setClonePort(ois.readInt());

        // Get the port where the clone should listen for connections
        config.setSslClonePort(ois.readInt());

        config.setAnimationServerIp(ois.readUTF());
        config.setAnimationServerPort(ois.readInt());

        okTalkingToManager = true;

      } catch (UnknownHostException e) {
        Log.e(TAG, "Could not talk to manager: " + e);
      } catch (IOException e) {
        Log.e(TAG, "Could not talk to manager: " + e);
      } catch (Exception e) {
        Log.e(TAG, "Could not talk to manager: " + e);
      } finally {
        Log.d(TAG, "Done talking to Manager");
        // Close the connection with the Manager
        RapidUtils.closeQuietly(ois);
        RapidUtils.closeQuietly(oos);
        RapidUtils.closeQuietly(dirManagerSocket);

        managerThreadFinished = true;
        startClientHandler();
      }
    }
  }

  private void startClientHandler() {

    if (!okTalkingToManager) {
      Log.w(TAG, "Could not talk to the manager. Starting with default values.");

      config.setClonePort(Configuration.DEFAULT_CLONE_PORT);
      config.setSslClonePort(Configuration.DEFAULT_SSL_CLONE_PORT);
    }

    Log.d(TAG, "Starting NetworkProfilerServer on port " + config.getClonePortBandwidthTest());
    new Thread(new NetworkProfilerServer(config)).start();

    Log.d(TAG, "Starting CloneThread on port " + config.getClonePort());
    new Thread(new CloneThread(context, config)).start();

    if (config.isCryptoInitialized()) {
      Log.d(TAG, "Starting CloneSslThread on port " + config.getSslClonePort());
      new Thread(new CloneSslThread(context, config)).start();
    } else {
      Log.w(TAG,
          "Cannot start the CloneSSLThread since the cryptographic initialization was not succesful");
    }
  }

  /**
   * FIXME This thread should run only on the mobile devices, so it should not be started
   * automatically here but the user should have to press a button to start it.
   */
  private void startD2DThread() {
    if (mBroadcastHandler == null) {
      mBroadcastHandler = new Handler();
    }
    if (mBroadcastRunnable == null) {
      mBroadcastRunnable = new BroadcastRunnable();
    }
    mBroadcastHandler.postDelayed(mBroadcastRunnable, Constants.D2D_BROADCAST_INTERVAL);
  }

  private class BroadcastRunnable implements Runnable {
    public void run() {
      Log.i(TAG, "Running the broadcast message runnable");
      if (context == null) {
        Log.e(TAG, "Context is null!!!");
      }
      broadcastMessage(new D2DMessage(context, D2DMessage.MsgType.HELLO));
      mBroadcastHandler.postDelayed(mBroadcastRunnable, Constants.D2D_BROADCAST_INTERVAL);
    }

    private void broadcastMessage(final D2DMessage msg) {
      new Thread() {
        public void run() {
          DatagramPacket packet;
          MulticastSocket socket = null;
          try {
            byte[] data = Utils.objectToByteArray(msg);

            socket = new MulticastSocket(Constants.D2D_BROADCAST_PORT);
            socket.setBroadcast(true);

            InetAddress myIpAddress = Utils.getIpAddress();
            Log.i(TAG, "My IP address: " + myIpAddress);
            InetAddress broadcastAddress = Utils.getBroadcast(myIpAddress);
            Log.i(TAG, "Broadcast IP address: " + broadcastAddress);
            try {
              packet = new DatagramPacket(data, data.length, broadcastAddress,
                  Constants.D2D_BROADCAST_PORT);
              socket.send(packet);
              Log.i(TAG, "==>> Broadcast message sent to " + broadcastAddress);
              Log.i(TAG, "==>> CMD: " + msg);
            } catch (UnknownHostException e) {
              Log.i(TAG, "UnknownHostException while sending data: " + e);
              e.printStackTrace();
            } catch (IOException e) {
              Log.i(TAG, "IOException while sending data: " + e);
              e.printStackTrace();
            } catch (NullPointerException e) {
              Log.i(TAG, "NullPointerException while sending data: " + e);
              e.printStackTrace();
            }
          } catch (IOException e) {
            e.printStackTrace();
          } finally {
            if (socket != null) {
              socket.close();
            }
          }
        }
      }.start();
    }
  }
}
