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
package eu.project.rapid.ac;

import java.io.BufferedInputStream;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.lang.reflect.Array;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.UnknownHostException;
import java.security.KeyManagementException;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.UnrecoverableKeyException;
import java.security.cert.Certificate;
import java.security.cert.CertificateException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import javax.net.ssl.HandshakeCompletedEvent;
import javax.net.ssl.HandshakeCompletedListener;
import javax.net.ssl.KeyManagerFactory;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSocket;
import javax.net.ssl.SSLSocketFactory;
import javax.net.ssl.TrustManagerFactory;

import android.app.ProgressDialog;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.pm.PackageManager.NameNotFoundException;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.AsyncTask;
import android.os.Build;
import android.preference.PreferenceManager;
import android.util.Log;
import eu.project.rapid.ac.d2d.D2DClientService;
import eu.project.rapid.ac.d2d.PhoneSpecs;
import eu.project.rapid.ac.db.DBCache;
import eu.project.rapid.ac.profilers.DeviceProfiler;
import eu.project.rapid.ac.profilers.NetworkProfiler;
import eu.project.rapid.ac.profilers.Profiler;
import eu.project.rapid.ac.profilers.ProgramProfiler;
import eu.project.rapid.ac.utils.Constants;
import eu.project.rapid.ac.utils.Utils;
import eu.project.rapid.common.Clone;
import eu.project.rapid.common.Configuration;
import eu.project.rapid.common.RapidConstants;
import eu.project.rapid.common.RapidConstants.COMM_TYPE;
import eu.project.rapid.common.RapidMessages;
import eu.project.rapid.common.RapidMessages.AnimationMsg;
import eu.project.rapid.common.RapidUtils;
import eu.project.rapid.gvirtusfe.Frontend;

/**
 * The most important class of the framework for the client program - controls DSE, profilers,
 * communicates with remote server.
 * 
 */

public class DFE {

  private static final String TAG = "DFE";

  public static boolean CONNECT_TO_PREVIOUS_VM = false;

  private Configuration config;
  public static final int SDK_INT = Build.VERSION.SDK_INT;

  private static int mRegime;
  public static final int REGIME_CLIENT = 1;
  public static final int REGIME_SERVER = 2;
  public static COMM_TYPE commType = COMM_TYPE.SSL;
  private int userChoice = Constants.LOCATION_DYNAMIC_TIME_ENERGY;
  private Method prepareDataMethod = null;

  private long mPureLocalDuration;
  private long mPureRemoteDuration;
  private long prepareDataDuration;

  // private Object result;
  private String mAppName;
  private Context mContext;
  private PackageManager mPManager;

  public static boolean onLineClear = false;
  public static boolean onLineSSL = false;
  public static boolean dfeReady = false;
  private int nrClones;
  private DSE mDSE;
  NetworkProfiler netProfiler;

  // private boolean profilersEnabled = true;

  // GVirtuS frontend is responsible for running the CUDA code.
  private Frontend gVirtusFrontend;
  private static Clone sClone;
  private static Socket mSocket;
  private static OutputStream mOutStream;
  private static ObjectOutputStream mObjOutStream;
  private static InputStream mInStream;
  private static ObjectInputStream mObjInStream;
  private final Object syncCreateCloseConnection = new Object();

  private long myId = -1;
  private String vmIp = "";
  private String vmmIp = "";
  private ArrayList<String> vmmIPs;
  // public LogRecord lastLogRecord;
  private PhoneSpecs myPhoneSpecs;
  private static final int vmNrVCPUs = 1; // FIXME: number of CPUs on the VM
  private static final int vmMemSize = 512; // FIXME
  private static final int vmNrGpuCores = 1200; // FIXME

  private Set<PhoneSpecs> d2dSetPhones = new TreeSet<PhoneSpecs>();
  private ScheduledThreadPoolExecutor d2dSetReaderThread;

  private ProgressDialog pd = null;

  /**
   * Interface to be implemented by some class that wants to be updated about some events.
   * 
   * @author sokol
   */
  public interface DfeCallback {
    public void vmConnectionStatusUpdate(); // Send updates about the VM connection status.
  }

  /**
   * Create DFE which decides where to execute remoteable methods and handles execution.
   * 
   * @param clone The clone to connect with.<br>
   *        If null then the DFE will connect to the manager and will ask for the clone.
   * @param appName Application name, usually the result of getPackageName().
   * @param pManager Package manager for finding apk file of the application.
   * @param context
   */
  public DFE(String appName, PackageManager pManager, Context context, Clone clone) {

    Log.d(TAG, "DFE Created");
    DFE.mRegime = REGIME_CLIENT;
    this.mAppName = appName;
    this.mPManager = pManager;
    this.mContext = context;
    sClone = clone;
    this.myPhoneSpecs = PhoneSpecs.getPhoneSpecs(mContext);

    Log.i(TAG, "Current device: " + myPhoneSpecs);

    createRapidFoldersIfNotExist();
    readConfigurationFile();
    initializeCrypto();

    // The prev id is useful to the DS so that it can release already allocated VMs.
    SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this.mContext);
    myId = prefs.getLong(Constants.MY_OLD_ID, -1);
    vmIp = prefs.getString(Constants.PREV_VM_IP, "");
    vmmIp = prefs.getString(Constants.PREV_VMM_IP, "");

    mDSE = new DSE(userChoice);

    netProfiler = new NetworkProfiler(mContext, config);
    netProfiler.registerNetworkStateTrackers();

    // Start the thread that will deal with the D2D communication
    startD2DThread();

    // Show a spinning dialog while connecting to the Manager and to the clone.
    this.pd = ProgressDialog.show(mContext, "Working...", "Initial network tasks...", true, false);
    (new InitialNetworkTasks()).execute(sClone);
  }

  public DFE(String appName, PackageManager pManager, Context context) {
    this(appName, pManager, context, null);
  }

  /**
   * To be used on server side, only local execution
   */
  public DFE() {
    mRegime = REGIME_SERVER;
  }

  private void createRapidFoldersIfNotExist() {
    File rapidDir = new File(Constants.RAPID_FOLDER);
    if (!rapidDir.exists()) {
      if (!rapidDir.mkdirs()) {
        Log.w(TAG, "Could not create the Rapid folder: " + Constants.RAPID_FOLDER);
      }
    }

    File rapidTestDir = new File(Constants.TEST_LOGS_FOLDER);
    if (!rapidTestDir.exists()) {
      if (!rapidTestDir.mkdirs()) {
        Log.w(TAG, "Could not create the Rapid folder: " + Constants.TEST_LOGS_FOLDER);
      }
    }
  }

  private void readConfigurationFile() {

    try {
      // Read the config file to read the IP and port of Manager
      config = new Configuration(Constants.PHONE_CONFIG_FILE);
      config.parseConfigFile();
    } catch (FileNotFoundException e) {
      Log.e(TAG, "Config file not found: " + Constants.PHONE_CONFIG_FILE);
      config = new Configuration();
    }
  }


  /**
   * Create the necessary stuff for the SSL connection.
   */
  private void initializeCrypto() {
    try {
      Log.i(TAG, "Started reading the cryptographic keys");

      KeyStore trustStore = KeyStore.getInstance("BKS");
      trustStore.load(new FileInputStream(Constants.SSL_CA_TRUSTSTORE),
          Constants.SSL_DEFAULT_PASSW.toCharArray());
      TrustManagerFactory trustManagerFactory =
          TrustManagerFactory.getInstance(KeyManagerFactory.getDefaultAlgorithm());
      trustManagerFactory.init(trustStore);

      KeyStore keyStore = KeyStore.getInstance("BKS");
      keyStore.load(new FileInputStream(Constants.SSL_KEYSTORE),
          Constants.SSL_DEFAULT_PASSW.toCharArray());
      KeyManagerFactory kmf =
          KeyManagerFactory.getInstance(KeyManagerFactory.getDefaultAlgorithm());
      kmf.init(keyStore, Constants.SSL_DEFAULT_PASSW.toCharArray());

      PrivateKey myPrivateKey = (PrivateKey) keyStore.getKey(Constants.SSL_CERT_ALIAS,
          Constants.SSL_DEFAULT_PASSW.toCharArray());
      Certificate cert = keyStore.getCertificate(Constants.SSL_CERT_ALIAS);
      PublicKey myPublicKey = cert.getPublicKey();
      Log.i(TAG, "Certificate: " + cert.toString());
      Log.i(TAG, "PrivateKey algorithm: " + myPrivateKey.getAlgorithm());
      Log.i(TAG, "PublicKey algorithm: " + myPrivateKey.getAlgorithm());

      SSLContext sslContext = SSLContext.getInstance("TLS");
      sslContext.init(null, trustManagerFactory.getTrustManagers(), null);
      SSLSocketFactory sslFactory = sslContext.getSocketFactory();
      Log.i(TAG, "SSL Factory created");

      config.setCryptoInitialized(true);
      config.setPrivateKey(myPrivateKey);
      config.setPublicKey(myPublicKey);
      config.setSslFactory(sslFactory);
      config.setSslContext(sslContext);

    } catch (KeyStoreException | NoSuchAlgorithmException | KeyManagementException
        | UnrecoverableKeyException | CertificateException | IOException e) {
      Log.e(TAG, "Could not initialize the crypto parameters - " + e);
    }
  }

  private class InitialNetworkTasks extends AsyncTask<Clone, String, Void> {

    private static final String TAG = "InitialNetworkTasks";

    @Override
    protected Void doInBackground(Clone... clone) {

      // Check if the primary animation server is reachable
      boolean primaryAnimationServerReachable = RapidUtils.isPrimaryAnimationServerRunning(
          config.getAnimationServerIp(), config.getAnimationServerPort());
      Log.i(TAG, "Primary animation server reachable: " + primaryAnimationServerReachable);
      if (!primaryAnimationServerReachable) {
        config.setAnimationServerIp(RapidConstants.DEFAULT_SECONDARY_ANIMATION_SERVER_IP);
        config.setAnimationServerPort(RapidConstants.DEFAULT_SECONDARY_ANIMATION_SERVER_PORT);
      }

      // Anyway, show the default image, where the AC tries to connects to DS, SLAM, etc. If this
      // fails, then we will show the D2D initial image
      RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_INITIAL_IMG);

      if (clone[0] == null) {
        publishProgress("Registering with the DS and the SLAM...");
        registerWithDsAndSlam();
      } else {
        publishProgress("Using the clone given by the user: " + clone[0]);
        sClone = clone[0];
      }

      config.setClone(sClone);

      RapidUtils.sendAnimationMsg(config, CONNECT_TO_PREVIOUS_VM ? AnimationMsg.AC_PREV_REGISTER_VM
          : AnimationMsg.AC_NEW_REGISTER_VM);
      if (commType == COMM_TYPE.CLEAR) {
        publishProgress("Clear connection with the clone: " + sClone);
        establishConnection();
      } else { // (commType == COMM_TYPE.SSL)
        publishProgress("SSL connection with the clone: " + sClone);
        if (!establishSslConnection()) {
          Log.w(TAG, "Setting commType to CLEAR");
          commType = COMM_TYPE.CLEAR;
          establishConnection();
        }
      }

      // If the connection was successful then try to send the app to the clone
      if (onLineClear || onLineSSL) {
        RapidUtils.sendAnimationMsg(config,
            CONNECT_TO_PREVIOUS_VM ? AnimationMsg.AC_PREV_CONN_VM : AnimationMsg.AC_NEW_CONN_VM);

        Log.i(TAG, "The communication type established with the clone is: " + commType);

        if (config.getGvirtusIp() == null) {
          // If gvirtusIp is null, then gvirtus backend is running on the physical machine where
          // the VM is running.
          // Try to find a way here to get the ip address of the physical machine.
          // config.setGvirtusIp(TODO: ip address of the physical machine where the VM is running);
        }

        publishProgress("Registering application with the RAPID system...");
        RapidUtils.sendAnimationMsg(config,
            CONNECT_TO_PREVIOUS_VM ? AnimationMsg.AC_PREV_APK_VM : AnimationMsg.AC_NEW_APK_VM);
        sendApk();

        // Find rtt to the server
        // Measure the data rate when just connected
        publishProgress("Sending/receiving data for 3 seconds to measure the ulRate and dlRate...");
        RapidUtils.sendAnimationMsg(config,
            CONNECT_TO_PREVIOUS_VM ? AnimationMsg.AC_PREV_RTT_VM : AnimationMsg.AC_NEW_RTT_VM);
        NetworkProfiler.rttPing(mInStream, mOutStream);
        RapidUtils.sendAnimationMsg(config, CONNECT_TO_PREVIOUS_VM ? AnimationMsg.AC_PREV_DL_RATE_VM
            : AnimationMsg.AC_NEW_DL_RATE_VM);
        NetworkProfiler.measureDlRate(sClone.getIp(), config.getClonePortBandwidthTest());
        RapidUtils.sendAnimationMsg(config, CONNECT_TO_PREVIOUS_VM ? AnimationMsg.AC_PREV_UL_RATE_VM
            : AnimationMsg.AC_NEW_UL_RATE_VM);
        NetworkProfiler.measureUlRate(sClone.getIp(), config.getClonePortBandwidthTest());
        RapidUtils.sendAnimationMsg(config, CONNECT_TO_PREVIOUS_VM
            ? AnimationMsg.AC_PREV_REGISTRATION_OK_VM : AnimationMsg.AC_NEW_REGISTRATION_OK_VM);

        try {
          ((DfeCallback) mContext).vmConnectionStatusUpdate();
        } catch (ClassCastException e) {
          Log.i(TAG, "This class doesn't implement callback methods.");
        }
      } else {
        RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_REGISTER_VM_ERROR);
      }

      // if (config.getGvirtusIp() != null) {
      // // Create a gvirtus frontend object that is responsible for executing the CUDA code.
      // publishProgress("Connecting with GVirtuS backend...");
      // gVirtusFrontend = new Frontend(config.getGvirtusIp(), config.getGvirtusPort());
      // }

      return null;
    }

    @Override
    protected void onProgressUpdate(String... progress) {
      Log.i(TAG, progress[0]);
      if (pd != null) {
        pd.setMessage(progress[0]);
      }
    }

    @Override
    protected void onPostExecute(Void result) {
      Log.i(TAG, "Finished initial network tasks");
      dfeReady = true;
      if (pd != null) {
        pd.dismiss();
      }
    }

    @Override
    protected void onPreExecute() {
      Log.i(TAG, "Started initial network tasks");
    }

    private boolean registerWithDsAndSlam() {
      Log.i(TAG, "Registering...");
      boolean registeredWithSlam = false;

      if (registerWithDs()) {
        // register with SLAM
        int vmmIndex = 0;

        if (vmmIPs != null) {
          do {
            registeredWithSlam = registerWithSlam(vmmIPs.get(vmmIndex));
            vmmIndex++;
          } while (!registeredWithSlam && vmmIndex < vmmIPs.size());
        }
      }
      return registeredWithSlam;
    }

    /**
     * Read the config file to get the IP and port of the DS. The DS will return a list of available
     * SLAMs, choose the best one from the list and connect to it to ask for a VM.
     * 
     * @throws IOException
     * @throws UnknownHostException
     * @throws ClassNotFoundException
     */
    @SuppressWarnings("unchecked")
    private boolean registerWithDs() {

      Log.d(TAG, "Starting as phone with ID: " + myId);
      RapidUtils.sendAnimationMsg(config,
          CONNECT_TO_PREVIOUS_VM ? AnimationMsg.AC_PREV_VM_DS : AnimationMsg.AC_NEW_REGISTER_DS);

      Socket dsSocket = null;
      ObjectOutputStream dsOut = null;
      ObjectInputStream dsIn = null;

      Log.i(TAG, "Registering with DS " + config.getDSIp() + ":" + config.getDSPort());
      try {
        dsSocket = new Socket();
        dsSocket.connect(new InetSocketAddress(config.getDSIp(), config.getDSPort()), 10 * 1000);
        dsOut = new ObjectOutputStream(dsSocket.getOutputStream());
        dsIn = new ObjectInputStream(dsSocket.getInputStream());

        // Send the name and id to the DS
        if (CONNECT_TO_PREVIOUS_VM) {
          // RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_PREV_VM_DS);
          Log.i(TAG, "AC_REGISTER_PREV_DS");
          // Send message format: command (java byte), userId (java long), qosFlag (java int)
          dsOut.writeByte(RapidMessages.AC_REGISTER_PREV_DS);
          dsOut.writeLong(myId); // send my user ID so that my previous VM can be released
        } else { // Connect to a new VM
          // RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_NEW_REGISTER_DS);
          Log.i(TAG, "AC_REGISTER_NEW_DS");
          dsOut.writeByte(RapidMessages.AC_REGISTER_NEW_DS);

          dsOut.writeLong(myId); // send my user ID so that my previous VM can be released
          // FIXME: should not use static values here.
          dsOut.writeInt(vmNrVCPUs); // send vcpuNum as int
          dsOut.writeInt(vmMemSize); // send memSize as int
          dsOut.writeInt(vmNrGpuCores); // send gpuCores as int
        }

        dsOut.flush();

        // Receive message format: status (java byte), userId (java long), SLAM ipAddress (java UTF)
        byte status = dsIn.readByte();
        Log.i(TAG, "Return Status: " + (status == RapidMessages.OK ? "OK" : "ERROR"));
        if (status == RapidMessages.OK) {
          myId = dsIn.readLong();
          Log.i(TAG, "userId is: " + myId);

          // Read the list of VMMs, which will be sorted based on free resources
          vmmIPs = (ArrayList<String>) dsIn.readObject();

          // Read the SLAM IP and port
          String slamIp = dsIn.readUTF();
          int slamPort = dsIn.readInt();
          config.setSlamIp(slamIp);
          config.setSlamPort(slamPort);
          Log.i(TAG, "SLAM address is: " + slamIp + ":" + slamPort);

          return true;
        }
      } catch (Exception e) {
        Log.e(TAG, "Error while connecting with the DS: " + e);
      } finally {
        RapidUtils.closeQuietly(dsOut);
        RapidUtils.closeQuietly(dsIn);
        RapidUtils.closeQuietly(dsSocket);
      }

      return false;
    }

    /**
     * @throws IOException
     * @throws UnknownHostException
     * @throws ClassNotFoundException
     */
    private boolean registerWithSlam(String vmmIp) {

      Socket slamSocket = null;
      ObjectOutputStream oos = null;
      ObjectInputStream ois = null;

      Log.i(TAG, "Registering with SLAM " + config.getSlamIp() + ":" + config.getSlamPort());
      try {

        slamSocket = new Socket();
        slamSocket.connect(new InetSocketAddress(config.getSlamIp(), config.getSlamPort()),
            10 * 1000);

        oos = new ObjectOutputStream(slamSocket.getOutputStream());
        ois = new ObjectInputStream(slamSocket.getInputStream());

        RapidUtils.sendAnimationMsg(config, CONNECT_TO_PREVIOUS_VM
            ? AnimationMsg.AC_PREV_REGISTER_SLAM : AnimationMsg.AC_NEW_REGISTER_SLAM);

        // Send the ID to the SLAM
        oos.writeByte(RapidMessages.AC_REGISTER_SLAM);
        oos.writeLong(myId);
        oos.writeInt(RapidConstants.OS.ANDROID.ordinal());

        // Send the vmmId and vmmPort to the SLAM so it can start the VM
        oos.writeUTF(vmmIp);
        oos.writeInt(config.getVmmPort());

        // FIXME: should not use static values here.
        oos.writeInt(vmNrVCPUs); // send vcpuNum as int
        oos.writeInt(vmMemSize); // send memSize as int
        oos.writeInt(vmNrGpuCores); // send gpuCores as int

        oos.flush();

        int slamResponse = ois.readByte();
        if (slamResponse == RapidMessages.OK) {
          Log.i(TAG, "SLAM OK, getting the VM details");
          vmIp = ois.readUTF();

          sClone = new Clone("", vmIp);
          sClone.setId((int) myId);

          Log.i(TAG, "Saving my ID and the vmIp: " + myId + ", " + vmIp);
          SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(mContext);
          SharedPreferences.Editor editor = prefs.edit();

          editor.putLong(Constants.MY_OLD_ID, myId);
          editor.putString(Constants.PREV_VM_IP, vmIp);

          Log.i(TAG, "Saving the VMM IP: " + vmmIp);
          editor.putString(Constants.PREV_VMM_IP, vmmIp);
          editor.commit();

          return true;
        } else if (slamResponse == RapidMessages.ERROR) {
          Log.e(TAG, "SLAM registration replied with ERROR, VM will be null");
        } else {
          Log.e(TAG, "SLAM registration replied with uknown message " + slamResponse
              + ", VM will be null");
        }
      } catch (IOException e) {
        Log.e(TAG, "IOException while talking to the SLAM: " + e);
      } finally {
        RapidUtils.closeQuietly(oos);
        RapidUtils.closeQuietly(ois);
        RapidUtils.closeQuietly(slamSocket);
      }

      return false;
    }
  }

  public void onDestroy() {
    Log.d(TAG, "onDestroy");
    dfeReady = false;
    DeviceProfiler.onDestroy();
    netProfiler.onDestroy();
    DBCache.saveDbCache();
    closeConnection();
    // FIXME Should I also stop the D2D listening service here or should I leave it running?
    RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_INITIAL_IMG);
  }

  private void startD2DThread() {
    Log.i(TAG, "Starting the D2D listening service...");
    Intent d2dServiceIntent = new Intent(mContext, D2DClientService.class);
    mContext.startService(d2dServiceIntent);

    // This thread will run with a certain frequency to read the D2D devices that are written by the
    // D2DClientService on the sdcard. This is needed because another DFE may have written the
    // devices on the file.
    d2dSetReaderThread = (ScheduledThreadPoolExecutor) Executors.newScheduledThreadPool(5);
    d2dSetReaderThread.scheduleWithFixedDelay(new D2DSetReader(), 0,
        D2DClientService.FREQUENCY_READ_D2D_SET, TimeUnit.MILLISECONDS);
  }

  private class D2DSetReader implements Runnable {

    @SuppressWarnings("unchecked")
    @Override
    public void run() {
      try {
        Log.i(TAG, "Reading the saved D2D phones...");
        d2dSetPhones = (Set<PhoneSpecs>) Utils.readObjectFromFile(Constants.FILE_D2D_PHONES);

        if (d2dSetPhones != null && d2dSetPhones.size() > 0) {
          RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_RECEIVED_D2D);
        } else {
          RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_NO_MORE_D2D);
        }

        Log.i(TAG, "List of D2D phones:");
        for (PhoneSpecs p : d2dSetPhones) {
          Log.i(TAG, p.toString());
        }
      } catch (IOException | ClassNotFoundException e) {
        Log.e(TAG, "Error on D2DSetReader while trying to read the saved set of D2D phones: " + e);
      }
    }
  }

  /**
   * Set up streams for the socket connection, perform initial communication with the clone.
   */
  private boolean establishConnection() {
    synchronized (syncCreateCloseConnection) {
      try {
        long sTime = System.nanoTime();
        long startTxBytes = NetworkProfiler.getProcessTxBytes();
        long startRxBytes = NetworkProfiler.getProcessRxBytes();

        Log.i(TAG, "Connecting in CLEAR with AS on: " + sClone.getIp() + ":" + sClone.getPort());
        mSocket = new Socket();
        mSocket.connect(new InetSocketAddress(sClone.getIp(), sClone.getPort()), 10 * 1000);

        mOutStream = mSocket.getOutputStream();
        mInStream = mSocket.getInputStream();
        mObjOutStream = new ObjectOutputStream(mOutStream);
        mObjInStream = new ObjectInputStream(mInStream);

        long dur = System.nanoTime() - sTime;
        long totalTxBytes = NetworkProfiler.getProcessTxBytes() - startTxBytes;
        long totalRxBytes = NetworkProfiler.getProcessRxBytes() - startRxBytes;

        Log.d(TAG, "Socket and streams set-up time - " + dur / 1000000 + "ms");
        Log.d(TAG, "Total bytes sent: " + totalTxBytes);
        Log.d(TAG, "Total bytes received: " + totalRxBytes);
        return onLineClear = true;

      } catch (Exception e) {
        e.printStackTrace();
        fallBackToLocalExecution("Connection setup with the clone failed: " + e);
      } finally {
        onLineSSL = false;
      }
      return onLineClear = false;
    }
  }

  /**
   * Set up streams for the secure socket connection, perform initial communication with the server.
   */
  private boolean establishSslConnection() {

    if (!config.isCryptoInitialized()) {
      Log.e(TAG,
          "The initialization of the cryptographic keys is not done correctly. Cannot perform ssl connection!");
      return false;
    }

    synchronized (syncCreateCloseConnection) {
      try {
        // RapidUtils.sendAnimationMsg(config, RapidMessages.AC_CONNECT_VM);

        Long sTime = System.nanoTime();
        long startTxBytes = NetworkProfiler.getProcessTxBytes();
        long startRxBytes = NetworkProfiler.getProcessRxBytes();

        Log.i(TAG, "Connecting in SSL with clone: " + sClone.getIp() + ":" + sClone.getSslPort());

        mSocket = config.getSslFactory().createSocket(sClone.getIp(), sClone.getSslPort());
        // Log.i(TAG, "getEnableSessionCreation: " + ((SSLSocket)
        // mSocket).getEnableSessionCreation());
        // ((SSLSocket) mSocket).setEnableSessionCreation(false);

        // sslContext.getClientSessionContext().getSession(null).invalidate();

        ((SSLSocket) mSocket).addHandshakeCompletedListener(new SSLHandshakeCompletedListener());
        Log.i(TAG, "socket created");

        // Log.i(TAG, "Enabled cipher suites: ");
        // for (String s : ((SSLSocket) mSocket).getEnabledCipherSuites()) {
        // Log.i(TAG, s);
        // }

        mOutStream = mSocket.getOutputStream();
        mInStream = mSocket.getInputStream();
        mObjOutStream = new ObjectOutputStream(mOutStream);
        mObjInStream = new ObjectInputStream(mInStream);

        long dur = System.nanoTime() - sTime;
        long totalTxBytes = NetworkProfiler.getProcessTxBytes() - startTxBytes;
        long totalRxBytes = NetworkProfiler.getProcessRxBytes() - startRxBytes;

        Log.d(TAG, "Socket and streams set-up time - " + dur / 1000000 + "ms");
        Log.d(TAG, "Total bytes sent: " + totalTxBytes);
        Log.d(TAG, "Total bytes received: " + totalRxBytes);
        return onLineSSL = true;

      } catch (UnknownHostException e) {
        e.printStackTrace();
        fallBackToLocalExecution("UnknownHostException - Connection setup to server failed: " + e);
      } catch (IOException e) {
        e.printStackTrace();
        fallBackToLocalExecution("IOException - Connection setup to server failed: " + e);
      } catch (Exception e) {
        e.printStackTrace();
        fallBackToLocalExecution("Exception - Connection setup to server failed: " + e);
      } finally {
        onLineClear = false;
      }

      return onLineSSL = false;
    }
  }

  private class SSLHandshakeCompletedListener implements HandshakeCompletedListener {

    @Override
    public void handshakeCompleted(HandshakeCompletedEvent event) {
      Log.i(TAG, "SSL handshake completed");

      try {
        // Log.i(TAG, "getCipherSuite: " + event.getCipherSuite());
        // Log.i(TAG, "algorithm: " + config.getPrivateKey().getAlgorithm());
        // Log.i(TAG, "modulusBitLength: " + ((RSAPrivateKey)
        // config.getPrivateKey()).getModulus().bitLength());

        // SSLSession session = event.getSession();
        // Log.i(TAG, "getProtocol: " + session.getProtocol());
        // Log.i(TAG, "getPeerHost: " + session.getPeerHost());
        // Log.i(TAG, "getId: " + RapidUtils.bytesToHex(session.getId()));
        // Log.i(TAG, "getCreationTime: " + session.getCreationTime());

        // java.security.cert.Certificate[] certs = event.getPeerCertificates();
        // for (int i = 0; i < certs.length; i++)
        // {
        // if (!(certs[i] instanceof java.security.cert.X509Certificate)) continue;
        // java.security.cert.X509Certificate cert = (java.security.cert.X509Certificate) certs[i];
        // Log.i(TAG, "Cert #" + i + ": " + cert.getSubjectDN().getName());
        // }
      } catch (Exception e) {
        Log.e(TAG, "SSL handshake completed with errors: " + e);
      }
    }

  }

  private void closeConnection() {
    new Thread(new Runnable() {
      @Override
      public void run() {
        synchronized (syncCreateCloseConnection) {
          // RapidUtils.sendAnimationMsg(config, RapidMessages.AC_DISCONNECT_VM);
          RapidUtils.closeQuietly(mObjOutStream);
          RapidUtils.closeQuietly(mObjInStream);
          RapidUtils.closeQuietly(mSocket);
          onLineClear = onLineSSL = false;
        }
      }
    }).start();
  }

  private void fallBackToLocalExecution(String message) {
    Log.e(TAG, message);
    onLineClear = onLineSSL = false;
  }

  /**
   * @param appName The application name.
   * @param methodName The current method that we want to offload from this application.<br>
   *        Different methods of the same application will have a different set of parameters.
   * @return The execution location which can be one of: LOCAL, REMOTE.<br>
   */
  private int findExecLocation(String appName, String methodName) {

    int execLocation = mDSE.findExecLocation(appName, methodName);
    Log.i(TAG, "Execution location: " + execLocation);

    return execLocation;
  }

  /**
   * Ask the DSE to find the best offloading scheme based on user choice. Developer can use this
   * method in order to prepare the data input based on the decision the framework will make about
   * the execution location.
   * 
   * @param methodName The name of the method that will be executed.
   * @return
   */
  private int findExecLocation(String methodName) {
    return findExecLocation(mAppName, methodName);
  }

  /**
   * Wrapper of the execute method with no parameters for the executable method
   * 
   * @param m
   * @param o
   * @return
   * @throws Throwable
   */
  public Object execute(Method m, Object o) throws Throwable {
    return execute(m, (Object[]) null, o);
  }

  /**
   * Call DSE to decide where to execute the operation, start profilers, execute (either locally or
   * remotely), collect profiling data and return execution results.
   * 
   * @param m method to be executed
   * @param pValues with parameter values
   * @param o on object
   * @return result of execution, or an exception if it happened
   * @throws NoSuchMethodException
   * @throws ClassNotFoundException
   * @throws IllegalAccessException
   * @throws SecurityException
   * @throws IllegalArgumentException
   */
  public Object execute(Method m, Object[] pValues, Object o) throws IllegalArgumentException,
      SecurityException, IllegalAccessException, ClassNotFoundException, NoSuchMethodException {

    Object localResult, remoteResult, totalResult = null;

    int execLocation = findExecLocation(m.getName());

    ExecutorService executor = Executors.newFixedThreadPool(2);
    if (execLocation == Constants.LOCATION_HYBRID) {
      Future<Object> futureLocalResult =
          executor.submit(new TaskRunner(Constants.LOCATION_LOCAL, m, pValues, o));
      Future<Object> futureRemoteResult =
          executor.submit(new TaskRunner(Constants.LOCATION_REMOTE, m, pValues, o));

      try {
        localResult = futureLocalResult.get();
        remoteResult = futureRemoteResult.get();

        // Reduce the partial results
        Method reduceMethod = o.getClass().getDeclaredMethod(m.getName() + "Reduce",
            Array.newInstance(m.getReturnType(), 2).getClass());
        reduceMethod.setAccessible(true);
        Object partialResults = Array.newInstance(m.getReturnType(), 2);
        Array.set(partialResults, 0, localResult);
        Array.set(partialResults, 1, remoteResult);

        totalResult = reduceMethod.invoke(o, new Object[] {partialResults});
      } catch (InterruptedException e) {
        Log.e(TAG, "Error on FutureTask while trying to run the method hybrid: " + e);
      } catch (ExecutionException e) {
        Log.e(TAG, "Error on FutureTask while trying to run the method hybrid: " + e);
      } catch (InvocationTargetException e) {
        Log.e(TAG, "Error on FutureTask while trying to run the method hybrid: " + e);
      }
    } else { // ecexLocation LOCAL or REMOTE
      Future<Object> futureTotalResult =
          executor.submit(new TaskRunner(execLocation, m, pValues, o));

      try {
        totalResult = futureTotalResult.get();
      } catch (InterruptedException | ExecutionException e) {
        Log.e(TAG, "Error on FutureTask while trying to run the method remotely or locally: " + e);
      }
    }

    executor.shutdown();

    return totalResult;
  }

  private class TaskRunner implements Callable<Object> {
    int execLocation;
    Method m;
    Object[] pValues;
    Object o;
    Object result;

    public TaskRunner(int execLocation, Method m, Object[] pValues, Object o) {
      this.execLocation = execLocation;
      this.m = m;
      this.pValues = pValues;
      this.o = o;
    }

    @Override
    public Object call() {

      RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_INITIAL_IMG);
      RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_PREPARE_DATA);
      if (execLocation == Constants.LOCATION_LOCAL) {
        try {

          // First try to see if we can offload this task to a more powerful device that is in D2D
          // distance.
          // Do this only if we are not connected to a clone, otherwise it becomes a mess.
          if (!onLineClear && !onLineSSL) {

            // I'm sure this cast is correct since it has been us who wrote the object before.
            try {
              if (d2dSetPhones != null && d2dSetPhones.size() > 0) {
                Iterator<PhoneSpecs> it = d2dSetPhones.iterator();
                // This is the best phone from the D2D ones since the set is sorted and this is the
                // first element.
                PhoneSpecs otherPhone = it.next();
                if (otherPhone.compareTo(myPhoneSpecs) > 0) {
                  RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_OFFLOAD_D2D);
                  this.result = executeD2D(otherPhone);
                }
              }
            } catch (IOException | ClassNotFoundException | SecurityException
                | NoSuchMethodException e) {
              Log.e(TAG, "Error while trying to run the method D2D: " + e);
            }
          }

          // If the D2D execution didn't take place or something happened that the execution was
          // interrupted the result would still be null.
          if (this.result == null) {
            RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_DECISION_LOCAL);
            this.result = executeLocally(m, pValues, o);
          }

        } catch (IllegalArgumentException | IllegalAccessException | InvocationTargetException e) {
          Log.e(TAG, "Error while running the method locally: " + e);
        }
      } else if (execLocation == Constants.LOCATION_REMOTE) {
        try {
          this.result = executeRemotely(m, pValues, o);
          if (this.result instanceof InvocationTargetException) {
            // The remote execution throwed an exception, try to run the method locally.
            Log.w(TAG, "The result was InvocationTargetException. Running the method locally");
            this.result = executeLocally(m, pValues, o);
          }
        } catch (IllegalArgumentException | SecurityException | IllegalAccessException
            | InvocationTargetException | ClassNotFoundException | NoSuchMethodException e) {
          Log.e(TAG, "Error while trying to run the method remotely: " + e);
        }
      }

      return this.result;
    }

    /**
     * Execute the method locally
     * 
     * @param m
     * @param pValues
     * @param o
     * @return
     * @throws IllegalArgumentException
     * @throws IllegalAccessException
     * @throws InvocationTargetException
     */
    private Object executeLocally(Method m, Object[] pValues, Object o)
        throws IllegalArgumentException, IllegalAccessException, InvocationTargetException {

      Profiler profiler = null;
      ProgramProfiler progProfiler = new ProgramProfiler(mAppName, m.getName());
      DeviceProfiler devProfiler = new DeviceProfiler(mContext);
      NetworkProfiler netProfiler = null;
      profiler = new Profiler(mRegime, progProfiler, netProfiler, devProfiler);

      // Start tracking execution statistics for the method
      profiler.startExecutionInfoTracking();

      // Make sure that the method is accessible
      // RapidUtils.sendAnimationMsg(config, RapidMessages.AC_EXEC_LOCAL);
      Object result = null;
      Long startTime = System.nanoTime();
      m.setAccessible(true);
      result = m.invoke(o, pValues); // Access it
      mPureLocalDuration = System.nanoTime() - startTime;
      Log.d(TAG, "LOCAL " + m.getName() + ": Actual Invocation duration - "
          + mPureLocalDuration / 1000000 + "ms");

      // Collect execution statistics
      profiler.stopAndLogExecutionInfoTracking(prepareDataDuration, mPureLocalDuration);

      RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_LOCAL_FINISHED);

      return result;
    }

    /**
     * Execute the method on a phone that is close to us so that we can connect on D2D mode. If we
     * are here, it means that this phone is not connected to a clone, so we can define the clone to
     * be this D2D device.
     * 
     * @param otherPhone
     * @return
     * @throws IllegalArgumentException
     * @throws IllegalAccessException
     * @throws InvocationTargetException
     * @throws IOException
     * @throws ClassNotFoundException
     * @throws NoSuchMethodException
     * @throws SecurityException
     */
    private Object executeD2D(PhoneSpecs otherPhone)
        throws IllegalArgumentException, IllegalAccessException, InvocationTargetException,
        IOException, ClassNotFoundException, SecurityException, NoSuchMethodException {

      // otherPhone.setIp("192.168.43.1");
      Log.i(TAG, "Trying to execute the method using D2D on device: " + otherPhone);
      sClone = new Clone("vb-D2D device", otherPhone.getIp(), config.getClonePort());
      establishConnection();
      sendApk();
      Object result = executeRemotely(m, pValues, o);
      closeConnection();
      return result;
    }

    /**
     * Execute method remotely
     * 
     * @param m
     * @param pValues
     * @param o
     * @return
     * @throws IllegalArgumentException
     * @throws IllegalAccessException
     * @throws InvocationTargetException
     * @throws NoSuchMethodException
     * @throws ClassNotFoundException
     * @throws SecurityException
     */
    private Object executeRemotely(Method m, Object[] pValues, Object o)
        throws IllegalArgumentException, IllegalAccessException, InvocationTargetException,
        SecurityException, ClassNotFoundException, NoSuchMethodException {
      Object result = null;

      RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_DECISION_OFFLOAD_AS);

      // Maybe the developer has implemented the prepareDataOnClient() method that helps him prepare
      // the data based on where the execution will take place then call it.
      // Prepare the data by calling the prepareData(localFraction) implemented by the developer.
      try {
        long s = System.nanoTime();
        prepareDataMethod = o.getClass().getDeclaredMethod("prepareDataOnClient");
        prepareDataMethod.setAccessible(true);
        prepareDataMethod.invoke(o);
        prepareDataDuration = System.nanoTime() - s;
      } catch (NoSuchMethodException e) {
        Log.w(TAG, "The method prepareDataOnClient() does not exist");
      }

      ProgramProfiler progProfiler = new ProgramProfiler(mAppName, m.getName());
      DeviceProfiler devProfiler = new DeviceProfiler(mContext);
      NetworkProfiler netProfiler = new NetworkProfiler();
      Profiler profiler = new Profiler(mRegime, progProfiler, netProfiler, devProfiler);

      // Start tracking execution statistics for the method
      profiler.startExecutionInfoTracking();

      try {
        Long startTime = System.nanoTime();
        mOutStream.write(RapidMessages.AC_OFFLOAD_REQ_AS);
        result = sendAndExecute(m, pValues, o, mObjInStream, mObjOutStream);

        Long duration = System.nanoTime() - startTime;
        Log.d(TAG, "REMOTE " + m.getName() + ": Actual Send-Receive duration - "
            + duration / 1000000 + "ms");
        // Collect execution statistics
        profiler.stopAndLogExecutionInfoTracking(prepareDataDuration, mPureRemoteDuration);

        RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_OFFLOADING_FINISHED);
      } catch (Exception e) {
        // No such host exists, execute locally
        Log.e(TAG, "REMOTE ERROR: " + m.getName() + ": " + e);
        e.printStackTrace();
        profiler.stopAndDiscardExecutionInfoTracking();
        result = executeLocally(m, pValues, o);
        // ConnectionRepair repair = new ConnectionRepair();
        // repair.start();
      }
      return result;
    }

    /**
     * Send the object (along with method and parameters) to the remote server for execution
     * 
     * @param o
     * @param m
     * @param pValues
     * @param objOut
     * @throws IOException
     */
    private void sendObject(Object o, Method m, Object[] pValues, ObjectOutputStream objOut)
        throws IOException {
      objOut.reset();
      Log.d(TAG, "Write Object and data");

      // Send the number of clones needed to execute the method
      objOut.writeInt(nrClones);

      // Send object for execution
      objOut.writeObject(o);

      // Send the method to be executed
      // Log.d(TAG, "Write Method - " + m.getName());
      objOut.writeObject(m.getName());

      // Log.d(TAG, "Write method parameter types");
      objOut.writeObject(m.getParameterTypes());

      // Log.d(TAG, "Write method parameter values");
      objOut.writeObject(pValues);
      objOut.flush();
    }

    /**
     * Send the object, the method to be executed and parameter values to the remote server for
     * execution.
     * 
     * @param m method to be executed
     * @param pValues parameter values of the remoted method
     * @param o the remoted object
     * @param objIn ObjectInputStream which to read results from
     * @param objOut ObjectOutputStream which to write the data to
     * @return result of the remoted method or an exception that occurs during execution
     * @throws IOException
     * @throws ClassNotFoundException
     * @throws NoSuchMethodException
     * @throws InvocationTargetException
     * @throws IllegalAccessException
     * @throws SecurityException
     * @throws IllegalArgumentException
     */
    private Object sendAndExecute(Method m, Object[] pValues, Object o, ObjectInputStream objIn,
        ObjectOutputStream objOut)
        throws IOException, ClassNotFoundException, IllegalArgumentException, SecurityException,
        IllegalAccessException, InvocationTargetException, NoSuchMethodException {

      // Send the object itself
      sendObject(o, m, pValues, objOut);

      // TODO To be more precise, this message should be sent by the AS, but for simplicity I put it
      // here.
      RapidUtils.sendAnimationMsg(config, AnimationMsg.AS_RUN_METHOD);

      // Read the results from the server
      Log.d(TAG, "Read Result");
      Object response = objIn.readObject();

      // TODO To be more precise, this message should be sent by the AS, but for simplicity I put it
      // here.
      RapidUtils.sendAnimationMsg(config, AnimationMsg.AS_RESULT_AC);

      ResultContainer container = (ResultContainer) response;
      Object result;

      Class<?>[] pTypes = {Remoteable.class};
      try {
        // Use the copyState method that must be defined for all Remoteable
        // classes to copy the state of relevant fields to the local object
        o.getClass().getMethod("copyState", pTypes).invoke(o, container.objState);
      } catch (NullPointerException e) {
        // Do nothing - exception happened remotely and hence there is
        // no object state returned.
        // The exception will be returned in the function result anyway.
        Log.d(TAG, "Exception received from remote server - " + container.functionResult);
      }

      result = container.functionResult;
      mPureRemoteDuration = container.pureExecutionDuration;

      Log.d(TAG, "Finished remote execution");

      return result;
    }
  }

  /**
   * Send APK file to the remote server
   * 
   * @param apkName file name of the APK file (full path)
   * @param objOut ObjectOutputStream to write the file to
   * @throws IOException
   * @throws NameNotFoundException
   */
  private void sendApk() {

    try {
      Log.d(TAG, "Getting apk data");
      String apkName = mPManager.getApplicationInfo(mAppName, 0).sourceDir;
      File apkFile = new File(apkName);
      Log.d(TAG, "Apk name - " + apkName);

      mOutStream.write(RapidMessages.AC_REGISTER_AS);
      // Send apkName and apkLength to clone.
      // The clone will compare these information with what he has and tell
      // if he doesn't have the apk or this one differs in size.
      mObjOutStream.writeObject(mAppName);
      mObjOutStream.writeInt((int) apkFile.length());
      mObjOutStream.flush();
      int response = mInStream.read();

      if (response == RapidMessages.AS_APP_REQ_AC) {
        // Send the APK file if needed

        FileInputStream fin = new FileInputStream(apkFile);
        BufferedInputStream bis = new BufferedInputStream(fin);

        // Send the file
        Log.d(TAG, "Sending apk");
        int BUFFER_SIZE = 8192;
        byte[] tempArray = new byte[BUFFER_SIZE];
        int read = 0;
        while ((read = bis.read(tempArray, 0, tempArray.length)) > -1) {
          mObjOutStream.write(tempArray, 0, read);
          // Log.d(TAG, "Sent " + totalRead + " of " + apkFile.length() + " bytes");
        }
        mObjOutStream.flush();
        RapidUtils.closeQuietly(bis);
      }
    } catch (IOException e) {
      fallBackToLocalExecution("IOException: " + e.getMessage());
    } catch (NameNotFoundException e) {
      fallBackToLocalExecution("Application not found: " + e.getMessage());
    } catch (Exception e) {
      fallBackToLocalExecution("Exception: " + e.getMessage());
    }
  }


  /**
   * Take care of a broken connection - try restarting it when something breaks down immediately or
   * alternatively listen to changing network conditions
   */
  public class ConnectionRepair extends Thread {
    /**
     * Simple reestablish the connection
     */
    @Override
    public void run() {
      // Try simply restarting the connection
      Log.d(TAG, "Trying to reestablish connection to the server");
      synchronized (this) {
        if (!onLineClear && !onLineSSL) {
          // Show a spinning dialog while connecting to the Manager and to the clone.
          // pd = ProgressDialog.show(mContext, "Working..", "Initial network tasks...", true,
          // false);
          (new InitialNetworkTasks()).execute(sClone);

          // establishConnection();
        }
      }

      // If still offline, establish intent listeners that would try to
      // restart the connection when the service comes back up
      synchronized (this) {
        if (!onLineClear && !onLineSSL) {
          Log.d(TAG, "Reestablishing failed - register listeners for reconnecting");
          final ConnectivityManager connectivityManager =
              (ConnectivityManager) mContext.getSystemService(Context.CONNECTIVITY_SERVICE);

          BroadcastReceiver networkStateReceiver = new BroadcastReceiver() {
            public void onReceive(Context context, Intent intent) {
              context.unregisterReceiver(this);
              NetworkInfo netInfo = connectivityManager.getActiveNetworkInfo();
              if (netInfo != null) {
                Log.d(TAG, "Network back up, try reestablishing the connection");
                // establishConnection();

                pd = ProgressDialog.show(mContext, "Working..", "Initial network tasks...", true,
                    false);
                (new InitialNetworkTasks()).execute(sClone);
              }
            }
          };
          IntentFilter networkStateFilter =
              new IntentFilter(ConnectivityManager.CONNECTIVITY_ACTION);
          mContext.registerReceiver(networkStateReceiver, networkStateFilter);
        }
      }
    }
  }

  public String getConnectionType() {
    return NetworkProfiler.currentNetworkTypeName;
  }

  public void setUserChoice(int userChoice) {
    this.userChoice = userChoice;
    mDSE.setUserChoice(userChoice);
  }

  public int getUserChoice() {
    return userChoice;
  }

  public int getRegime() {
    return mRegime;
  }

  public void setNrClones(int nrClones) {
    Log.i(TAG, "Changing nrClones to: " + nrClones);
    this.nrClones = nrClones;
  }

  public Frontend getGvirtusFrontend() {
    return gVirtusFrontend;
  }

  public void setGvirtusFrontend(Frontend gVirtusFrontend) {
    this.gVirtusFrontend = gVirtusFrontend;
  }

  public Configuration getConfig() {
    return config;
  }

  public Context getContext() {
    return mContext;
  }

  public void setDataRate(int dataRate) {
    NetworkProfiler.setDataRate(dataRate);
  }

  /**
   * Used to measure the costs of connection with the clone when using different communication
   * types.
   * 
   * @param givenCommType CLEAR, SSL
   * @param buffLogFile
   * @throws IOException
   */
  public void testConnection(COMM_TYPE givenCommType, BufferedWriter buffLogFile)
      throws IOException {

    if (onLineClear || onLineSSL) {
      closeConnection();
    }
    commType = givenCommType;

    long startTime = System.nanoTime();

    if (givenCommType == COMM_TYPE.SSL) {
      establishSslConnection();
    } else {
      establishConnection();
    }

    long totalTime = System.nanoTime() - startTime;

    if (buffLogFile != null) {
      buffLogFile.write(totalTime + "\n");
    }
  }

  /**
   * Used to measure the costs of sending data of different size with different communication
   * protocols.
   * 
   * @param givenCommType CLEAR, SSL
   * @param nrBytesToSend
   * @param bytesToSend
   * @param buffLogFile
   * @return
   * @throws IOException
   */
  public long testSendBytes(int nrBytesToSend, byte[] bytesToSend, BufferedWriter buffLogFile)
      throws IOException {

    long txTime = -1;
    long txBytes = -1;

    switch (nrBytesToSend) {
      case 1:
        txBytes = NetworkProfiler.getProcessTxBytes();
        txTime = System.nanoTime();
        mOutStream.write(RapidMessages.PING);
        mInStream.read();
        txTime = System.nanoTime() - txTime;
        txBytes = NetworkProfiler.getProcessTxBytes() - txBytes;
        break;

      case 4:
        mOutStream.write(RapidMessages.SEND_INT);
        // sleep(3*1000);

        txBytes = NetworkProfiler.getProcessTxBytes();
        txTime = System.nanoTime();
        mObjOutStream.writeInt((int) System.currentTimeMillis());
        mObjOutStream.flush();
        mInStream.read();
        txTime = System.nanoTime() - txTime;
        txBytes = NetworkProfiler.getProcessTxBytes() - txBytes;

        // txTime = mObjInStream.readLong();

        break;

      default:
        mOutStream.write(RapidMessages.SEND_BYTES);
        // sleep(3*1000);

        txBytes = NetworkProfiler.getProcessTxBytes();
        txTime = System.nanoTime();
        mObjOutStream.writeObject(bytesToSend);
        mObjOutStream.flush();
        mInStream.read();
        txTime = System.nanoTime() - txTime;
        txBytes = NetworkProfiler.getProcessTxBytes() - txBytes;
        // txTime = mObjInStream.readLong();

        break;
    }

    if (buffLogFile != null) {
      buffLogFile.write(txTime + "\n");
      buffLogFile.flush();
    }

    Log.i(TAG, "Sent " + nrBytesToSend + " bytes in " + txTime / 1000000000.0 + " seconds.");

    return txBytes;
  }

  public long testReceiveBytes(int nrBytesToReceive, BufferedWriter buffLogFile)
      throws IOException, ClassNotFoundException {

    long rxBytes = -1;
    long rxTime = -1;

    switch (nrBytesToReceive) {
      case 1:
        mOutStream.write(RapidMessages.PING);

        rxBytes = NetworkProfiler.getProcessRxBytes();
        rxTime = System.nanoTime();
        mInStream.read();
        rxTime = System.nanoTime() - rxTime;
        rxBytes = NetworkProfiler.getProcessRxBytes() - rxBytes;
        break;

      case 4:
        mOutStream.write(RapidMessages.RECEIVE_INT);

        rxBytes = NetworkProfiler.getProcessRxBytes();
        // rxTime = System.nanoTime();
        mObjInStream.readInt();
        mOutStream.write(1);
        mOutStream.flush();
        rxTime = mObjInStream.readLong();

        // sleep(8*1000);
        // rxTime = System.nanoTime() - rxTime;
        rxBytes = NetworkProfiler.getProcessRxBytes() - rxBytes;

        break;

      default:
        mOutStream.write(RapidMessages.RECEIVE_BYTES);
        mObjOutStream.writeInt(nrBytesToReceive);
        mObjOutStream.flush();

        rxBytes = NetworkProfiler.getProcessRxBytes();
        // rxTime = System.nanoTime();
        mObjInStream.readObject();
        mOutStream.write(1);
        mOutStream.flush();
        rxTime = mObjInStream.readLong();

        // sleep(8*1000);
        // rxTime = System.nanoTime() - rxTime;
        rxBytes = NetworkProfiler.getProcessRxBytes() - rxBytes;

        break;
    }

    if (buffLogFile != null) {
      buffLogFile.write(rxTime + "\n");
      buffLogFile.flush();
    }

    Log.i(TAG, "Received " + nrBytesToReceive + " bytes in " + rxTime / 1000000000.0 + " seconds.");

    return rxBytes;
  }

  @SuppressWarnings("unused")
  private void sleep(long millis) {
    try {
      Thread.sleep(millis);
    } catch (InterruptedException e) {
    }
  }
}
