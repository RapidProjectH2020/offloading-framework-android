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
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

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
import eu.project.rapid.ac.d2d.D2DService;
import eu.project.rapid.ac.db.DatabaseQuery;
import eu.project.rapid.ac.profilers.DeviceProfiler;
import eu.project.rapid.ac.profilers.LogRecord;
import eu.project.rapid.ac.profilers.NetworkProfiler;
import eu.project.rapid.ac.profilers.Profiler;
import eu.project.rapid.ac.profilers.ProgramProfiler;
import eu.project.rapid.ac.utils.Constants;
import eu.project.rapid.common.Clone;
import eu.project.rapid.common.Configuration;
import eu.project.rapid.common.RapidMessages;
import eu.project.rapid.common.RapidUtils;

/**
 * The interface to the framework for the client program - controls DSE, profilers, communicates
 * with remote server
 * 
 */

public class DFE {

  private static final String TAG = "DFE";

  public static boolean CONNECT_TO_PREVIOUS_VM = true;
  public static final int COMM_CLEAR = 1;
  public static final int COMM_SSL = 2;

  private Configuration config;
  public static final int SDK_INT = Build.VERSION.SDK_INT;

  private static int mRegime;
  public static final int REGIME_CLIENT = 1;
  public static final int REGIME_SERVER = 2;
  public static int commType = DFE.COMM_CLEAR;
  private int userChoice = Constants.LOCATION_LOCAL;
  private double localDataFraction = 1;
  private Method prepareDataMethod = null;

  private long mPureLocalDuration;
  private long mPureRemoteDuration;
  private long prepareDataDuration;
  private Long totalTxBytesObject;

  // private Object result;
  private String mAppName;
  private Context mContext;
  private PackageManager mPManager;

  public static boolean onLine = false;
  public static boolean dfeReady = false;
  private int nrClones;
  private DSE mDSE;
  private DeviceProfiler mDevProfiler;
  private NetworkProfiler netProfiler;

  private static Clone sClone;
  private static Socket mSocket;
  private static OutputStream mOutStream;
  private static ObjectOutputStream mObjOutStream;
  private static InputStream mInStream;
  private static DynamicObjectInputStream mObjInStream;

  public LogRecord lastLogRecord;
  private int myId = -1;
  private int myIdWithDS = -1;

  private ProgressDialog pd = null;

  /**
   * Interface to be implemented by some class that wants to be updated about some events.
   * 
   * @author sokol
   *
   */
  public interface DfeCallback {
    public void vmConnectionStatusUpdate(); // Get updates about the VM connection status.
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

    readConfigurationFile();
    initializeCrypto();

    // This makes the framework reconnect to the previous clone
    if (CONNECT_TO_PREVIOUS_VM) {
      SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this.mContext);
      myId = prefs.getInt(Constants.MY_OLD_ID, -1);
      myIdWithDS = prefs.getInt(Constants.MY_OLD_ID_WITH_DS, -1);
    }

    mDSE = new DSE(mContext, userChoice);

    mDevProfiler = DeviceProfiler.getInstance(context);
    // mDevProfiler.trackBatteryLevel();
    netProfiler = new NetworkProfiler(context, config);
    netProfiler.registerNetworkStateTrackers();

    // Create the database
    DatabaseQuery query = new DatabaseQuery(context);
    try {
      // Close the database
      query.destroy();
    } catch (Throwable e) {
      Log.e(TAG, "Could not close the database: " + e.getMessage());
    }

    // Start the service that will deal with the D2D communication
    startD2DService();

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

  private void readConfigurationFile() {

    File rapidDir = new File(Constants.RAPID_FOLDER);
    if (!rapidDir.exists()) {
      if (!rapidDir.mkdirs()) {
        Log.w(TAG, "Could not create the Rapid folder: " + Constants.RAPID_FOLDER);
      }
    }

    try {
      // Read the config file to read the IP and port of Manager
      config = new Configuration(Constants.PHONE_CONFIG_FILE);
      config.parseConfigFile();
    } catch (FileNotFoundException e) {
      // e.printStackTrace();
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

    } catch (KeyStoreException e) {
      Log.e(TAG, "Could not initialize the crypto parameters - " + e);
    } catch (NoSuchAlgorithmException e) {
      Log.e(TAG, "Could not initialize the crypto parameters - " + e);
    } catch (CertificateException e) {
      Log.e(TAG, "Could not initialize the crypto parameters - " + e);
    } catch (FileNotFoundException e) {
      Log.e(TAG, "Could not initialize the crypto parameters - " + e);
    } catch (IOException e) {
      Log.e(TAG, "Could not initialize the crypto parameters - " + e);
    } catch (KeyManagementException e) {
      Log.e(TAG, "Could not initialize the crypto parameters - " + e);
    } catch (UnrecoverableKeyException e) {
      Log.e(TAG, "Could not initialize the crypto parameters - " + e);
    }
  }

  private class InitialNetworkTasks extends AsyncTask<Clone, String, Void> {

    private static final String TAG = "InitialNetworkTasks";

    @Override
    protected Void doInBackground(Clone... clone) {
      if (clone[0] == null) {
        publishProgress("Connecting to the DS to get available VM Managers");
        getInfoFromDS();

        publishProgress("Connecting to the Manager " + config.getManagerIp() + ":"
            + config.getManagerPort() + " to ask for the clone");
        sClone = getInfoFromManager();
      } else {
        publishProgress("Using the clone given by the user: " + clone[0]);
        sClone = clone[0];
      }

      config.setClone(sClone);

      RapidUtils.sendAnimationMsg(config, RapidMessages.AC_REGISTER_VM);

      if (commType == DFE.COMM_CLEAR) {
        publishProgress("Clear connection with the clone: " + sClone);
        establishConnection();
      } else { // (commType == RapidMessages.COMM_SSL)
        publishProgress("SSL connection with the clone: " + sClone);
        if (!establishSslConnection()) {
          Log.w(TAG, "Setting commType to CLEAR");
          commType = DFE.COMM_CLEAR;
          establishConnection();
        }
      }

      // If the connection was successful then try to send the app to the clone
      if (onLine) {
        Log.i(TAG, "The communication type established with the clone is: " + commType);

        publishProgress("Registering the APK with the clone...");
        RapidUtils.sendAnimationMsg(config, RapidMessages.AC_SEND_APK);
        sendApk();

        // Find rtt to the server
        // Measure the data rate when just connected
        publishProgress("Sending/receiving data for 3 seconds to measure the ulRate and dlRate...");
        RapidUtils.sendAnimationMsg(config, RapidMessages.AC_RTT_MEASUREMENT);
        NetworkProfiler.rttPing(mInStream, mOutStream);
        RapidUtils.sendAnimationMsg(config, RapidMessages.AC_DL_MEASUREMENT);
        NetworkProfiler.measureDlRate();
        RapidUtils.sendAnimationMsg(config, RapidMessages.AC_UL_MEASUREMENT);
        NetworkProfiler.measureUlRate();
        RapidUtils.sendAnimationMsg(config, RapidMessages.AC_REGISTER_VM_OK);

        try {
          ((DfeCallback) mContext).vmConnectionStatusUpdate();
        } catch (ClassCastException e) {
          Log.i(TAG, "This class doesn't implement callback methods.");
        }
      }

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
  }

  public void onDestroy() {
    dfeReady = false;
    mDevProfiler.onDestroy();
    netProfiler.onDestroy();

    closeConnection();
  }

  private void startD2DService() {
    Log.i(TAG, "Sending intent to start the D2D service.");
    Intent d2dServiceIntent = new Intent(mContext, D2DService.class);
    mContext.startService(d2dServiceIntent);
  }

  /**
   * Read the config file to get the IP and port of the DS. The DS will return a list of available
   * VMMs, choose the best one from the list and connect to it to ask for a VM.
   * 
   * @throws IOException
   * @throws UnknownHostException
   * @throws ClassNotFoundException
   */
  private void getInfoFromDS() {

    Log.d(TAG, "Starting as phone with ID: " + myId);

    Socket dsSocket = null;
    ObjectOutputStream oos = null;
    ObjectInputStream ois = null;

    try {
      dsSocket = new Socket();
      dsSocket.connect(new InetSocketAddress(config.getDSIp(), config.getDSPort()), 20 * 1000);

      OutputStream os = dsSocket.getOutputStream();
      InputStream is = dsSocket.getInputStream();

      os.write(RapidMessages.PHONE_CONNECTION);

      oos = new ObjectOutputStream(os);
      ois = new ObjectInputStream(is);

      // Send the name and id to the DS
      if (CONNECT_TO_PREVIOUS_VM) {
        os.write(RapidMessages.AC_REGISTER_PREV_DS);
      } else {
        os.write(RapidMessages.AC_REGISTER_NEW_DS);
      }

      oos.writeInt(myIdWithDS);
      oos.flush();

      myIdWithDS = ois.readInt();
      int vmmId = ois.readInt();
      String vmmIp = ois.readUTF();
      config.setManagerPort(ois.readInt());
      config.setAnimationServerIp(ois.readUTF());
      config.setAnimationServerPort(ois.readInt());

      // If the VM Manager is running on the same machine as the DS then the DS may return
      // localhost as vmmIP.
      if (vmmIp.equalsIgnoreCase("localhost") || vmmIp.equals("127.0.0.1")) {
        vmmIp = config.getDSIp();
      }
      config.setManagerIp(vmmIp);

      Log.i(TAG, "Saving my ID with the DS: " + myIdWithDS);
      SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(mContext);
      SharedPreferences.Editor editor = prefs.edit();
      editor.putInt(Constants.MY_OLD_ID_WITH_DS, myIdWithDS);
      editor.commit();

    } catch (IOException e) {
      Log.e(TAG, "IOException while talking to the DS: " + e);
      // e.printStackTrace();
    } catch (Exception e) {
      Log.e(TAG, "Exception while talking to the DS: " + e);
      // e.printStackTrace();
    } finally {
      RapidUtils.closeQuietly(oos);
      RapidUtils.closeQuietly(ois);
      RapidUtils.closeQuietly(dsSocket);
    }
  }

  /**
   * Read the config file to get the IP and port of Manager.
   * 
   * @throws IOException
   * @throws UnknownHostException
   * @throws ClassNotFoundException
   */
  private Clone getInfoFromManager() {

    Log.d(TAG, "Starting as phone with ID: " + myId);

    Socket managerSocket = null;
    ObjectOutputStream oos = null;
    ObjectInputStream ois = null;
    Clone clone = null;

    try {
      managerSocket = new Socket();
      managerSocket.connect(new InetSocketAddress(config.getManagerIp(), config.getManagerPort()),
          5 * 1000);

      OutputStream os = managerSocket.getOutputStream();
      InputStream is = managerSocket.getInputStream();

      os.write(RapidMessages.PHONE_CONNECTION);

      oos = new ObjectOutputStream(os);
      ois = new ObjectInputStream(is);

      if (CONNECT_TO_PREVIOUS_VM) {
        RapidUtils.sendAnimationMsg(config, RapidMessages.AC_REGISTER_VMM_PREV);
      } else {
        RapidUtils.sendAnimationMsg(config, RapidMessages.AC_REGISTER_VMM_NEW);
      }

      // Send the name and id to the manager
      os.write(RapidMessages.PHONE_AUTHENTICATION);
      oos.writeInt(myId);
      oos.flush();

      clone = (Clone) ois.readObject();
      if (clone.getName() == null) {
        Log.w(TAG, "Manager could not assign a clone using id: " + myId);
      } else {
        myId = clone.getId();

        Log.i(TAG, "Saving my ID: " + myId);
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(mContext);
        SharedPreferences.Editor editor = prefs.edit();
        editor.putInt(Constants.MY_OLD_ID, myId);
        editor.commit();
      }

    } catch (IOException e) {
      Log.e(TAG, "IOException while talking to the manager: " + e);
      // e.printStackTrace();
    } catch (ClassNotFoundException e) {
      Log.e(TAG, "ClassNotFoundException while receiving the clone from the manager: " + e);
    } catch (Exception e) {
      Log.e(TAG, "Exception while talking to the manager: " + e);
    } finally {
      RapidUtils.closeQuietly(oos);
      RapidUtils.closeQuietly(ois);
      RapidUtils.closeQuietly(managerSocket);
    }

    return clone;
  }

  /**
   * Set up streams for the socket connection, perform initial communication with the clone.
   */
  private boolean establishConnection() {
    try {
      long sTime = System.nanoTime();
      long startTxBytes = NetworkProfiler.getProcessTxBytes();
      long startRxBytes = NetworkProfiler.getProcessRxBytes();

      RapidUtils.sendAnimationMsg(config, RapidMessages.AC_CONNECT_VM);

      mSocket = new Socket();
      mSocket.connect(new InetSocketAddress(sClone.getIp(), sClone.getPort()), 10 * 1000);

      mOutStream = mSocket.getOutputStream();
      mInStream = mSocket.getInputStream();
      mObjOutStream = new ObjectOutputStream(mOutStream);
      mObjInStream = new DynamicObjectInputStream(mInStream);

      onLine = true;

      long dur = System.nanoTime() - sTime;
      long totalTxBytes = NetworkProfiler.getProcessTxBytes() - startTxBytes;
      long totalRxBytes = NetworkProfiler.getProcessRxBytes() - startRxBytes;

      Log.d(TAG, "Socket and streams set-up time - " + dur / 1000000 + "ms");
      Log.d(TAG, "Total bytes sent: " + totalTxBytes);
      Log.d(TAG, "Total bytes received: " + totalRxBytes);

    } catch (UnknownHostException e) {
      fallBackToLocalExecution("Connection setup with the clone failed: " + e);
    } catch (IOException e) {
      fallBackToLocalExecution("Connection setup with the clone failed: " + e);
    } catch (Exception e) {
      fallBackToLocalExecution("Could not connect with the clone: " + e);
    }

    return true;

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

    try {
      RapidUtils.sendAnimationMsg(config, RapidMessages.AC_CONNECT_VM);

      Long sTime = System.nanoTime();
      long startTxBytes = NetworkProfiler.getProcessTxBytes();
      long startRxBytes = NetworkProfiler.getProcessRxBytes();

      Log.i(TAG, "Trying to connect to clone: " + sClone.getIp() + ":" + sClone.getSslPort());

      mSocket = config.getSslFactory().createSocket(sClone.getIp(), sClone.getSslPort());
      // Log.i(TAG, "getEnableSessionCreation: " + ((SSLSocket)
      // mSocket).getEnableSessionCreation());
      // ((SSLSocket) mSocket).setEnableSessionCreation(false);

      // sslContext.getClientSessionContext().getSession(null).invalidate();

      ((SSLSocket) mSocket).addHandshakeCompletedListener(new MyHandshakeCompletedListener());
      Log.i(TAG, "socket created");

      // Log.i(TAG, "Enabled cipher suites: ");
      // for (String s : ((SSLSocket) mSocket).getEnabledCipherSuites()) {
      // Log.i(TAG, s);
      // }

      mOutStream = mSocket.getOutputStream();
      mInStream = mSocket.getInputStream();
      mObjOutStream = new ObjectOutputStream(mOutStream);
      mObjInStream = new DynamicObjectInputStream(mInStream);

      onLine = true;

      long dur = System.nanoTime() - sTime;
      long totalTxBytes = NetworkProfiler.getProcessTxBytes() - startTxBytes;
      long totalRxBytes = NetworkProfiler.getProcessRxBytes() - startRxBytes;

      Log.d(TAG, "Socket and streams set-up time - " + dur / 1000000 + "ms");
      Log.d(TAG, "Total bytes sent: " + totalTxBytes);
      Log.d(TAG, "Total bytes received: " + totalRxBytes);

    } catch (UnknownHostException e) {
      fallBackToLocalExecution("Connection setup to server failed: " + e);
    } catch (IOException e) {
      fallBackToLocalExecution("Connection setup to server failed: " + e);
    } catch (Exception e) {
      fallBackToLocalExecution("Connection setup to server failed: " + e);
    }

    return true;
  }

  private class MyHandshakeCompletedListener implements HandshakeCompletedListener {

    @Override
    public void handshakeCompleted(HandshakeCompletedEvent event) {
      Log.i(TAG, "handshake completed");

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
        Log.e(TAG, "handshakeCompleted: " + e);
      }
    }

  }

  private void closeConnection() {

    RapidUtils.sendAnimationMsg(config, RapidMessages.AC_DISCONNECT_VM);

    RapidUtils.closeQuietly(mObjOutStream);
    RapidUtils.closeQuietly(mObjInStream);
    RapidUtils.closeQuietly(mSocket);
    onLine = false;
  }

  private void fallBackToLocalExecution(String message) {
    Log.e(TAG, message);

    onLine = false;
  }

  /**
   * Call the execution solver to connect to the Manager for performing the data-partition
   * algorithm.
   * 
   * @param appName The application name.
   * @param methodName The current method that we want to offload from this application.<br>
   *        Different methods of the same application will have a different set of parameters.
   * @return The execution location which can be one of: LOCAL, REMOTE.<br>
   */
  public int findExecLocation(String appName, String methodName) {

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
  public int findExecLocation(String methodName) {
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
    // if (execLocation == RapidConstants.LOCATION_LOCAL) {
    // localDataFraction = 1;
    // }
    // else if (execLocation == RapidConstants.LOCATION_REMOTE) {
    // localDataFraction = 0;
    // }

    // Maybe the developer has implemented the prepareData(float) method that helps him prepare the
    // data based on where the execution will take place then call it.
    // Prepare the data by calling the prepareData(localFraction) implemented by the developer.
    try {
      // long s = System.nanoTime();
      prepareDataMethod = o.getClass().getDeclaredMethod("prepareData", double.class);
      prepareDataMethod.setAccessible(true);
      // prepareDataMethod.invoke(o, localDataFraction);
      // prepareDataDuration = System.nanoTime() - s;
    } catch (NoSuchMethodException e) {
      Log.w(TAG, "The method prepareData() does not exist");
      prepareDataMethod = null;
    }

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
        // TODO Auto-generated catch block
        e.printStackTrace();
      } catch (ExecutionException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
      } catch (InvocationTargetException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
      }
    } else { // ecexLocation LOCAL or REMOTE
      Future<Object> futureTotalResult =
          executor.submit(new TaskRunner(execLocation, m, pValues, o));

      try {
        totalResult = futureTotalResult.get();
      } catch (InterruptedException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
      } catch (ExecutionException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
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

      if (execLocation == Constants.LOCATION_LOCAL) {
        try {
          this.result = executeLocally(m, pValues, o);
        } catch (IllegalArgumentException e) {
          // TODO Auto-generated catch block
          e.printStackTrace();
        } catch (IllegalAccessException e) {
          // TODO Auto-generated catch block
          e.printStackTrace();
        } catch (InvocationTargetException e) {
          // TODO Auto-generated catch block
          e.printStackTrace();
        }
      } else if (execLocation == Constants.LOCATION_REMOTE) {
        try {
          this.result = executeRemotely(m, pValues, o);
          if (this.result instanceof InvocationTargetException) {
            // The remote execution throwed an exception, try to run the method locally.
            Log.w(TAG,
                "The returned result was InvocationTargetException. Running the method locally");
            this.result = executeLocally(m, pValues, o);
          }
        } catch (IllegalArgumentException e) {
          // TODO Auto-generated catch block
          e.printStackTrace();
        } catch (SecurityException e) {
          // TODO Auto-generated catch block
          e.printStackTrace();
        } catch (IllegalAccessException e) {
          // TODO Auto-generated catch block
          e.printStackTrace();
        } catch (InvocationTargetException e) {
          // TODO Auto-generated catch block
          e.printStackTrace();
        } catch (ClassNotFoundException e) {
          // TODO Auto-generated catch block
          e.printStackTrace();
        } catch (NoSuchMethodException e) {
          // TODO Auto-generated catch block
          e.printStackTrace();
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

      RapidUtils.sendAnimationMsg(config, RapidMessages.AC_DECISION_LOCAL);

      localDataFraction = 1;
      if (prepareDataMethod != null) {
        RapidUtils.sendAnimationMsg(config, RapidMessages.AC_PREPARE_DATA);
        long s = System.nanoTime();
        prepareDataMethod.invoke(o, localDataFraction);
        prepareDataDuration = System.nanoTime() - s;
      }

      ProgramProfiler progProfiler = new ProgramProfiler(mAppName, m.getName());
      DeviceProfiler devProfiler = DeviceProfiler.getInstance(mContext);
      NetworkProfiler netProfiler = null;
      Profiler profiler = new Profiler(mRegime, mContext, progProfiler, netProfiler, devProfiler);

      // Start tracking execution statistics for the method
      profiler.startExecutionInfoTracking();

      // Make sure that the method is accessible
      RapidUtils.sendAnimationMsg(config, RapidMessages.AC_EXEC_LOCAL);
      Object result = null;
      Long startTime = System.nanoTime();
      m.setAccessible(true);
      result = m.invoke(o, pValues); // Access it
      mPureLocalDuration = System.nanoTime() - startTime;
      Log.d(TAG, "LOCAL " + m.getName() + ": Actual Invocation duration - "
          + mPureLocalDuration / 1000000 + "ms");

      RapidUtils.sendAnimationMsg(config, RapidMessages.AC_FINISHED_LOCAL);
      // Collect execution statistics
      profiler.stopAndLogExecutionInfoTracking(prepareDataDuration, mPureLocalDuration);
      lastLogRecord = profiler.lastLogRecord;

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

      RapidUtils.sendAnimationMsg(config, RapidMessages.AC_DECISION_REMOTE);

      localDataFraction = 0;
      if (prepareDataMethod != null) {
        RapidUtils.sendAnimationMsg(config, RapidMessages.AC_PREPARE_DATA);
        long s = System.nanoTime();
        prepareDataMethod.invoke(o, localDataFraction);
        prepareDataDuration = System.nanoTime() - s;
      }

      ProgramProfiler progProfiler = new ProgramProfiler(mAppName, m.getName());
      DeviceProfiler devProfiler = DeviceProfiler.getInstance(mContext);
      NetworkProfiler netProfiler = new NetworkProfiler();
      Profiler profiler = new Profiler(mRegime, mContext, progProfiler, netProfiler, devProfiler);

      // Start tracking execution statistics for the method
      profiler.startExecutionInfoTracking();

      try {
        Long startTime = System.nanoTime();
        mOutStream.write(RapidMessages.EXECUTE);
        RapidUtils.sendAnimationMsg(config, RapidMessages.AC_REMOTE_SEND_DATA);
        result = sendAndExecute(m, pValues, o, mObjInStream, mObjOutStream);

        Long duration = System.nanoTime() - startTime;
        Log.d(TAG, "REMOTE " + m.getName() + ": Actual Send-Receive duration - "
            + duration / 1000000 + "ms");
        // Collect execution statistics
        profiler.stopAndLogExecutionInfoTracking(prepareDataDuration, mPureRemoteDuration);
        lastLogRecord = profiler.lastLogRecord;
      } catch (Exception e) {
        // No such host exists, execute locally
        Log.e(TAG, "REMOTE ERROR: " + m.getName() + ": " + e);
        e.printStackTrace();
        result = executeLocally(m, pValues, o);
        ConnectionRepair repair = new ConnectionRepair();
        repair.start();
      }

      RapidUtils.sendAnimationMsg(config, RapidMessages.AC_FINISHED_REMOTE);
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

      Long startTx = NetworkProfiler.getProcessTxBytes();

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

      totalTxBytesObject = NetworkProfiler.getProcessTxBytes() - startTx;
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
    private Object sendAndExecute(Method m, Object[] pValues, Object o,
        DynamicObjectInputStream objIn, ObjectOutputStream objOut)
        throws IOException, ClassNotFoundException, IllegalArgumentException, SecurityException,
        IllegalAccessException, InvocationTargetException, NoSuchMethodException {

      // Send the object itself
      sendObject(o, m, pValues, objOut);

      // Read the results from the server
      Log.d(TAG, "Read Result");

      Long startSend = System.nanoTime();
      Long startRx = NetworkProfiler.getProcessRxBytes();
      Object response = objIn.readObject();

      // Estimate the perceived bandwidth
      NetworkProfiler.addNewDlRateEstimate(NetworkProfiler.getProcessRxBytes() - startRx,
          System.nanoTime() - startSend);

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

      // Estimate the perceived bandwidth
      NetworkProfiler.addNewUlRateEstimate(totalTxBytesObject, container.getObjectDuration);

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

      mOutStream.write(RapidMessages.APK_REGISTER);
      // Send apkName and apkLength to clone.
      // The clone will compare these information with what he has and tell
      // if he doesn't have the apk or this one differs in size.
      mObjOutStream.writeObject(mAppName);
      mObjOutStream.writeInt((int) apkFile.length());
      mObjOutStream.flush();
      int response = mInStream.read();

      if (response == RapidMessages.APK_REQUEST) {
        // Send the APK file if needed

        FileInputStream fin = new FileInputStream(apkFile);
        BufferedInputStream bis = new BufferedInputStream(fin);

        // Send the file
        long startTime = System.nanoTime();
        long startTxBytes = NetworkProfiler.getProcessTxBytes();
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

        NetworkProfiler.addNewUlRateEstimate(NetworkProfiler.getProcessTxBytes() - startTxBytes,
            System.nanoTime() - startTime);
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
        if (!onLine) {
          // Show a spinning dialog while connecting to the Manager and to the clone.
          pd = ProgressDialog.show(mContext, "Working..", "Initial network tasks...", true, false);
          (new InitialNetworkTasks()).execute(sClone);

          // establishConnection();
        }
      }

      // If still offline, establish intent listeners that would try to
      // restart the connection when the service comes back up
      synchronized (this) {
        if (!onLine) {
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

  public void setUserChoice(int userChoice) {
    this.userChoice = userChoice;
    mDSE.setUserChoice(userChoice);
  }

  public int getUserChoice() {
    return userChoice;
  }

  public void setDataRate(int dataRate) {
    if (netProfiler != null)
      netProfiler.setDataRate(dataRate);
  }

  public int getRegime() {
    return mRegime;
  }

  public void setNrClones(int nrClones) {
    Log.i(TAG, "Changing nrClones to: " + nrClones);
    this.nrClones = nrClones;
  }

  public Configuration getConfig() {
    return config;
  }

  public double getLocalDataFraction() {
    return localDataFraction;
  }

  public void setLocalDataFraction(double localDataFraction) {
    this.localDataFraction = localDataFraction;
  }

  /**
   * Used to measure the costs of connection with the clone when using different communication
   * types.
   * 
   * @param givenCommType CLEAR, SSL, SSL_NO_REUSE, PubPrivKey
   * @param buffLogFile
   * @throws IOException
   */
  public void testConnection(int givenCommType, BufferedWriter buffLogFile) throws IOException {

    if (onLine) {
      closeConnection();
    }
    commType = givenCommType;

    long startTime = System.nanoTime();

    if (givenCommType == DFE.COMM_SSL) {
      establishSslConnection();
    } else {
      establishConnection();
    }

    long totalTime = System.nanoTime() - startTime;

    if (buffLogFile != null)
      buffLogFile.write(totalTime + "\n");
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

        // txBytes = NetworkProfiler.getProcessTxBytes();
        txTime = System.nanoTime();
        mObjOutStream.writeInt((int) System.currentTimeMillis());
        mObjOutStream.flush();
        mInStream.read();
        txTime = System.nanoTime() - txTime;

        // sleep(7*1000);
        // txBytes = NetworkProfiler.getProcessTxBytes() - txBytes;

        // txTime = mObjInStream.readLong();

        break;

      default:
        mOutStream.write(RapidMessages.SEND_BYTES);
        // sleep(3*1000);

        // txBytes = NetworkProfiler.getProcessTxBytes();
        txTime = System.nanoTime();
        mObjOutStream.writeObject(bytesToSend);
        mObjOutStream.flush();
        mInStream.read();
        txTime = System.nanoTime() - txTime;
        // txBytes = NetworkProfiler.getProcessTxBytes() - txBytes;

        // sleep(57*1000);
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

        // rxBytes = NetworkProfiler.getProcessRxBytes();
        // rxTime = System.nanoTime();
        mObjInStream.readInt();
        mOutStream.write(1);
        mOutStream.flush();
        rxTime = mObjInStream.readLong();

        // sleep(8*1000);
        // rxTime = System.nanoTime() - rxTime;
        // rxBytes = NetworkProfiler.getProcessRxBytes() - rxBytes;

        break;

      default:
        mOutStream.write(RapidMessages.RECEIVE_BYTES);
        mObjOutStream.writeInt(nrBytesToReceive);
        mObjOutStream.flush();

        // rxBytes = NetworkProfiler.getProcessRxBytes();
        // rxTime = System.nanoTime();
        mObjInStream.readObject();
        mOutStream.write(1);
        mOutStream.flush();
        rxTime = mObjInStream.readLong();

        // sleep(8*1000);
        // rxTime = System.nanoTime() - rxTime;
        // rxBytes = NetworkProfiler.getProcessRxBytes() - rxBytes;

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
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }
}
