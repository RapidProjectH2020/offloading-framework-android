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
package eu.project.rapid.ac.utils;

import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.RandomAccessFile;
import java.io.StreamCorruptedException;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.InterfaceAddress;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.charset.Charset;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import android.content.Context;
import android.telephony.TelephonyManager;
import android.util.Log;

public class Utils {

  private static final String TAG = "RapidUtils";
  private final static char[] hexArray = "0123456789ABCDEF".toCharArray();
  private final static Map<Character, Integer> hexToDec = new HashMap<Character, Integer>();
  private static Object syncFeatureObject = new Object();

  static {
    hexToDec.put('A', 10);
    hexToDec.put('B', 11);
    hexToDec.put('C', 12);
    hexToDec.put('D', 13);
    hexToDec.put('E', 14);
    hexToDec.put('F', 15);
  }

  public static String bytesToHex(byte[] bytes) {

    char[] hexChars = new char[bytes.length * 2];
    for (int j = 0; j < bytes.length; j++) {
      int v = bytes[j] & 0xFF;
      hexChars[j * 2] = hexArray[v >>> 4];
      hexChars[j * 2 + 1] = hexArray[v & 0x0F];
    }
    return new String(hexChars).trim();
  }

  public static byte[] hexToBytes(String hexString) {
    byte[] bytes = new byte[hexString.length() / 2];

    for (int i = 0; i < hexString.length(); i += 2) {
      char c1 = hexString.charAt(i);
      char c2 = hexString.charAt(i + 1);

      int n1 = Character.getNumericValue(c1);
      if (n1 < 0) {
        n1 = hexToDec.get(c1);
      }

      int n2 = Character.getNumericValue(c2);
      if (n2 < 0) {
        n2 = hexToDec.get(c2);
      }

      bytes[i / 2] = (byte) (n1 * 16 + n2);
    }

    return bytes;
  }

  public static byte[] objectToByteArray(Object o) throws IOException {
    byte[] bytes = null;
    ByteArrayOutputStream bos = null;
    ObjectOutputStream oos = null;
    try {
      bos = new ByteArrayOutputStream();
      oos = new ObjectOutputStream(bos);
      oos.writeObject(o);
      oos.flush();
      bytes = bos.toByteArray();
    } finally {
      if (oos != null) {
        oos.close();
      }
      if (bos != null) {
        bos.close();
      }
    }
    return bytes;
  }

  public static Object byteArrayToObject(byte[] bytes)
      throws StreamCorruptedException, IOException, ClassNotFoundException {
    Object obj = null;
    ByteArrayInputStream bis = null;
    ObjectInputStream ois = null;
    try {
      bis = new ByteArrayInputStream(bytes);
      ois = new ObjectInputStream(bis);
      obj = ois.readObject();
    } finally {
      if (bis != null) {
        bis.close();
      }
      if (ois != null) {
        ois.close();
      }
    }
    return obj;
  }

  /**
   * @param context The context of the app calling this method.
   * @return The unique ID of this device, which usually is the IMEI.
   */
  private static String getDeviceId(Context context) {
    TelephonyManager telephonyManager =
        (TelephonyManager) context.getSystemService(Context.TELEPHONY_SERVICE);
    String deviceId = telephonyManager.getDeviceId();

    return (deviceId != null ? deviceId : "null");
  }

  public static String getDeviceIdHashHex(Context context) {
    return sha256HashHex(getDeviceId(context));
  }

  public static byte[] getDeviceIdHashByteArray(Context context) {
    return sha256HashByteArray(getDeviceId(context));
  }

  /**
   * read the file: /sys/devices/system/cpu/possible<br>
   * output: 0-n
   * 
   * @return (n+1) or -1 if file not found.
   */
  public static int getDeviceNrCPUs() {
    // read the file: /sys/devices/system/cpu/possible
    // output: 0-3
    Scanner s = null;
    String fileName = "/sys/devices/system/cpu/possible";
    try {
      s = (new Scanner(new File(fileName)));
      s.useDelimiter("[-\n]");
      s.nextInt();
      return s.nextInt() + 1;
    } catch (Exception e) {
      Log.e(TAG, "Could not read number of CPUs: " + e);
    } finally {
      if (s != null) {
        s.close();
      }
    }

    return -1;
  }

  /**
   * read the file: /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq<br>
   * output: 1190400 (in KHz)
   * 
   * @return The frequency of cpu0 in KHz or -1 if file not found
   */
  public static int getDeviceCPUFreq() {
    String fileName = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq";
    Scanner s = null;
    try {
      s = (new Scanner(new File(fileName)));
      return s.nextInt();
    } catch (Exception e) {
      Log.e(TAG, "Could not read frequence: " + e);
    } finally {
      if (s != null) {
        s.close();
      }
    }

    return -1;
  }

  /**
   * @param s The string to be hash-ed.
   * @return The byte array hash digested result.
   */
  public static byte[] sha256HashByteArray(String s) {
    byte[] hash = null;
    try {
      MessageDigest md = MessageDigest.getInstance("SHA-256");
      hash = md.digest(s.getBytes());
    } catch (NoSuchAlgorithmException e) {
      Log.e(TAG, "Algorithm not found while hashing string: " + e);
      // e.printStackTrace();
    }

    return hash;
  }

  public static String sha256HashHex(String s) {
    return bytesToHex(sha256HashByteArray(s));
  }

  /**
   * This utility method will be used to write an object on a file. The object can be a Set, a Map,
   * etc.<br>
   * The method creates a lock file (if it doesn't exist) and tries to get a <b>blocking lock</b> on
   * the lock file. After writing the object the lock is released, so that other processes that want
   * to read the file can access it by getting the lock.
   * 
   * @param filePath The full path of the file where to write the object. If the file exists it will
   *        first be deleted and then created from scratch.
   * @param obj The object to write on the file.
   * @throws IOException
   */
  public static void writeObjectToFile(String filePath, Object obj) throws IOException {
    // Get a lock on the lockFile so that concurrent DFEs don't mess with each other by
    // reading/writing the d2dSetFile.
    File lockFile = new File(filePath + ".lock");
    // Create a FileChannel that can read and write that file.
    // This will create the file if it doesn't exit.
    RandomAccessFile file = new RandomAccessFile(lockFile, "rw");
    FileChannel f = file.getChannel();

    // Try to get an exclusive lock on the file.
    // FileLock lock = f.tryLock();
    FileLock lock = f.lock();

    // Now we have the lock, so we can write on the file
    File outFile = new File(filePath);
    if (outFile.exists()) {
      outFile.delete();
    }
    outFile.createNewFile();
    FileOutputStream fout = new FileOutputStream(outFile);
    ObjectOutputStream oos = new ObjectOutputStream(fout);
    oos.writeObject(obj);
    oos.close();

    // Now we release the lock and close the lockFile
    lock.release();
    file.close();
  }

  /**
   * Reads the previously serialized object from the <code>filename</code>.<br>
   * This method will try to get a <b>non blocking lock</b> on a lock file.
   * 
   * @param filePath The full path of the file from where to read the object.
   * @return The serialized object previously written using the method
   *         <code>writeObjectToFile</code>
   * @throws IOException
   * @throws ClassNotFoundException
   */
  public static Object readObjectFromFile(String filePath)
      throws IOException, ClassNotFoundException {
    Object obj = null;

    // First try to get the lock on a lock file
    File lockFile = new File(filePath + ".lock");
    if (!lockFile.exists()) {
      // It means that no other process has written an object before.
      return null;
    }

    // Create a FileChannel that can read and write that file.
    // This will create the file if it doesn't exit.
    RandomAccessFile file = new RandomAccessFile(lockFile, "rw");
    FileChannel f = file.getChannel();

    // Try to get an exclusive lock on the file.
    // FileLock lock = f.tryLock();
    FileLock lock = f.lock();

    // Now we have the lock, so we can read from the file
    File inFile = new File(filePath);
    FileInputStream fis = new FileInputStream(inFile);
    ObjectInputStream ois = new ObjectInputStream(fis);
    obj = ois.readObject();
    ois.close();

    // Now we release the lock and close the lockFile
    lock.release();
    file.close();
    return obj;
  }

  /**
   * Get IP address from first non-localhost interface
   * 
   * @return address or null
   */
  public static InetAddress getIpAddress() {
    try {
      List<NetworkInterface> interfaces = Collections.list(NetworkInterface.getNetworkInterfaces());
      for (NetworkInterface intf : interfaces) {
        // Log.i(TAG, "Interface: " + intf);
        List<InetAddress> addrs = Collections.list(intf.getInetAddresses());
        for (InetAddress addr : addrs) {
          // Sokol: FIXME remove the hard coded "wlan" check
          // Log.i(TAG, "IP: " + addr);
          if (intf.getDisplayName().contains("wlan") && !addr.isLoopbackAddress()
          // && InetAddressUtils.isIPv4Address(addr.getHostAddress())) {
              && addr instanceof Inet4Address) {
            return addr;
          }
          // On emulator
          if (intf.getDisplayName().contains("eth0") && !addr.isLoopbackAddress()
          // && InetAddressUtils.isIPv4Address(addr.getHostAddress())) {
              && addr instanceof Inet4Address) {
            return addr;
          }
        }
      }
    } catch (Exception e) {
      Log.i(TAG, "Exception while getting IP address: " + e);
    }
    return null;
  }

  public static InetAddress getBroadcast(InetAddress myIpAddress) {

    NetworkInterface temp;
    InetAddress iAddr = null;
    try {
      temp = NetworkInterface.getByInetAddress(myIpAddress);
      List<InterfaceAddress> addresses = temp.getInterfaceAddresses();

      for (InterfaceAddress inetAddress : addresses) {
        iAddr = inetAddress.getBroadcast();
      }
      Log.d(TAG, "iAddr=" + iAddr);
      return iAddr;

    } catch (SocketException e) {

      e.printStackTrace();
      Log.d(TAG, "getBroadcast" + e.getMessage());
    }
    return null;
  }

  /**
   * An empty file will be created automatically on the clone by Acceleration-Server. The presence
   * or absence of this file can let the method know if it is running on the phone or on the clone.
   * 
   * @return <b>True</b> if it is running on the clone<br>
   *         <b>False</b> if it is running on the phone.
   */
  public static boolean isOffloaded() {
    try {
      File tempFile = new File(Constants.FILE_OFFLOADED);
      return tempFile.exists();
    } catch (Exception e) {
      return true;
    }
  }

  /**
   * Execute a shell command on an Android device
   * 
   * @param TAG
   * @param cmd
   * @param asRoot
   * @return
   */
  public static int executeAndroidShellCommand(String TAG, String cmd, boolean asRoot) {
    Process p = null;
    DataOutputStream outs = null;
    int shellComandExitValue = 0;

    try {
      long startTime = System.currentTimeMillis();

      if (asRoot) {
        p = Runtime.getRuntime().exec("su");
        outs = new DataOutputStream(p.getOutputStream());
        outs.writeBytes(cmd + "\n");
        outs.writeBytes("exit\n");
        outs.close();
      } else {
        p = Runtime.getRuntime().exec(cmd);
        outs = new DataOutputStream(p.getOutputStream());
        outs.writeBytes("exit\n");
        outs.close();
      }

      shellComandExitValue = p.waitFor();
      Log.i(TAG, "Executed cmd: " + cmd + " in " + (System.currentTimeMillis() - startTime)
          + " ms (exitValue: " + shellComandExitValue + ")");

    } catch (IOException e) {
      e.printStackTrace();
    } catch (InterruptedException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } finally {
      // destroyProcess(p);
      try {
        if (outs != null)
          outs.close();
        // p.destroy();
      } catch (IOException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
      }
    }

    return shellComandExitValue;
  }

  /**
   * Write the ID of this clone on the file "/mnt/sdcard/rapid/cloneId".<br>
   * The IDs are assigned by the main clone during the PING and are consecutive. The main clone has
   * cloneHelperId equal to 0. The IDs can be used by the developer when parallelizing the
   * applications among multiple clones. He can use the IDs to split the data input and assign
   * portions to clones based on their ID.
   * 
   * @param cloneHelperId the ID of this clone assigned by the main clone.
   */
  public static void writeCloneHelperId(int cloneHelperId) {
    try {
      File cloneIdFile = new File(Constants.CLONE_ID_FILE);
      FileWriter cloneIdWriter = new FileWriter(cloneIdFile);
      cloneIdWriter.write(String.valueOf(cloneHelperId));
      cloneIdWriter.close();
    } catch (IOException e) {
      e.printStackTrace();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  /**
   * Read the file "/mnt/sdcard/rapid/cloneId" for the ID of this clone.
   * 
   * @return 0 if this is the phone or the main clone (the file may even not exist in these cases)
   *         <br>
   *         CLONE_ID otherwise
   */
  public static int readCloneHelperId() {
    Scanner cloneIdReader = null;
    int cloneId = 0;

    try {
      File cloneIdFile = new File(Constants.CLONE_ID_FILE);
      cloneIdReader = new Scanner(cloneIdFile);
      cloneId = cloneIdReader.nextInt();
    } catch (Exception e) {
      // Stay quiet, we know it.
    } finally {
      try {
        cloneIdReader.close();
      } catch (Exception e) {
        Log.e(TAG,
            "CloneId file is not here, this means that this is the main clone (or the phone)");
      }
    }

    return cloneId;
  }

  /**
   * Delete the file containing the cloneHelperId.
   */
  public static void deleteCloneHelperId() {
    File cloneIdFile = new File(Constants.CLONE_ID_FILE);
    cloneIdFile.delete();
  }

  private static final String SCRIPT_FILE = "temp_sokol.sh";

  public static int runScript(Context ctx, String script, StringBuilder res, boolean asroot) {
    final File file = new File(ctx.getDir("bin", 0), SCRIPT_FILE);
    final ScriptRunner runner = new ScriptRunner(file, script, res, asroot);
    runner.start();
    try {
      runner.join();
    } catch (InterruptedException ex) {
    }
    return runner.exitcode;
  }

  /**
   * Run a script in Android.
   *
   */
  public static final class ScriptRunner extends Thread {
    private final File file;
    private final String script;
    private final StringBuilder res;
    private final boolean asroot;
    public int exitcode = -1;
    private Process exec;
    private static final String TAG = "ScriptRunner";

    /**
     * Creates a new script runner.
     * 
     * @param file temporary script file
     * @param script script to run
     * @param res response output
     * @param asroot if true, executes the script as root
     */
    public ScriptRunner(File file, String script, StringBuilder res, boolean asroot) {
      this.file = file;
      this.script = script;
      this.res = res;
      this.asroot = asroot;
    }

    @Override
    public void run() {
      try {
        Log.d(TAG, "Running script: " + script);

        file.createNewFile();
        final String abspath = file.getAbsolutePath();
        // make sure we have execution permission on the script file
        Runtime.getRuntime().exec("chmod 777 " + abspath).waitFor();
        // Write the script to be executed
        final OutputStreamWriter out = new OutputStreamWriter(new FileOutputStream(file));
        if (new File("/system/bin/sh").exists()) {
          out.write("#!/system/bin/sh\n");
        }
        out.write(script);
        if (!script.endsWith("\n"))
          out.write("\n");
        out.write("exit\n");
        out.flush();
        out.close();
        if (this.asroot) {
          // Create the "su" request to run the script
          exec = Runtime.getRuntime().exec("su -c " + abspath);
        } else {
          // Create the "sh" request to run the script
          exec = Runtime.getRuntime().exec("sh " + abspath);
        }
        final InputStream stdout = exec.getInputStream();
        final InputStream stderr = exec.getErrorStream();
        final byte buf[] = new byte[8192];
        int read = 0;
        while (true) {
          final Process localexec = exec;
          if (localexec == null)
            break;
          try {
            // get the process exit code - will raise IllegalThreadStateException if still running
            this.exitcode = localexec.exitValue();
          } catch (IllegalThreadStateException ex) {
            // The process is still running
          }
          // Read stdout
          if (stdout.available() > 0) {
            read = stdout.read(buf);
            if (res != null)
              res.append(new String(buf, 0, read));
          }
          // Read stderr
          if (stderr.available() > 0) {
            read = stderr.read(buf);
            if (res != null)
              res.append(new String(buf, 0, read));
          }
          if (this.exitcode != -1) {
            // finished
            break;
          }
          // Sleep for the next round
          Thread.sleep(50);
        }
      } catch (InterruptedException ex) {
        Log.i(TAG, "InterruptedException ");
        ex.printStackTrace();
        if (res != null)
          res.append("\nOperation timed-out");
      } catch (Exception ex) {
        Log.i(TAG, "Exception");
        ex.printStackTrace();
        if (res != null)
          res.append("\n" + ex);
      } finally {
        destroy();
      }
    }

    /**
     * Destroy this script runner
     */
    public synchronized void destroy() {
      if (exec != null)
        exec.destroy();
      exec = null;
    }
  }

  public static long copy(InputStream from, OutputStream to) throws IOException {
    // checkNotNull(from);
    // checkNotNull(to);
    byte[] buf = createBuffer();
    long total = 0;
    while (true) {
      int r = from.read(buf);
      if (r == -1) {
        break;
      }
      to.write(buf, 0, r);
      total += r;
    }
    return total;
  }

  static byte[] createBuffer() {
    return new byte[8192];
  }

  public static byte[] toByteArray(InputStream in) throws IOException {
    // Presize the ByteArrayOutputStream since we know how large it will need
    // to be, unless that value is less than the default ByteArrayOutputStream
    // size (32).
    ByteArrayOutputStream out = new ByteArrayOutputStream(Math.max(32, in.available()));
    copy(in, out);
    return out.toByteArray();
  }

  public static String readAssetFileAsString(Context context, String fileName) throws IOException {
    InputStream is = context.getAssets().open(fileName);

    byte[] bytes = toByteArray(is);
    is.close();
    return new String(bytes, Charset.forName("UTF-8"));
  }


  /**
   * Write an object to the internal memory of this application. We will use this method to store
   * the features of the user so that other components of our application can read them. The object
   * will then be read using the method <code>readObjectFromInternalFile</code>. We synchronize the
   * writing and reading process so that we don't end up reading a partial object.
   * 
   * @param context The context of the application. Needed to access the internal file directory.
   * @param o The object to store.
   * @param fileName The name of the file where to store the object.
   * @throws IOException
   */
  public static void saveObjectToInternalFile(Context context, Object o, String fileName)
      throws IOException {
    File f = new File(context.getFilesDir(), fileName);

    synchronized (syncFeatureObject) {
      if (f.exists()) {
        if (f.delete()) {
          Log.i(TAG, "File " + f.getName() + " deleted");
        } else {
          Log.i(TAG, "Could not delete file " + f.getName() + " (maybe it doesn't exist)");
        }
      }

      if (f.createNewFile()) {
        FileOutputStream fos = new FileOutputStream(f);
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(o);
        oos.close();
      }
    }
  }

  /**
   *
   * @param context The context of the application. Needed to access the internal file directory.
   * @param fileName The name of the file where the object was saved before.
   * @return The object that was read from the file.
   * @throws IOException
   * @throws ClassNotFoundException
   */
  public static Object readObjectFromInternalFile(Context context, String fileName)
      throws IOException, ClassNotFoundException {
    File f = new File(context.getFilesDir(), fileName);

    synchronized (syncFeatureObject) {
      if (!f.exists()) {
        throw new FileNotFoundException("File " + fileName + " doesn't exist");
      }

      ObjectInputStream ois = new ObjectInputStream(new FileInputStream(f));
      Object o = ois.readObject();
      ois.close();
      return o;
    }
  }

  public static BufferedWriter createMeasurementFile(String filePath, String header) {
    return createMeasurementFile(filePath, header, false);
  }

  /**
   * Creates a file where we can write the measurements.
   * 
   * @param filePath
   * @param header
   * @return A BuuferedWriter that we should not forget to close when we're done.
   */
  public static BufferedWriter createMeasurementFile(String filePath, String header,
      boolean appending) {
    File dseTestFile = new File(filePath);
    BufferedWriter dseTestFileBuf = null;
    try {
      dseTestFileBuf = new BufferedWriter(new FileWriter(dseTestFile, appending));
      if (!appending) {
        dseTestFileBuf.write(header);
      }
    } catch (IOException e1) {
      Log.w(TAG, "Could not create file " + filePath + ": " + e1);
    }

    return dseTestFileBuf;
  }
}
