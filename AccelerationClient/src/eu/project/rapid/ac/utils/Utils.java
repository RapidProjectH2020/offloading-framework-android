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

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import android.content.Context;
import android.util.Log;

public class Utils {

  private static final String TAG = "RapidUtils";
  private final static char[] hexArray = "0123456789ABCDEF".toCharArray();
  private final static Map<Character, Integer> hexToDec = new HashMap<Character, Integer>();

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
    return new String(hexChars);
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
}
