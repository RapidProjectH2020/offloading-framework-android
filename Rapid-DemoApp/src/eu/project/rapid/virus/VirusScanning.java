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
package eu.project.rapid.virus;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Method;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

import android.content.Context;
import android.util.Log;
import eu.project.rapid.ac.DFE;
import eu.project.rapid.ac.Remote;
import eu.project.rapid.ac.Remoteable;
import eu.project.rapid.ac.utils.Constants;
import eu.project.rapid.ac.utils.Utils;
import eu.project.rapid.ac.utils.ZipHandler;

/**
 * The virus scanning application.
 */
public class VirusScanning extends Remoteable {

  private static final long serialVersionUID = -1839651210541446342L;

  private static final String TAG = "VirusScanning";

  private transient String[] signatureDB;
  private byte[] zippedFolder;
  private static final int SIGNATURE_SIZE = 1024;
  private int lastFileOnPhone = -1;
  private double localFraction;

  protected transient DFE dfe;
  protected int nrClones = 1;
  private char[] fileBuffer;
  private char[] fileHeaderBuffer = new char[SIGNATURE_SIZE];

  public VirusScanning(Context ctx, DFE dfe, int nrClones) {
    this.dfe = dfe;
    this.nrClones = nrClones;
  }

  /**
   * Get the fraction of data that should be processed by the phone and split the input in two
   * parts: (S_P0, S_P1) such that S_P0 + S_P1 = S_app.<br>
   * This should be implemented by the developer. In the virus scanning case we split the files in
   * two groups.
   *
   * @param localFraction The fraction of data input that will be processed locally on the phone.
   */
  public void prepareData(double localFraction) {

    this.localFraction = localFraction;

    Log.i(TAG, "Started folder compression");

    File zippedFile = null;
    File folderToScan = new File(Constants.VIRUS_FOLDER_TO_SCAN);
    String[] fileNames = folderToScan.list();

    // The first files should be scanned on phone.
    lastFileOnPhone = (int) (localFraction * fileNames.length);
    Log.i(TAG, "localFraction: " + localFraction + "\nfl: " + fileNames.length
        + "\nlastFileOnPhone: " + lastFileOnPhone);

    if (localFraction == 1) {
      Log.w(TAG, "No need to prepare the data. All execution is going to be local.");
      return;
    } else {
      long s = System.currentTimeMillis();
      try {
        ZipHandler.zipFolder(Constants.VIRUS_FOLDER_TO_SCAN, Constants.VIRUS_FOLDER_ZIP,
            lastFileOnPhone, fileNames.length);
      } catch (Exception e1) {
        // TODO Auto-generated catch block
        e1.printStackTrace();
      }
      Log.d(TAG, "Time to zip the files: " + (System.currentTimeMillis() - s) + " ms");

      // Read the zip folder to make it ready for remote processing
      zippedFile = new File(Constants.VIRUS_FOLDER_ZIP);
    }

    FileInputStream in = null;
    try {
      in = new FileInputStream(zippedFile);
      int length = (int) zippedFile.length();
      int read = 0;
      int totalRead = 0;
      zippedFolder = new byte[length];
      while (read != -1 && totalRead < length) {
        read = in.read(zippedFolder, totalRead, length - totalRead);
        totalRead += read;
        Log.i(TAG, "Read " + totalRead);
      }
    } catch (IOException e) {
      Log.e(TAG, "Cannot find the zip file to send on the remote side" + e);
    } catch (Exception e) {
      Log.e(TAG, "Cannot find the zip file to send on the remote side" + e);
    } finally {
      try {
        in.close();
      } catch (Exception e) {
        Log.e(TAG, "Could not close the FileStream of the zipped folder");
      }
    }

    Log.i(TAG, "Finished folder compression");
  }

  /**
   * The method that starts the scanning process using the DFE.
   *
   * @return Number of viruses found.
   */
  public int scanFolder() {
    int nrVirusesFound = 0;

    Method toExecute;
    Class<?>[] paramTypes = null;
    Object[] paramValues = null;

    try {
      toExecute = this.getClass().getDeclaredMethod("localScanFolder", paramTypes);
      nrVirusesFound = (Integer) dfe.execute(toExecute, paramValues, this);
    } catch (SecurityException e) {
      // Should never get here
      e.printStackTrace();
      throw e;
    } catch (NoSuchMethodException e) {
      // Should never get here
      e.printStackTrace();
    } catch (Throwable e) {
      e.printStackTrace();
    }

    return nrVirusesFound;
  }

  @Remote
  public int localScanFolder() {

    Log.i(TAG, "Scan folder");

    Log.i(TAG, "Started signature initialization from folder: " + Constants.VIRUS_DB_PATH);
    initSignatureDB(Constants.VIRUS_DB_PATH);
    Log.i(TAG, "Finished signature initialization. Number of signatures: " + signatureDB.length);

    boolean isOffloaded = Utils.isOffloaded();

    Log.i(TAG, "Scanning folder: " + Constants.VIRUS_FOLDER_TO_SCAN);
    File folderToScan;
    File[] filesToScan;

    int start = 0;
    int end = 0;

    if (isOffloaded) {
      Log.i(TAG, "isOffloaded true");

      int cloneHelperId = Utils.readCloneHelperId();

      // We know we are on the remote side and we have a zipped folder
      // containing the files to be scanned.
      extractZippedFiles();
      folderToScan = new File(Constants.VIRUS_FOLDER_TO_SCAN);
      filesToScan = folderToScan.listFiles();

      Log.i(TAG, "Number of files in folder: " + filesToScan.length);

      // Integer division, some files may be not considered
      int howManyFiles = (int) ((filesToScan.length) / nrClones);
      start = cloneHelperId * howManyFiles; // cloneHelperId starts from 0
                                            // (the main clone)
      end = start + howManyFiles;

      // If this is the clone with the highest id let him take care
      // of the files not considered due to the integer division.
      if (cloneHelperId == nrClones - 1) {
        end += filesToScan.length % nrClones;
      }

    } else {
      Log.i(TAG, "isOffloaded false");

      folderToScan = new File(Constants.VIRUS_FOLDER_TO_SCAN);
      filesToScan = folderToScan.listFiles();

      // The first files are scanned on the phone the last ones remotely
      // (in case of data partition)
      start = 0;
      end = lastFileOnPhone;
    }

    int nrVirusesFound = 0;
    Log.i(TAG, "Nr files to scan: " + (end - start));
    Log.i(TAG, "Checking files: " + start + "-" + (end - 1));
    for (int i = start; i < end; i++) {
      // Log.i(TAG, "Checking file: " + filesToScan[i]);
      if (checkIfFileVirus(filesToScan[i])) {
        // Log.i(TAG, "Virus found");
        nrVirusesFound++;
      }
    }

    Log.i(TAG, "Number of viruses found: " + nrVirusesFound);
    return nrVirusesFound;
  }

  /**
   * When having more than one clone running the method, we will obtain partial results that should
   * be combined to get the final result. This will be done automatically by the main clone when
   * calling this method.
   *
   * @param params Array of partial results.
   * @return The total result.
   */
  public int localScanFolderReduce(int[] params) {
    int nrViruses = 0;
    for (int i = 0; i < params.length; i++) {
      nrViruses += params[i];
    }
    return nrViruses;
  }

  private void extractZippedFiles() {

    try {
      File f = new File(Constants.VIRUS_FOLDER_ZIP);

      FileOutputStream out = new FileOutputStream(f);
      out.write(zippedFolder);
      out.close();

      // Delete the old directory containing the files we have scanned in
      // previous runs
      File temp = new File(Constants.VIRUS_FOLDER_TO_SCAN);
      if (!temp.exists()) {
        temp.mkdir();
      } else {
        Utils.executeAndroidShellCommand(TAG,
            "rm " + Constants.VIRUS_FOLDER_TO_SCAN + File.separator + "*", false);
      }

      long s = System.currentTimeMillis();
      ZipHandler.extractFolder(Constants.VIRUS_FOLDER_ZIP);
      Log.i(TAG, "Time to extract the files: " + (System.currentTimeMillis() - s) + " ms");

    } catch (FileNotFoundException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  /**
   * Check all the subsequences of length SIGNATURE_SIZE for the virus signature.
   *
   * @param fileToScan
   * @return
   */
  private boolean checkIfFileVirus(File fileToScan) {
    MessageDigest md;

    try {
      md = MessageDigest.getInstance("SHA-1");

      int length = (int) fileToScan.length();

      if (fileBuffer == null || fileBuffer.length != length)
        fileBuffer = new char[length];
      // Log.i(TAG, "Checking file " + fileToScan.getName());
      // Log.i(TAG, "Length of file " + length);

      FileReader currentFile = new FileReader(fileToScan);
      int totalRead = 0;
      int read = 0;
      do {
        totalRead += read;
        read = currentFile.read(fileBuffer, totalRead, length - totalRead);
      } while (read > 0);
      currentFile.close();

      if (totalRead > 0) {
        for (int i = 0; i < 100; i++) {
          System.arraycopy(fileBuffer, i, fileHeaderBuffer, 0, SIGNATURE_SIZE);
          String signature = new String(fileHeaderBuffer);
          signature = Utils.bytesToHex(md.digest(signature.getBytes()));
          if (isInVirusDB(signature)) {
            return true;
          }
        }
      }
    } catch (NoSuchAlgorithmException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    } catch (Exception e) {
      e.printStackTrace();
    }
    return false;
  }

  private boolean isInVirusDB(String signature) {
    for (int i = 0; i < signatureDB.length; i++) {
      if (signature.equals(signatureDB[i]))
        return true;
    }
    return false;
  }

  private void initSignatureDB(String pathToSignatures) {
    try {
      MessageDigest md = MessageDigest.getInstance("SHA-1");
      File signatureFolder = new File(pathToSignatures);
      File[] demoViruses = signatureFolder.listFiles();

      signatureDB = new String[demoViruses.length];
      char[] buffer = new char[SIGNATURE_SIZE];

      int i = 0;
      for (File virus : demoViruses) {

        FileReader signatureFile = new FileReader(virus);
        int totalRead = 0;
        int read = 0;
        while (totalRead != SIGNATURE_SIZE) {
          read = signatureFile.read(buffer, totalRead, buffer.length - totalRead);
          totalRead += read;
        }
        if (totalRead > 0)
          signatureDB[i++] = Utils.bytesToHex(md.digest(new String(buffer).getBytes()));
        signatureFile.close();
      }

    } catch (FileNotFoundException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    } catch (NoSuchAlgorithmException e) {
      Log.e(TAG, "NoSuchAlgorithmException " + e.getMessage());
    } catch (Exception e) {
      e.printStackTrace();
      Log.e(TAG, "Exception " + e.getMessage());
    }
  }

  @Override
  public void copyState(Remoteable state) {

  }

  public void setNumberOfClones(int nrClones) {
    this.nrClones = nrClones;
  }
}

