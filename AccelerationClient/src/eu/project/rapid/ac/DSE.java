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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import android.content.Context;
import android.database.Cursor;
import android.util.Log;
import eu.project.rapid.ac.db.DatabaseQuery;
import eu.project.rapid.ac.profilers.NetworkProfiler;
import eu.project.rapid.ac.utils.Constants;

/**
 * DSE decides whether to execute the requested method locally or remotely.
 */
class DSE {

  private static final String TAG = "DSE";
  private static final boolean VERBOSE_LOG = false;

  // To be used in case of no previous remote execution.
  // If the ulRate and dlRate are bigger than these values then we offload
  private static final int MIN_UL_RATE_OFFLOAD_1_TIME = 256 * 1000; // b/s
  private static final int MIN_DL_RATE_OFFLOAD_1_TIME = 256 * 1000; // b/s

  private Context mContext;
  private int userChoice;

  private int programOrientedDecNr = 1;

  DSE(Context context, int userChoice) {
    this.mContext = context;
    this.userChoice = userChoice;
  }

  /**
   * Decide whether to execute remotely, locally, or hybrid
   * 
   * @return The type of execution: LOCAL, REMOTE, HYBRID
   */
  public int findExecLocation(String appName, String methodName) {

    if (!DFE.onLine || userChoice == Constants.LOCATION_LOCAL) {
      return Constants.LOCATION_LOCAL;
    } else if (userChoice == Constants.LOCATION_REMOTE) {
      return Constants.LOCATION_REMOTE;
    } else if (userChoice == Constants.LOCATION_HYBRID) {
      return Constants.LOCATION_HYBRID;
    } else { // if (userChoice == RapidConstants.LOCATION_DYNAMIC) {

      // Measure how much it takes to decide where to run the method when using program-oriented
      // offloading
      String preOffloadPath = Constants.RAPID_FOLDER + "/test_logs/program_oriented_costs.csv";
      File preOffloadLogFile = new File(preOffloadPath);
      BufferedWriter preOffloadBuffLogFile = null;
      try {
        boolean createdNewFile = preOffloadLogFile.createNewFile();
        preOffloadBuffLogFile = new BufferedWriter(new FileWriter(preOffloadLogFile, true));
        if (createdNewFile) {
          preOffloadBuffLogFile.write("#DEC_NR(1/2/3),Total_Time(ms),Timestamp\n");
        } else {
          Log.e(TAG, "Could not create preOffloadFile " + preOffloadPath);
        }
      } catch (IOException e1) {
        Log.w(TAG, "Could not create preOffloadFile " + preOffloadPath + ": " + e1);
      }

      // Use the iterations if we want to measure how much it takes to decide
      int ulRate = NetworkProfiler.lastUlRate.getBw();
      int dlRate = NetworkProfiler.lastDlRate.getBw();
      int nrIterations = 1;
      boolean shouldOffload = false;
      double decisionTime = 0;
      for (int i = 0; i < nrIterations; i++) {
        long startDecisionTime = System.currentTimeMillis();
        shouldOffload = shouldOffload(appName, methodName, ulRate, dlRate);
        decisionTime += (System.currentTimeMillis() - startDecisionTime);
      }
      decisionTime /= nrIterations;

      if (preOffloadBuffLogFile != null) {
        try {
          preOffloadBuffLogFile.append(
              programOrientedDecNr + "," + decisionTime + "," + System.currentTimeMillis() + "\n");
        } catch (IOException e) {
          Log.w(TAG, "Could not write in preLogFile: " + e);
        } finally {
          try {
            preOffloadBuffLogFile.close();
          } catch (IOException e) {
            Log.w(TAG, "Error closing preLogFile: " + e);
          }
        }
      }

      if (shouldOffload) {
        Log.d(TAG, "Execute Remotely - True");
        return Constants.LOCATION_REMOTE;
      } else {
        Log.d(TAG, "Execute Remotely - False");
        return Constants.LOCATION_LOCAL;
      }
    }
  }

  /**
   * 
   * @param appName
   * @param methodName
   * @param currUlRate
   * @param currDlRate
   * @return <b>True</b> if the method should be executed remotely<br>
   *         <b>False</b> otherwise.
   */
  private boolean shouldOffload(String appName, String methodName, int currUlRate, int currDlRate) {
    DatabaseQuery query = new DatabaseQuery(mContext);

    Log.i(TAG, "Trying to decide where to execute the method: appName=" + appName + ", methodName="
        + methodName + ", currUlRate=" + currUlRate + ", currDlRate=" + currDlRate);

    // Variables needed for the local executions
    int nrLocalExec = 0;
    long meanDurLocal = 0, meanEnergyLocal = 0;

    // Variables needed for the remote executions
    int nrRemoteExec = 0;
    long meanDurRemote1 = 0, meanEnergyRemote1 = 0;
    long meanDurRemote2 = 0, meanEnergyRemote2 = 0;
    long meanDurRemote = 0, meanEnergyRemote = 0;

    /**
     * Check if the method has been executed LOCALLY in previous runs
     */
    String localSelection = DatabaseQuery.KEY_APP_NAME + " = ? AND " + DatabaseQuery.KEY_METHOD_NAME
        + " = ? AND " + DatabaseQuery.KEY_EXEC_LOCATION + " = ?";

    String[] localSelectionArgs = new String[] {appName, methodName, "LOCAL"};

    Cursor localResults = query.getAllEntries(new String[] {DatabaseQuery.KEY_EXEC_DURATION,
        DatabaseQuery.KEY_EXEC_ENERGY, DatabaseQuery.KEY_TIMESTAMP}, localSelection,
        localSelectionArgs);
    nrLocalExec = localResults.getCount();

    /**
     * Check if the method has been executed REMOTELY in previous runs
     */
    String remoteSelection =
        DatabaseQuery.KEY_APP_NAME + " = ? AND " + DatabaseQuery.KEY_METHOD_NAME + " = ? AND "
            + DatabaseQuery.KEY_EXEC_LOCATION + " = ? AND " + DatabaseQuery.KEY_NETWORK_TYPE
            + " = ? AND " + DatabaseQuery.KEY_NETWORK_SUBTYPE + " = ?";

    String[] remoteSelectionArgs = new String[] {appName, methodName, "REMOTE",
        NetworkProfiler.currentNetworkTypeName, NetworkProfiler.currentNetworkSubtypeName};

    Cursor remoteResults = query.getAllEntries(
        new String[] {DatabaseQuery.KEY_EXEC_DURATION, DatabaseQuery.KEY_EXEC_ENERGY,
            DatabaseQuery.KEY_UL_RATE, DatabaseQuery.KEY_DL_RATE, DatabaseQuery.KEY_TIMESTAMP},
        remoteSelection, remoteSelectionArgs);
    nrRemoteExec = remoteResults.getCount();

    // DECISION 1
    // If the number of previous remote executions is zero and the current connection is good
    // then offload the method to see how it goes.
    if (nrRemoteExec == 0) {

      programOrientedDecNr = 1;

      if (currUlRate > MIN_UL_RATE_OFFLOAD_1_TIME && currDlRate > MIN_DL_RATE_OFFLOAD_1_TIME) {
        Log.i(TAG, "Decision 1: No previous remote executions. Good connectivity.");
        return true;
      } else {
        Log.i(TAG, "Decision 1: No previous remote executions. Bad connectivity.");
        return false;
      }
    }

    // Local part
    // Calculate the meanDurLocal and meanEnergyLocal from the previous runs.
    // Give more weight to recent measurements.
    // Assume the DB query returns the rows ordered by ID (check this).
    if (VERBOSE_LOG)
      Log.i(TAG, "------------ The local executions of the method:");
    long localDuration, localEnergy;
    long[] localTimestamps = new long[nrLocalExec];
    int i = 0;
    while (localResults.moveToNext()) {
      localDuration = Long.parseLong(
          localResults.getString(localResults.getColumnIndex(DatabaseQuery.KEY_EXEC_DURATION)));
      localEnergy = Long.parseLong(
          localResults.getString(localResults.getColumnIndex(DatabaseQuery.KEY_EXEC_ENERGY)));
      localTimestamps[i] = Long.parseLong(
          localResults.getString(localResults.getColumnIndex(DatabaseQuery.KEY_TIMESTAMP)));

      meanDurLocal += localDuration;
      meanEnergyLocal += localEnergy;

      i++;
      if (i > 1) {
        meanDurLocal /= 2;
        meanEnergyLocal /= 2;
      }

      if (VERBOSE_LOG) {
        Log.i(TAG, "duration: " + localDuration + " energy: " + localEnergy + " timestamp: "
            + localResults.getString(localResults.getColumnIndex(DatabaseQuery.KEY_TIMESTAMP)));
      }
    }
    Log.i(TAG, "nrLocalExec: " + nrLocalExec);
    Log.i(TAG, "meanDurLocal: " + meanDurLocal + "ns (" + meanDurLocal / 1000000000.0
        + "s) meanEnergyLocal: " + meanEnergyLocal);


    // Remote part
    long[] remoteDurations = new long[nrRemoteExec];
    long[] remoteEnergies = new long[nrRemoteExec];
    int[] remoteUlRates = new int[nrRemoteExec];
    int[] remoteDlRates = new int[nrRemoteExec];
    long[] remoteTimestamps = new long[nrRemoteExec];
    i = 0;
    if (VERBOSE_LOG)
      Log.i(TAG, "------------ The remote executions of the method:");
    while (remoteResults.moveToNext()) {
      remoteDurations[i] = Long.parseLong(
          remoteResults.getString(remoteResults.getColumnIndex(DatabaseQuery.KEY_EXEC_DURATION)));
      remoteEnergies[i] = Long.parseLong(
          remoteResults.getString(remoteResults.getColumnIndex(DatabaseQuery.KEY_EXEC_ENERGY)));
      remoteUlRates[i] = Integer.parseInt(
          remoteResults.getString(remoteResults.getColumnIndex(DatabaseQuery.KEY_UL_RATE)));
      remoteDlRates[i] = Integer.parseInt(
          remoteResults.getString(remoteResults.getColumnIndex(DatabaseQuery.KEY_DL_RATE)));
      remoteTimestamps[i] = Long.parseLong(
          remoteResults.getString(remoteResults.getColumnIndex(DatabaseQuery.KEY_TIMESTAMP)));

      if (VERBOSE_LOG) {
        Log.i(TAG, "duration: " + remoteDurations[i] + " energy: " + remoteEnergies[i] + " ulRate: "
            + remoteUlRates[i] + " dlRate: " + remoteDlRates[i] + " timestamp: "
            + remoteResults.getString(remoteResults.getColumnIndex(DatabaseQuery.KEY_TIMESTAMP)));
      }

      i++;
    }
    Log.i(TAG, "nrRemoteExec: " + nrRemoteExec);

    // Close the database
    try {
      query.destroy();
    } catch (Throwable e) {
      e.printStackTrace();
    }

    // DECISION 2
    int NR_TIMES_SWITCH_SIDES = 10;
    // Last local timestamp
    long lastLocalTimestamp = (nrLocalExec == 0) ? 0 : localTimestamps[localTimestamps.length - 1];
    // Pass all the remote timestamps that are smaller than the last local timestamp
    for (i = 0; i < nrRemoteExec && remoteTimestamps[i] < lastLocalTimestamp; i++);
    if ((nrRemoteExec - i) >= NR_TIMES_SWITCH_SIDES) {
      Log.i(TAG, "Decision 2: Too many remote executions in a row.");
      programOrientedDecNr = 2;
      return false;
    }

    // Do the same thing for the remote side
    // Last remote timestamp
    long lastRemoteTimestamp =
        (nrRemoteExec == 0) ? 0 : remoteTimestamps[remoteTimestamps.length - 1];
    // Pass all the local timestamps that are smaller than the last remote timestamp
    for (i = 0; i < nrLocalExec && localTimestamps[i] < lastRemoteTimestamp; i++);
    if ((nrLocalExec - i) >= NR_TIMES_SWITCH_SIDES) {
      Log.i(TAG, "Decision 2: Too many local executions in a row.");
      programOrientedDecNr = 2;
      return true;
    }

    // DECISION 3
    // Calculate two different mean values for the offloaded execution:
    // 1. The first are the same as for the local execution, gives more weight to recent runs
    // 2. The second are calculated as the average of the three closest values to the currentUlRate
    // and currDlRate
    int minDistIndex1 = 0, minDistIndex2 = 0, minDistIndex3 = 0;
    double minDist1 = Double.POSITIVE_INFINITY, minDist2 = Double.POSITIVE_INFINITY,
        minDist3 = Double.POSITIVE_INFINITY;
    for (i = 0; i < nrRemoteExec; i++) {
      // Calculate the first meanDuration and meanEnergy
      // The first formula is the same as for the local executions,
      // gives more importance to the last measurements.
      meanDurRemote1 += remoteDurations[i];
      meanEnergyRemote1 += remoteEnergies[i];
      if (i > 0) {
        meanDurRemote1 /= 2;
        meanEnergyRemote1 /= 2;
      }

      // Keep the indexes of the three measurements that have
      // the smallest distance dist(ulRate, dlRate, currUlRate, currDlRate)
      // minDist1 < minDist2 < minDist3
      double newDist = dist(remoteUlRates[i], remoteDlRates[i], currUlRate, currDlRate);
      if (newDist < minDist1) {
        minDist3 = minDist2;
        minDistIndex3 = minDistIndex2;

        minDist2 = minDist1;
        minDistIndex2 = minDistIndex1;

        minDist1 = newDist;
        minDistIndex1 = i;
      } else if (newDist < minDist2) {
        minDist3 = minDist2;
        minDistIndex3 = minDistIndex2;

        minDist2 = newDist;
        minDistIndex2 = i;
      } else if (newDist < minDist3) {
        minDist3 = newDist;
        minDistIndex3 = i;
      }
    }

    // Give more weight to the closest point
    meanDurRemote2 = (((remoteDurations[minDistIndex3] + remoteDurations[minDistIndex2]) / 2)
        + remoteDurations[minDistIndex1]) / 2;
    meanEnergyRemote2 = (((remoteEnergies[minDistIndex3] + remoteEnergies[minDistIndex2]) / 2)
        + remoteEnergies[minDistIndex1]) / 2;

    meanDurRemote = (meanDurRemote1 + meanDurRemote2) / 2;
    meanEnergyRemote = (meanEnergyRemote1 + meanEnergyRemote2) / 2;

    Log.d(TAG, "meanDurRemote1: " + meanDurRemote1 + "  meanDurRemote2: " + meanDurRemote2);
    Log.d(TAG, "meanDurRemote: " + meanDurRemote + "ns (" + meanDurRemote / 1000000000.0 + "s)");

    Log.d(TAG, "meanEnergyRemote1: " + meanEnergyRemote1 + "  meanEnergyRemote2: "
        + meanEnergyRemote2 + "  meanEnergyRemote: " + meanEnergyRemote);

    Log.i(TAG, "Decision 3.");
    programOrientedDecNr = 3;
    if (userChoice == Constants.LOCATION_DYNAMIC_TIME) {
      Log.d(TAG, "Making a choice for fast execution");
      return meanDurRemote <= meanDurLocal;
    } else if (userChoice == Constants.LOCATION_DYNAMIC_ENERGY) {
      Log.d(TAG, "Making a choice for low energy");
      return meanEnergyRemote <= meanEnergyLocal;
    } else {
      Log.d(TAG, "Making a choice for low energy and fast execution");
      return (meanDurRemote <= meanDurLocal) && (meanEnergyRemote <= meanEnergyLocal);
    }
  }

  private double dist(int ul1, int dl1, int ul2, int dl2) {
    return Math.sqrt((ul2 - ul1) * (ul2 - ul1) + (dl2 - dl1) * (dl2 - dl1));
  }

  public void setUserChoice(int userChoice) {
    this.userChoice = userChoice;
  }

  public int getUserChoice() {
    return this.userChoice;
  }

  @SuppressWarnings("unused")
  private void sleep(int millis) {
    try {
      Thread.sleep(millis);
    } catch (InterruptedException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }
}
