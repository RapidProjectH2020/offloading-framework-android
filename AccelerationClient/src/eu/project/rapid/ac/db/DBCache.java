package eu.project.rapid.ac.db;

import java.io.IOException;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

import android.util.Log;
import eu.project.rapid.ac.utils.Constants;
import eu.project.rapid.ac.utils.Utils;

/**
 * While RAPID is running there is no need to access the database. We use this class to keep the
 * needed information so that the offloading decision is faster. When the app is closed the entries
 * of this cached DB are saved.
 * 
 * @author sokol
 *
 */
public class DBCache {

  private static final String TAG = "DBCache";

  private static int nrElements;
  private static DBCache dbCache;
  private static Map<String, Deque<DBEntry>> dbMap; // appName is the key

  @SuppressWarnings("unchecked")
  private DBCache() {
    try {
      Log.i(TAG, "Reading the dbCache from file: " + Constants.FILE_DB_CACHE);
      dbMap = (Map<String, Deque<DBEntry>>) Utils.readObjectFromFile(Constants.FILE_DB_CACHE);
    } catch (ClassNotFoundException | IOException e) {
      Log.w(TAG, "Could not read the dbCache from file: " + e);
    }

    if (dbMap == null) {
      dbMap = new HashMap<>();
    }
  };

  public static DBCache getDbCache() {
    if (dbCache == null) {
      Log.i(TAG, "Creating the dbCache object");
      dbCache = new DBCache();
    }

    return dbCache;
  }

  public void insertEntry(DBEntry entry) {
    String key = entry.getMethodName();
    if (!dbMap.containsKey(key)) {
      dbMap.put(key, new LinkedList<DBEntry>());
    }

    while (dbMap.get(key).size() >= Constants.MAX_METHOD_EXEC_HISTORY) {
      dbMap.get(key).removeLast();
      nrElements--;
    }

    dbMap.get(key).addFirst(entry);
    nrElements++;
  }

  /**
   * To be used to retrieve all entries.
   * 
   * @param methodName
   * @return
   */
  public Deque<DBEntry> getAllEntriesFilteredOn(String methodName) {
    if (dbMap.containsKey(methodName)) {
      return dbMap.get(methodName);
    } else {
      return new LinkedList<>();
    }
  }

  /**
   * To be used for retrieving the entries of LOCAL execution. The elements are sorted in
   * incremental order based on timestamp.
   * 
   * @param methodName
   * @param appName
   * @param execLocation
   * @return
   */
  public Deque<DBEntry> getAllEntriesFilteredOn(String appName, String methodName,
      String execLocation) {

    assert dbMap != null;

    Deque<DBEntry> tempList = new LinkedList<>();
    if (dbMap.containsKey(methodName)) {
      for (DBEntry e : dbMap.get(methodName)) {
        // Log.i(TAG, "Checking entry: " + e.getAppName() + ", " + e.getMethodName() + ", "
        // + e.getExecLocation() + ", " + e.getTimestamp());
        if (e.getAppName().equals(appName) && e.getExecLocation().equals(execLocation)) {
          tempList.addLast(e);
        }
      }
    }

    return tempList;
  }

  /**
   * To be used for retrieving the entries of REMOTE execution. The elements are sorted in
   * incremental order based on timestamp.
   * 
   * @param methodName
   * @param appName
   * @param execLocation
   * @param networkType
   * @param networkSubtype
   * @return
   */
  public Deque<DBEntry> getAllEntriesFilteredOn(String appName, String methodName,
      String execLocation, String networkType, String networkSubtype) {

    assert dbMap != null;

    Deque<DBEntry> tempList = new LinkedList<>();
    if (dbMap.containsKey(methodName)) {
      for (DBEntry e : dbMap.get(methodName)) {
        if (e.getAppName().equals(appName) && e.getExecLocation().equals(execLocation)
            && e.getNetworkType().equals(networkType)
            && e.getNetworkSubType().equals(networkSubtype)) {
          tempList.addLast(e);
        }
      }
    }

    return tempList;
  }

  public void clearDbCache() {
    dbMap.clear();
    nrElements = 0;
  }

  /**
   * To be called by the DFE when the (Rapid) application is closed.
   */
  public static void saveDbCache() {
    try {
      Utils.writeObjectToFile(Constants.FILE_DB_CACHE, dbMap);
    } catch (IOException e) {
      Log.e(TAG, "Could not save the dbCache on the file: " + e);
    }
  }

  /**
   * Returns the number of entries in the DB cache, i.e. the number of different methods insterted
   * in the DB.
   * 
   * @return
   */
  public int size() {
    if (dbMap == null) {
      return 0;
    }

    return dbMap.size();
  }

  /**
   * Returns the number of total elements in the DB cache, i.e.: nrMethods * measurementsPerMethod.
   * 
   * @return
   */
  public int nrElements() {
    return nrElements;
  }

  public int getByteSize() {
    int bytes = -1;
    try {
      bytes = Utils.objectToByteArray(dbMap).length;
    } catch (IOException e) {
      Log.e(TAG, "Error while converting the DB cache map to byte array: " + e);
    }

    return bytes;
  }
}
