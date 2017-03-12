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
package eu.project.rapid.demo;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Random;

import android.app.Activity;
import android.app.ProgressDialog;
import android.graphics.Color;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemSelectedListener;
import android.widget.LinearLayout;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import eu.project.rapid.ac.DFE;
import eu.project.rapid.ac.utils.Constants;
import eu.project.rapid.common.Clone;
import eu.project.rapid.common.RapidConstants.COMM_TYPE;
import eu.project.rapid.queens.NQueens;
import eu.project.rapid.sudoku.Sudoku;
import eu.project.rapid.synthBenchmark.JniTest;
import eu.project.rapid.synthBenchmark.TestRemoteable;
import eu.project.rapid.virus.VirusScanning;

/**
 * The class that handles configuration parameters and starts the offloading process.
 */
public class StartExecution extends Activity implements DFE.DfeCallback {

  private static final String TAG = "StartExecution";

  public static int nrClones = 1;
  private LinearLayout layoutNrClones;
  private TextView textViewVmConnected;
  private RadioGroup executionRadioGroup;
  private Handler handler;

  private String vmIp;
  private DFE dfe;
  private boolean useRapidInfrastructure;

  /** Called when the activity is first created. */
  @Override
  public void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.main);
    Log.i(TAG, "onCreate");

    vmIp = getIntent().getStringExtra(MainActivity.KEY_VM_IP);
    useRapidInfrastructure =
        getIntent().getBooleanExtra(MainActivity.KEY_USE_RAPID_INFRASTRUCTURE, false);

    handler = new Handler();

    layoutNrClones = (LinearLayout) findViewById(R.id.layoutNrClones);
    if (useRapidInfrastructure) {
      layoutNrClones.setVisibility(View.VISIBLE);
      Spinner nrClonesSpinner = (Spinner) findViewById(R.id.spinnerNrClones);
      nrClonesSpinner.setOnItemSelectedListener(new NrClonesSelectedListener());
    } else {
      layoutNrClones.setVisibility(View.GONE);
    }

    executionRadioGroup = (RadioGroup) findViewById(R.id.executionRadioGroup);
    textViewVmConnected = (TextView) findViewById(R.id.textVmConnectionStatus);

    // If we don't specify the IP of the VM, we assume that we are using the Rapid infrastructure,
    // i.e. the DS, the VMM, the SLAM, etc., which means that the DFE will select automatically a
    // VM. We leave the user select a VM manually for fast deploy and testing.
    if (vmIp == null) {
      dfe = new DFE(getPackageName(), getPackageManager(), this);
    } else {
      dfe = new DFE(getPackageName(), getPackageManager(), this, new Clone("vb-clone-0", vmIp));
    }
  }

  private class VmConnectionStatusUpdater implements Runnable {

    public void run() {
      handler.post(new Runnable() {
        public void run() {
          textViewVmConnected.setTextColor(Color.GREEN);
          if (DFE.onLineClear) {
            textViewVmConnected.setText(R.string.textVmConnectedClear);
          } else if (DFE.onLineSSL) {
            textViewVmConnected.setText(R.string.textVmConnectedSSL);
          } else {
            textViewVmConnected.setTextColor(Color.RED);
            textViewVmConnected.setText(R.string.textVmDisconnected);
          }

          for (int i = 0; i < executionRadioGroup.getChildCount(); i++) {
            executionRadioGroup.getChildAt(i).setEnabled(true);
          }
        }
      });
    }
  }

  @Override
  public void onDestroy() {
    super.onDestroy();
    Log.i(TAG, "onDestroy");

    if (dfe != null) {
      dfe.onDestroy();
    }
  }

  @Override
  public void onPause() {
    super.onPause();
    Log.d(TAG, "OnPause");
  }

  public void onClickLoader1(View v) {
    TestRemoteable test = new TestRemoteable(dfe);
    String result = test.cpuLoader1();
    Toast.makeText(StartExecution.this, result, Toast.LENGTH_SHORT).show();
  }

  public void onClickLoader2(View v) {
    TestRemoteable test = new TestRemoteable(dfe);
    String result = test.cpuLoader2();
    Toast.makeText(StartExecution.this, result, Toast.LENGTH_SHORT).show();
  }

  public void onClickLoader3(View v) {
    TestRemoteable test = new TestRemoteable(dfe);
    long result = test.cpuLoader3((int) System.currentTimeMillis());
    Toast.makeText(StartExecution.this, "Result: " + result, Toast.LENGTH_SHORT).show();
  }

  public void onClickJni1(View v) {
    JniTest jni = new JniTest(dfe);

    String result = jni.jniCaller();
    Log.i(TAG, "Result of jni invocation: " + result);

    Toast.makeText(StartExecution.this, result, Toast.LENGTH_SHORT).show();
  }

  public void onClickSudoku(View v) {

    Sudoku sudoku = new Sudoku(dfe);

    boolean result = sudoku.hasSolution();

    if (result) {
      Toast.makeText(StartExecution.this, "Sudoku has solution", Toast.LENGTH_SHORT).show();
    } else {
      Toast.makeText(StartExecution.this, "Sudoku does not have solution", Toast.LENGTH_SHORT)
          .show();
    }
  }

  public void onClickVirusScanning(View v) {
    new VirusTask().execute();
  }

  private class VirusTask extends AsyncTask<Void, Void, Integer> {
    // Show a spinning dialog while solving the puzzle
    ProgressDialog pd = ProgressDialog.show(StartExecution.this, "Working...",
        "Scanning for viruses...", true, false);

    @Override
    protected Integer doInBackground(Void... params) {
      VirusScanning virusScanner = new VirusScanning(getApplicationContext(), dfe, nrClones);
      int nrIterations = 1;

      sleep(0 * 1000);

      int result = -1;
      for (int i = 0; i < nrIterations; i++) {

        result = virusScanner.scanFolder();

        Log.d(TAG, "Number of viruses found: " + result);

        if (i < (nrIterations - 1)) {
          sleep(1 * 1000);
        }
      }
      return result;
    }

    @Override
    protected void onPostExecute(Integer result) {
      Log.i(TAG, "Finished execution");
      if (pd != null) {
        pd.dismiss();
      }
    }
  }

  public void onClickQueenSolver(View v) {
    new NQueensTask().execute();
  }

  private class NQueensTask extends AsyncTask<Void, Void, Integer> {
    // Show a spinning dialog while solving the puzzle
    ProgressDialog pd =
        ProgressDialog.show(StartExecution.this, "Working...", "Solving N Queens...", true, false);

    @Override
    protected Integer doInBackground(Void... params) {

      Spinner nrQueensSpinner = (Spinner) findViewById(R.id.spinnerNrQueens);
      int nrQueens = Integer.parseInt((String) nrQueensSpinner.getSelectedItem());

      int result = -1;
      NQueens puzzle = new NQueens(dfe, nrClones);

      result = puzzle.solveNQueens(nrQueens);

      Log.i(TAG, nrQueens + "-Queens solved, solutions: " + result);
      return result;
    }

    @Override
    protected void onPostExecute(Integer result) {
      Log.i(TAG, "Finished execution");
      if (pd != null) {
        pd.dismiss();
      }
    }
  }

  public void onClickGvirtusDemo(View v) {
    new GvirtusCaller().execute();
  }

  private class GvirtusCaller extends AsyncTask<Void, Void, Void> {
    // Show a spinning dialog while running the GVirtuS demo
    ProgressDialog pd = ProgressDialog.show(StartExecution.this, "Working...",
        "Running the GVirtuS demo...", true, false);

    @Override
    protected Void doInBackground(Void... params) {
      int nrTests = 1;

      GVirtusDemo gvirtusDemo = new GVirtusDemo(dfe);
      for (int i = 0; i < nrTests; i++) {
        Log.i(TAG, "------------ Started running the GVirtuS deviceQuery demo.");
        try {
          gvirtusDemo.deviceQuery();
          Log.i(TAG, "Correctly executed the GVirtuS deviceQuery demo.");
        } catch (IOException e) {
          Log.e(TAG, "Error while running the GVirtuS deviceQuery demo: " + e);
        }
      }

      for (int i = 0; i < nrTests; i++) {
        Log.i(TAG,
            "------------ Started running the GVirtuS matrixMul demo. " + Charset.defaultCharset());
        try {
          gvirtusDemo.matrixMul2();
          Log.i(TAG, "Correctly executed the GVirtuS matrixMul demo.");
        } catch (IOException e) {
          Log.e(TAG, "Error while running the GVirtuS matrixMul demo: " + e);
        }
      }

      return null;
    }

    @Override
    protected void onPostExecute(Void result) {
      Log.i(TAG, "Finished execution");
      if (pd != null) {
        pd.dismiss();
      }
    }
  }

  public void onClickDseTesting(View v) {
    new DseTester().execute();
  }

  private class DseTester extends AsyncTask<Void, Void, Void> {
    // Show a spinning dialog while running the DSE demo
    ProgressDialog pd = ProgressDialog.show(StartExecution.this, "Working...",
        "Running the DSE test...", true, false);

    @Override
    protected Void doInBackground(Void... params) {
      Log.i(TAG, "Deleted the DSE testing implementation on 14/09/2016 15:47. Check the commits.");
      return null;
    }

    @Override
    protected void onPostExecute(Void result) {
      Log.i(TAG, "Finished DSE testing");
      if (pd != null) {
        pd.dismiss();
      }
    }
  }

  public void onRadioExecLocationChecked(View radioButton) {
    switch (radioButton.getId()) {

      case R.id.radio_local:
        dfe.setUserChoice(Constants.LOCATION_LOCAL);
        break;

      case R.id.radio_remote:
        dfe.setUserChoice(Constants.LOCATION_REMOTE);
        break;

      // case R.id.radio_hybrid:
      // dfe.setUserChoice(Constants.LOCATION_HYBRID);
      // break;

      case R.id.radio_exec_time:
        dfe.setUserChoice(Constants.LOCATION_DYNAMIC_TIME);
        break;

      case R.id.radio_energy:
        dfe.setUserChoice(Constants.LOCATION_DYNAMIC_ENERGY);
        break;

      case R.id.radio_exec_time_energy:
        dfe.setUserChoice(Constants.LOCATION_DYNAMIC_TIME_ENERGY);
        break;
    }
  }

  private class NrClonesSelectedListener implements OnItemSelectedListener {

    public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {

      nrClones = Integer.parseInt((String) parent.getItemAtPosition(pos));
      Log.i(TAG, "Number of clones: " + nrClones);
      dfe.setNrClones(nrClones);
    }

    public void onNothingSelected(AdapterView<?> arg0) {
      Log.i(TAG, "Nothing selected on clones spinner");
    }
  };

  public void onTestConnection(View v) {
    new TestConnection().execute();
    sleep(10 * 1000);
    onTestSendBytes(null);
  }


  private class TestConnection extends AsyncTask<Void, String, Void> {

    // Show a spinning dialog
    ProgressDialog pd = ProgressDialog.show(StartExecution.this, "Connection test...",
        "Measuring the connection overhead...", true, false);

    @Override
    protected Void doInBackground(Void... params) {
      Log.i(TAG, "Testing how much it takes to connect to the clone using different strategies.");

      sleep(10 * 1000); // Otherwise the pd will not start
      int NR_TESTS = 100;
      String[] stringCommTypes = {"CLEAR", "SSL"};
      COMM_TYPE[] commTypes = {COMM_TYPE.CLEAR, COMM_TYPE.SSL};

      for (int i = 0; i < commTypes.length; i++) {
        publishProgress("Measuring connection in: " + commTypes[i]);
        File logFile = new File(Constants.TEST_LOGS_FOLDER + "connection_test_" + stringCommTypes[i]
            + "_" + dfe.getConnectionType() + ".csv");
        try {
          logFile.delete();
          logFile.createNewFile();
          BufferedWriter buffLogFile = new BufferedWriter(new FileWriter(logFile, true));

          for (int i1 = 0; i1 < NR_TESTS; i1++) {
            sleep(1 * 1000);
            try {
              dfe.testConnection(commTypes[i], buffLogFile);
            } catch (IOException e) {
              e.printStackTrace();
            }
          }

          buffLogFile.close();
        } catch (IOException e) {
          e.printStackTrace();
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
      Log.i(TAG, "Finished measuring the connection overhead...");
      if (pd != null) {
        pd.dismiss();
      }
    }

    @Override
    protected void onPreExecute() {
      Log.i(TAG, "Started measuring the connection overhead...");
    }

  }

  public void onTestSendBytes(View v) {
    Log.i(TAG,
        "Testing how much it takes to send data of different size to the clone using different strategies.");
    new TestSendBytes().execute();
    sleep(10 * 1000);
    onTestReceiveBytes(null);
  }

  private class TestSendBytes extends AsyncTask<Void, String, Void> {

    // Show a spinning dialog
    ProgressDialog pd = ProgressDialog.show(StartExecution.this, "Working...",
        "Measuring the UL overhead...", true, false);

    @Override
    protected Void doInBackground(Void... params) {
      // SSL_NO_REUSE makes sense only for the connection test.
      // Once the connection is setup it works exactly like SSL.
      // Since I am going to show only the results for 4 bytes, 1KB, and 1MB measure the energy only
      // for these sizes
      COMM_TYPE[] commTypes = {COMM_TYPE.CLEAR, COMM_TYPE.SSL};
      String[] stringCommTypes = {"CLEAR", "SSL"};
      int[] nrTests = {200, 100, 10};
      int[] bytesToSendSize = {4, 1024, 1024 * 1024};
      String[] bytesSizeToSendString = {"4B", "1KB", "1MB"};

      // int[] nrTests = {5};
      // int[] bytesToSendSize = {1 * 1024 * 1024};
      // String[] bytesSizeToSendString = {"1MB"};

      ArrayList<byte[]> bytesToSend = new ArrayList<byte[]>();
      for (int i = 0; i < bytesToSendSize.length; i++) {
        byte[] temp = new byte[bytesToSendSize[i]];
        new Random().nextBytes(temp);
        bytesToSend.add(temp);
      }

      // Measure the increase of data size due to encryption strategies.
      File bytesLogFile = new File(Constants.TEST_LOGS_FOLDER + "send_bytes_test_size.csv");
      BufferedWriter buffBytesLogFile = null;

      try {
        bytesLogFile.delete();
        bytesLogFile.createNewFile();
        buffBytesLogFile = new BufferedWriter(new FileWriter(bytesLogFile, true));
      } catch (IOException e2) {
        e2.printStackTrace();
      }

      // Sleep 10 seconds before starting the energy measurement experiment to avoid the errors
      // introduced by pressing the button
      sleep(10 * 1000);

      for (int j = 0; j < bytesToSendSize.length; j++) {
        try {
          buffBytesLogFile.write(bytesSizeToSendString[j] + "\t");
        } catch (IOException e2) {
          e2.printStackTrace();
        }

        for (int i1 = 0; i1 < commTypes.length; i1++) {

          publishProgress("UL test of " + bytesSizeToSendString[j] + " bytes in " + commTypes[i1]);

          File logFile = new File(Constants.TEST_LOGS_FOLDER + "send_" + bytesToSendSize[j]
              + "_bytes_test_" + stringCommTypes[i1] + "_" + dfe.getConnectionType() + ".csv");

          try {
            logFile.delete();
            logFile.createNewFile();
            BufferedWriter buffLogFile = new BufferedWriter(new FileWriter(logFile, true));

            double totalTxBytes = 0;
            int nrSuccessTests = 0;
            for (int i11 = 0; i11 < nrTests[j]; i11++) {
              try {
                // Change the connection to the new communication protocol
                // I need to reset the connection otherwise the objects are not sent, only the
                // pointer is sent.
                dfe.testConnection(commTypes[i1], null);
                totalTxBytes +=
                    dfe.testSendBytes(bytesToSendSize[j], bytesToSend.get(j), buffLogFile);
                nrSuccessTests++;
              } catch (IOException e1) {
                e1.printStackTrace();
              }
            }

            // totalTxBytes /= nrTests[j];
            totalTxBytes = (nrSuccessTests > 0) ? totalTxBytes /= nrSuccessTests : -1;

            buffBytesLogFile.write(totalTxBytes + ((i1 == commTypes.length - 1) ? "\n" : "\t"));
            buffLogFile.close();
          } catch (IOException e) {
            e.printStackTrace();
          }
          sleep(1 * 1000);
        }
        sleep(1 * 1000);
      }

      try {
        buffBytesLogFile.close();
      } catch (IOException e) {
        e.printStackTrace();
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
      Log.i(TAG, "Finished measuring the UL overhead...");
      if (pd != null) {
        pd.dismiss();
      }
    }
  }

  public void onTestReceiveBytes(View v) {
    Log.i(TAG,
        "Testing how much it takes to receive data of different size from the clone using different strategies.");
    new TestReceiveBytes().execute();
  }

  private class TestReceiveBytes extends AsyncTask<Void, String, Void> {

    // Show a spinning dialog
    ProgressDialog pd = ProgressDialog.show(StartExecution.this, "Working...",
        "Measuring the DL overhead...", true, false);

    @Override
    protected Void doInBackground(Void... params) {
      // SSL_NO_REUSE makes sense only for the connection test.
      // Once the connection is setup it works exactly like SSL.
      // Since I am going to show only the results for 4 bytes, 1KB, and 1MB measure the energy only
      // for these sizes
      COMM_TYPE[] commTypes = {COMM_TYPE.CLEAR, COMM_TYPE.SSL};
      String[] stringCommTypes = {"CLEAR", "SSL"};
      int[] nrTests = {200, 100, 10};
      int[] bytesToReceiveSize = {4, 1024, 1024 * 1024};
      String[] bytesSizeToReceiveString = {"4B", "1KB", "1MB"};

      // int[] nrTests = {5};
      // int[] bytesToReceiveSize = {1 * 1024 * 1024};
      // String[] bytesSizeToReceiveString = {"1MB"};

      // Measure the increase of data size due to encryption strategies.
      File bytesLogFile = new File(Constants.TEST_LOGS_FOLDER + "receive_bytes_test_size.csv");
      BufferedWriter buffBytesLogFile = null;

      try {
        bytesLogFile.delete();
        bytesLogFile.createNewFile();
        buffBytesLogFile = new BufferedWriter(new FileWriter(bytesLogFile, true));
      } catch (IOException e2) {
        e2.printStackTrace();
      }

      // Sleep 10 seconds before starting the energy measurement experiment to avoid the errors
      // introduced by pressing the button
      sleep(10 * 1000);

      for (int bs = 0; bs < bytesToReceiveSize.length; bs++) {
        try {
          buffBytesLogFile.write(bytesSizeToReceiveString[bs] + "\t");
        } catch (IOException e2) {
          e2.printStackTrace();
        }

        for (int ct = 0; ct < commTypes.length; ct++) {

          publishProgress("DL test of " + bytesToReceiveSize[bs] + " bytes in " + commTypes[ct]);

          File logFile = new File(Constants.TEST_LOGS_FOLDER + "receive_" + bytesToReceiveSize[bs]
              + "_bytes_test_" + stringCommTypes[ct] + "_" + dfe.getConnectionType() + ".csv");

          try {
            logFile.delete();
            logFile.createNewFile();
            BufferedWriter buffLogFile = new BufferedWriter(new FileWriter(logFile, true));

            double totalRxBytes = 0;
            int nrSuccessTests = 0;
            for (int i = 0; i < nrTests[bs]; i++) {

              Log.i(TAG, "Receiving " + bytesToReceiveSize[bs] + " bytes using "
                  + stringCommTypes[ct] + " connection");

              try {
                // Change the connection to the new communication protocol
                dfe.testConnection(commTypes[ct], null);
                totalRxBytes += dfe.testReceiveBytes(bytesToReceiveSize[bs], buffLogFile);
                nrSuccessTests++;
              } catch (IOException e1) {
                e1.printStackTrace();
              }
            }
            totalRxBytes = (nrSuccessTests > 0) ? totalRxBytes /= nrSuccessTests : -1;

            buffBytesLogFile.write(totalRxBytes + ((ct == commTypes.length - 1) ? "\n" : "\t"));
            buffLogFile.close();
          } catch (IOException e) {
            e.printStackTrace();
          } catch (ClassNotFoundException e) {
            e.printStackTrace();
          }
          sleep(1 * 1000);
        }
        sleep(1 * 1000);
      }

      try {
        buffBytesLogFile.close();
      } catch (IOException e) {
        e.printStackTrace();
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
      Log.i(TAG, "Finished measuring the DL overhead...");
      if (pd != null) {
        pd.dismiss();
      }
    }

  }

  private void sleep(int millis) {
    try {
      Thread.sleep(millis);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
  }

  public void vmConnectionStatusUpdate() {
    new Thread(new VmConnectionStatusUpdater()).start();
  }
}
