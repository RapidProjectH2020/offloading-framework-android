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

import java.io.IOException;
import java.nio.charset.Charset;

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

  private TextView gvirtusOutputView;
  private String gvirtusOutputText;

  private String vmIp;
  private DFE dFE;
  private boolean useRapidInfrastructure;

  /** Called when the activity is first created. */
  @Override
  public void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.main);
    Log.i(TAG, "onCreate");

    gvirtusOutputView = (TextView) findViewById(R.id.gvirtusOutputView);

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
      dFE = new DFE(getPackageName(), getPackageManager(), this);
    } else {
      dFE = new DFE(getPackageName(), getPackageManager(), this, new Clone("vb-clone-0", vmIp));
    }
  }

  private class VmConnectionStatusUpdater implements Runnable {

    public void run() {
      if (DFE.onLine) {
        handler.post(new Runnable() {
          public void run() {
            textViewVmConnected.setText(R.string.textVmConnected);
            textViewVmConnected.setTextColor(Color.GREEN);

            for (int i = 0; i < executionRadioGroup.getChildCount(); i++) {
              executionRadioGroup.getChildAt(i).setEnabled(true);
            }
          }
        });
      }
    }
  }

  @Override
  public void onDestroy() {
    super.onDestroy();
    Log.i(TAG, "onDestroy");

    if (dFE != null) {
      dFE.onDestroy();
    }
  }

  @Override
  public void onPause() {
    super.onPause();
    Log.d(TAG, "OnPause");
  }

  public void onClickLoader1(View v) {
    TestRemoteable test = new TestRemoteable(dFE);
    String result = test.cpuLoader1();
    Toast.makeText(StartExecution.this, result, Toast.LENGTH_SHORT).show();
  }

  public void onClickLoader2(View v) {
    TestRemoteable test = new TestRemoteable(dFE);
    String result = test.cpuLoader2();
    Toast.makeText(StartExecution.this, result, Toast.LENGTH_SHORT).show();
  }

  public void onClickLoader3(View v) {
    TestRemoteable test = new TestRemoteable(dFE);
    long result = test.cpuLoader3((int) System.currentTimeMillis());
    Toast.makeText(StartExecution.this, "Result: " + result, Toast.LENGTH_SHORT).show();
  }

  public void onClickJni1(View v) {
    JniTest jni = new JniTest(dFE);

    String result = jni.jniCaller();

    Toast.makeText(StartExecution.this, result, Toast.LENGTH_SHORT).show();
  }

  public void onClickSudoku(View v) {

    Sudoku sudoku = new Sudoku(dFE);

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
    // Show a spinning dialog while scanning for viruses
    ProgressDialog pd = ProgressDialog.show(StartExecution.this, "Working...",
        "Scanning for viruses...", true, false);

    @Override
    protected Integer doInBackground(Void... params) {
      VirusScanning virusScanner = new VirusScanning(getApplicationContext(), dFE, nrClones);
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
      int nrTests = 1;

      Spinner nrQueensSpinner = (Spinner) findViewById(R.id.spinnerNrQueens);
      int nrQueens = Integer.parseInt((String) nrQueensSpinner.getSelectedItem());

      NQueens puzzle = new NQueens(dFE, nrClones);

      int result = -1;
      for (int i = 0; i < nrTests; i++) {
        // Choosing a random number of queens for testing purposes.
        // nrQueens = 4 + new Random().nextInt(3);

        Log.i(TAG, "Nr Queens: " + nrQueens);
        result = puzzle.solveNQueens(nrQueens);
        Log.i(TAG, "EightQueens solved, solutions: " + result);
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

      GVirtusDemo gvirtusDemo = new GVirtusDemo(dFE);
      for (int i = 0; i < nrTests; i++) {
        Log.i(TAG, "------------ Started running the GVirtuS deviceQuery demo.");
        try {
          gvirtusOutputText = gvirtusDemo.deviceQuery();
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
      gvirtusOutputView.setText("Correctly executed the GVirtuS deviceQuery demo.");
      gvirtusOutputView.setText(gvirtusOutputText);
    }
  }

  public void onRadioExecLocationChecked(View radioButton) {
    switch (radioButton.getId()) {

      case R.id.radio_local:
        dFE.setUserChoice(Constants.LOCATION_LOCAL);
        break;

      case R.id.radio_remote:
        dFE.setUserChoice(Constants.LOCATION_REMOTE);
        break;

      case R.id.radio_hybrid:
        dFE.setUserChoice(Constants.LOCATION_HYBRID);
        break;

      case R.id.radio_exec_time:
        dFE.setUserChoice(Constants.LOCATION_DYNAMIC_TIME);
        break;

      case R.id.radio_energy:
        dFE.setUserChoice(Constants.LOCATION_DYNAMIC_ENERGY);
        break;

      case R.id.radio_exec_time_energy:
        dFE.setUserChoice(Constants.LOCATION_DYNAMIC_TIME_ENERGY);
        break;
    }
  }

  private class NrClonesSelectedListener implements OnItemSelectedListener {

    public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {

      nrClones = Integer.parseInt((String) parent.getItemAtPosition(pos));
      Log.i(TAG, "Number of clones: " + nrClones);
      dFE.setNrClones(nrClones);
    }

    public void onNothingSelected(AdapterView<?> arg0) {
      Log.i(TAG, "Nothing selected on clones spinner");
    }
  };

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
