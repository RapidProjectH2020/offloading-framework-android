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

import java.security.Security;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.RadioGroup;
import android.widget.RadioGroup.OnCheckedChangeListener;
import eu.project.rapid.ac.DFE;
import eu.project.rapid.common.RapidConstants.COMM_TYPE;
import eu.project.rapid.common.RapidUtils;

/**
 * The main activity for the Android program.
 *
 */
public class MainActivity extends Activity {

  private static final String TAG = "MainActivity";
  public static final String KEY_VM_IP = "KEY_VM_IP";
  public static final String KEY_USE_RAPID_INFRASTRUCTURE = "KEY_USE_RAPID_INFRASTRUCTURE";

  private RadioGroup radioGroupStartAs;
  private RadioGroup radioGroupUseRapid;
  private EditText textVmIpAddress;
  private String vmIp;

  static {
    Security.insertProviderAt(new org.spongycastle.jce.provider.BouncyCastleProvider(), 1);
  }

  /** Called when the activity is first created. */
  @Override
  public void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    radioGroupUseRapid = (RadioGroup) findViewById(R.id.radioGroupUseRapid);
    radioGroupUseRapid.setOnCheckedChangeListener(new OnUseRapidRadioChecked());

    radioGroupStartAs = (RadioGroup) findViewById(R.id.radioGroupStartAs);
    radioGroupStartAs.setOnCheckedChangeListener(new OnStartAsRadioChecked());

    textVmIpAddress = (EditText) findViewById(R.id.editTextIpAddress);
    SharedPreferences prefs = getPreferences(Context.MODE_PRIVATE);
    vmIp = prefs.getString(KEY_VM_IP, null);
    textVmIpAddress.setText(vmIp);
  }

  public void onRadioCommunicationTypeChecked(View radioButton) {

    switch (radioButton.getId()) {

      case R.id.radio_clear_communication:
        DFE.commType = COMM_TYPE.CLEAR;
        Log.i(TAG, "The communication UE-VM should be performed in clear");
        break;

      case R.id.radio_ssl_communication:
        DFE.commType = COMM_TYPE.SSL;
        Log.i(TAG, "The communication UE-VM should be performed using SSL");
        break;

      case R.id.radio_choose_from_annotation_communication:
        Log.w(TAG, "TODO: The communication UE-VM will be decided based on methods' annotations");
        break;
    }
  }

  // Watch for button clicks
  public void onStartButton(View v) {
    Intent intent = new Intent(v.getContext(), StartExecution.class);

    // If the user is setting the IP of the VM automatically, we should check that the IP is
    // correctly formatted.
    if (radioGroupUseRapid.getCheckedRadioButtonId() == R.id.radioUseRapidNo) {
      vmIp = textVmIpAddress.getText().toString();

      if (!RapidUtils.validateIpAddress(vmIp)) {
        textVmIpAddress.setTextColor(Color.RED);
      } else {
        textVmIpAddress.setTextColor(Color.GREEN);
        Log.i(TAG, "Creating a connection with VM with IP: " + vmIp);
        intent.putExtra(KEY_VM_IP, vmIp);
        intent.putExtra(KEY_USE_RAPID_INFRASTRUCTURE, false);
        SharedPreferences prefs = getPreferences(Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = prefs.edit();
        editor.putString(KEY_VM_IP, vmIp);
        editor.commit();
        startActivity(intent);
      }
    } else {
      // Should use the Rapid infrastructure.
      intent.putExtra(KEY_USE_RAPID_INFRASTRUCTURE, true);
      startActivity(intent);
    }
  }

  private class OnStartAsRadioChecked implements OnCheckedChangeListener {
    public void onCheckedChanged(RadioGroup group, int checkedId) {

      switch (checkedId) {
        case R.id.radioStartAsNewVm:
          DFE.CONNECT_TO_PREVIOUS_VM = false;
          break;

        case R.id.radioStartAsOldVm:
          DFE.CONNECT_TO_PREVIOUS_VM = true;
          break;

        case R.id.radioStartAsD2D:
          break;
      }
    }
  }

  private class OnUseRapidRadioChecked implements OnCheckedChangeListener {

    public void onCheckedChanged(RadioGroup group, int checkedId) {
      switch (checkedId) {
        case R.id.radioUseRapidYes:
          textVmIpAddress.setVisibility(View.GONE);
          radioGroupStartAs.setVisibility(View.VISIBLE);
          break;

        case R.id.radioUseRapidNo:
          radioGroupStartAs.setVisibility(View.GONE);
          textVmIpAddress.setVisibility(View.VISIBLE);
          break;
      }
    }
  }
}
