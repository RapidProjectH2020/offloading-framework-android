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
package eu.project.rapid.ac.d2d;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.MulticastSocket;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import android.app.IntentService;
import android.content.Context;
import android.content.Intent;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.util.Log;
import eu.project.rapid.ac.d2d.D2DMessage.MsgType;
import eu.project.rapid.ac.utils.Constants;
import eu.project.rapid.ac.utils.Utils;

/**
 * This thread will be started by clients that run the DFE so that these clients can get the HELLO
 * messages sent by the devices that act as D2D Acceleration Server.
 * 
 * @author sokol
 *
 */
public class D2DClientService extends IntentService {

  private static final String TAG = D2DClientService.class.getName();
  ScheduledThreadPoolExecutor setWriterThread =
      (ScheduledThreadPoolExecutor) Executors.newScheduledThreadPool(5);
  private D2DSetWriter setWriterRunnable;
  public static final int FREQUENCY_WRITE_D2D_SET = 5 * 60 * 1013; // Every 5 minutes save the set
  public static final int FREQUENCY_READ_D2D_SET = 1 * 60 * 1011; // Every 1 minute read the set
  // from the file
  private MulticastSocket receiveSocket;
  private Set<PhoneSpecs> setD2dPhones = new TreeSet<PhoneSpecs>(); // Sorted by specs

  public D2DClientService() {
    super(D2DClientService.class.getName());
  }

  @Override
  protected void onHandleIntent(Intent intent) {
    if (setWriterRunnable == null) {
      setWriterRunnable = new D2DSetWriter();
    }
    setWriterThread.scheduleWithFixedDelay(setWriterRunnable, FREQUENCY_WRITE_D2D_SET,
        FREQUENCY_WRITE_D2D_SET, TimeUnit.MILLISECONDS);

    try {
      Log.i(TAG, "Thread started");
      writeSetOnFile();

      WifiManager.MulticastLock lock = null;
      WifiManager wifi = (WifiManager) getSystemService(Context.WIFI_SERVICE);
      Log.i(TAG, "Trying to acquire multicast lock...");
      if (Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.KITKAT) {
        if (lock == null) {
          Log.i(TAG, "lock was null, creating...");
          lock = wifi.createMulticastLock("WiFi_Lock");
        }
        lock.setReferenceCounted(true);
        lock.acquire();
        Log.i(TAG, "Lock acquired!");
      }

      receiveSocket = new MulticastSocket(Constants.D2D_BROADCAST_PORT);
      receiveSocket.setBroadcast(true);
      Log.i(TAG, "Started listening on multicast socket.");

      try {
        // This will be interrupted when the OS kills the service
        while (true) {
          Log.i(TAG, "Waiting for broadcasted data...");
          byte[] data = new byte[1024];
          DatagramPacket packet = new DatagramPacket(data, data.length);
          receiveSocket.receive(packet);
          Log.d(TAG, "Received a new broadcast packet from: " + packet.getAddress());
          processPacket(packet);
        }
      } catch (IOException e) {
        Log.d(TAG, "The socket was closed.");
      }

      Log.i(TAG, "Stopped receiving data!");
    } catch (IOException e) {
      // We expect this to happen when more than one DFE on the same phone will try to create
      // this service and the port will be busy. This way only one service will be listening for D2D
      // messages. This service will be responsible for writing the received messages on a file so
      // that the DFEs of all applications could read them.
      Log.d(TAG,
          "Could not create D2D multicast socket, maybe the service is already started by another DFE: "
              + e);
      // e.printStackTrace();
    }
  }

  /**
   * Process the packet received by another device in a D2D scenario. Create a D2Dmessage and if
   * this is a HELLO message then store the specifics of the other device into the Map. If a new
   * device is added to the map and more than 5 minutes have passed since the last time we saved the
   * devices on the file, then save the set in the filesystem so that other DFEs can read it.
   * 
   * @param packet
   */
  private void processPacket(DatagramPacket packet) {
    try {
      D2DMessage msg = new D2DMessage(packet.getData());
      Log.d(TAG, "Received: <== " + msg);
      if (msg.getMsgType() == MsgType.HELLO) {
        PhoneSpecs otherPhone = msg.getPhoneSpecs();
        if (setD2dPhones.contains(otherPhone)) {
          setD2dPhones.remove(otherPhone);
        }
        otherPhone.setTimestamp(System.currentTimeMillis());
        otherPhone.setIp(packet.getAddress().getHostAddress());
        setD2dPhones.add(otherPhone);
        // FIXME writing the set here is too heavy but I want this just for the demo. Later fix this
        // with a smarter alternative.
        writeSetOnFile();
      }
    } catch (IOException | ClassNotFoundException e) {
      Log.e(TAG, "Error while processing the packet: " + e);
    }
  }

  private class D2DSetWriter implements Runnable {
    @Override
    public void run() {
      // Write the set in the filesystem so that other DFEs can use the D2D phones when needed.
      Iterator<PhoneSpecs> it = setD2dPhones.iterator();
      // First clean the set from devices that have not been pinging recently
      while (it.hasNext()) {
        // If the last time we have seen this device is 5 pings before, then remove it.
        if ((System.currentTimeMillis() - it.next().getTimestamp()) > 5
            * Constants.D2D_BROADCAST_INTERVAL) {
          it.remove();
        }
      }
      writeSetOnFile();
    }
  }

  private void writeSetOnFile() {

    try {
      Log.i(TAG, "Writing set of D2D devices on the sdcard file");
      // This method is blocking, waiting for the lock on the file to be available.
      Utils.writeObjectToFile(Constants.FILE_D2D_PHONES, setD2dPhones);
      Log.i(TAG, "Finished writing set of D2D devices on the sdcard file");
    } catch (IOException e) {
      Log.e(TAG, "Error while writing set of D2D devices on the sdcard file: " + e);
    }
  }
}
