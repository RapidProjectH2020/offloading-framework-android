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
import java.io.Serializable;
import java.io.StreamCorruptedException;

import android.content.Context;
import eu.project.rapid.ac.utils.Utils;

/**
 * The message that will be broadcasted by the devices that run the Acceleration Server and are
 * willing to participate in D2D offloading service.
 */
public class D2DMessage implements Serializable {
  private static final long serialVersionUID = -8833550140715953630L;

  private MsgType msgType;
  private PhoneSpecs phoneSpecs;

  public enum MsgType {
    HELLO;
  }

  /**
   * Message that will be sent usually in broadcast by the device running the AccelerationServer
   * 
   * @param msgType
   */
  public D2DMessage(Context context, MsgType msgType) {
    this.msgType = msgType;
    this.phoneSpecs = PhoneSpecs.getPhoneSpecs(context);
  }

  /**
   * Constructor that will be usually used by the device running the AccelerationClient
   * 
   * @param byteArray
   * @throws ClassNotFoundException
   * @throws IOException
   * @throws StreamCorruptedException
   */
  public D2DMessage(byte[] byteArray)
      throws StreamCorruptedException, IOException, ClassNotFoundException {
    D2DMessage otherDevice = (D2DMessage) Utils.byteArrayToObject(byteArray);
    this.msgType = otherDevice.getMsgType();
    this.phoneSpecs = otherDevice.getPhoneSpecs();
  }

  /**
   * @return the msgType
   */
  public MsgType getMsgType() {
    return msgType;
  }

  /**
   * @param msgType the msgType to set
   */
  public void setMsgType(MsgType msgType) {
    this.msgType = msgType;
  }

  public PhoneSpecs getPhoneSpecs() {
    return phoneSpecs;
  }

  public void setPhoneSpecs(PhoneSpecs phoneSpecs) {
    this.phoneSpecs = phoneSpecs;
  }

  @Override
  public String toString() {
    return "[" + this.msgType.toString() + this.phoneSpecs + "]";
  }
}
