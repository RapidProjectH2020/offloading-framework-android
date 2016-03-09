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
package eu.project.rapid.ac.profilers;

public class NetworkBWRecord {
	
	private int bw;
	private long timestamp;
	
	public NetworkBWRecord() {
		this(-1, -1);
	}
	
	/**
	 * 
	 * @param ulRate
	 * @param dlRate
	 * @param timestamp
	 * @param location
	 */
	public NetworkBWRecord(int bw, long timestamp) {
		this.bw = bw;
		this.timestamp = timestamp;
	}
	
	/**
	 * @return the bw
	 */
	public int getBw() {
		return bw;
	}

	/**
	 * @param bw the bw to set
	 */
	public void setBw(int bw) {
		this.bw = bw;
	}

	/**
	 * @return the timestamp
	 */
	public long getTimestamp() {
		return timestamp;
	}
	/**
	 * @param timestamp the timestamp to set
	 */
	public void setTimestamp(long timestamp) {
		this.timestamp = timestamp;
	}
}
