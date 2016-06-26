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
package eu.project.rapid.ac;

import java.io.Serializable;

/**
 * Container of remote execution data - to send back results of the executed
 * operation, the state of the object and actual execution time
 * 
 */
public class ResultContainer implements Serializable {

	/**
     * 
     */
	private static final long serialVersionUID = 6289277906217259082L;

	public Object objState;
	public Object functionResult;
	public Long getObjectDuration;
	public Long pureExecutionDuration;

	/**
	 * Wrapper of results returned by remote server - state of the object the
	 * call was executed on and function result itself
	 * 
	 * @param state
	 *            state of the remoted object
	 * @param result
	 *            result of the function executed on the object
	 */
	public ResultContainer(Object state, Object result, Long getObjectDuration, Long duration) {
		objState = state;
		functionResult = result;
		this.getObjectDuration = getObjectDuration;
		pureExecutionDuration = duration;
	}

	/**
	 * Used when an exception happens, to return the exception as a result of
	 * remote invocation
	 * 
	 * @param result
	 */
	public ResultContainer(Object result, Long getObjectDuration) {
		objState = null;
		functionResult = result;
		this.getObjectDuration = getObjectDuration;
		pureExecutionDuration = null;
	}
}
