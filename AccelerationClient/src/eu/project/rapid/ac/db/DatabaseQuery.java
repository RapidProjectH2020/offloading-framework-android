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
package eu.project.rapid.ac.db;

import java.util.ArrayList;
import android.content.Context;
import android.database.Cursor;

/**
 * This class adds multiple entries to the database and pulls them back
 * out.
 */
public class DatabaseQuery {
	private ArrayList<String> arrayKeys = null;
	private ArrayList<String> arrayValues = null;
	private ArrayList<String> databaseKeys = null;
	private ArrayList<String> databaseKeyOptions = null;
	
	public static final String LOG_TABLE	 		= "logTable";
	public static final String KEY_APP_NAME 		= "appName";
	public static final String KEY_METHOD_NAME 		= "methodName";
	public static final String KEY_EXEC_LOCATION 	= "execLocation";
	public static final String KEY_NETWORK_TYPE 	= "networkType";
	public static final String KEY_NETWORK_SUBTYPE 	= "networkSubType";
	public static final String KEY_UL_RATE 			= "ulRate";
	public static final String KEY_DL_RATE 			= "dlRate";
	public static final String KEY_EXEC_DURATION 	= "execDuration";
	public static final String KEY_EXEC_ENERGY 		= "execEnergy";
	public static final String KEY_TIMESTAMP 		= "timestamp";
	
	private DBAdapter database;
	
	/**
	 * Initialize the ArrayList
	 * @param context Pass context from calling class.
	 */
	public DatabaseQuery(Context context) {
		// Create an ArrayList of keys and one of the options/parameters
		// for the keys.
		databaseKeys = new ArrayList<String>();
		databaseKeyOptions = new ArrayList<String>();
		
		databaseKeys.add(KEY_APP_NAME);
		databaseKeyOptions.add("text not null");
		
		databaseKeys.add(KEY_METHOD_NAME);
		databaseKeyOptions.add("text not null");
		
		databaseKeys.add(KEY_EXEC_LOCATION);
		databaseKeyOptions.add("text not null");
		
		databaseKeys.add(KEY_NETWORK_TYPE);
		databaseKeyOptions.add("text");
		
		databaseKeys.add(KEY_NETWORK_SUBTYPE);
		databaseKeyOptions.add("text");
		
		databaseKeys.add(KEY_UL_RATE);
		databaseKeyOptions.add("text");
		
		databaseKeys.add(KEY_DL_RATE);
		databaseKeyOptions.add("text");
		
		databaseKeys.add(KEY_EXEC_DURATION);
		databaseKeyOptions.add("text");
		
		databaseKeys.add(KEY_EXEC_ENERGY);
		databaseKeyOptions.add("text");
		
		databaseKeys.add(KEY_TIMESTAMP);
		databaseKeyOptions.add("text");
		
		// Call the database adapter to create the database
		database = new DBAdapter(context, LOG_TABLE, databaseKeys, databaseKeyOptions);
        database.open();
		arrayKeys = new ArrayList<String>();
		arrayValues = new ArrayList<String>();

	}
	
	/**
	 * Append data to an ArrayList to then submit to the database
	 * @param key Key of the value being appended to the Array.
	 * @param value Value to be appended to Array.
	 */
	public void appendData(String key, String value){
		arrayKeys.add(key);
		arrayValues.add(value);
	}
	
	/**
	 * This method adds the row created by appending data to the database.
	 * The parameters constitute one row of data.
	 */
	public void addRow(){
		database.insertEntry(arrayKeys, arrayValues);
	}
	
	public void updateRow(){
		database.updateEntry(arrayKeys, arrayValues);
	}
	
	/**
	 * Get data from the table.
	 * @param keys List of columns to include in the result.
	 * @param selection Return rows with the following string only. Null returns all rows.
	 * @param selectionArgs Arguments of the selection.
	 * @param groupBy Group results by.
	 * @param having A filter declare which row groups to include in the cursor.
	 * @param sortBy Column to sort elements by.
	 * @param sortOption ASC for ascending, DESC for descending.
	 * @return Returns an ArrayList with ONLY the results of the selected sortBy field.
	 */
	public ArrayList<String> getData(String[] keys, String selection, String[] 
	  selectionArgs, String groupBy, String having, String sortBy, String sortOption){
		
		ArrayList<String> list = new ArrayList<String>(); 
		Cursor results = database.getAllEntries(keys, selection, 
				selectionArgs, groupBy, having, sortBy, sortOption);
		while(results.moveToNext())
		{
			list.add(results.getString(results.getColumnIndex(sortBy)));
		}
		
		try{
			results.close();
		}
		finally {
		}
		
		return list;
	}
	
	/**
	 * Get data from the table.
	 * @param keys List of columns to include in the result.
	 * @param selection Return rows with the following string only. Null returns all rows.
	 * @param selectionArgs Arguments of the selection.
	 * @param groupBy Group results by.
	 * @param having A filter declare which row groups to include in the cursor.
	 * @param sortBy Column to sort elements by.
	 * @param sortOption ASC for ascending, DESC for descending.
	 * @return Returns a Cursor with ALL the rows sorted by the given value.
	 * 
	 * @author sokolkosta
	 */
	public Cursor getAllEntries(String[] keys, String selection, String[] selectionArgs) {
		return database.getAllEntries(keys, selection, selectionArgs);
	}
	
	/**
	 * Destroy the reporter.
	 * @throws Throwable
	 */
	public void destroy() throws Throwable{
        database.close();
	}
}
