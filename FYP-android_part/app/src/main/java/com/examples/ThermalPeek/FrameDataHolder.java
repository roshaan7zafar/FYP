/*******************************************************************
 * @title FLIR THERMAL SDK
 * @file FrameDataHolder.java
 * @Author FLIR Systems AB
 *
 * @brief Container class that holds references to Bitmap images
 *
 * Copyright 2019:    FLIR Systems
 ********************************************************************/

package com.examples.ThermalPeek;

import android.graphics.Bitmap;

class FrameDataHolder {

    public final Bitmap msxBitmap;


    FrameDataHolder(Bitmap msxBitmap){
        this.msxBitmap = msxBitmap;

    }
}
