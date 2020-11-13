/*******************************************************************
 * @title FLIR THERMAL SDK
 * @file CameraHandler.java
 * @Author FLIR Systems AB
 *
 * @brief Helper class that encapsulates *most* interactions with a FLIR ONE camera
 *
 * Copyright 2019:    FLIR Systems
 ********************************************************************/
package com.examples.ThermalPeek;

import com.flir.thermalsdk.live.CommunicationInterface;
import com.flir.thermalsdk.live.Identity;
import com.flir.thermalsdk.live.connectivity.ConnectionStatusListener;
import com.flir.thermalsdk.live.discovery.DiscoveryEventListener;
import com.flir.thermalsdk.live.discovery.DiscoveryFactory;

/**
 * Encapsulates the handling of a FLIR ONE camera or built in emulator, discovery, connecting and start receiving images.
 * All listeners are called from Thermal SDK on a non-ui thread
 * <p/>
 * Usage:
 * <pre>
 * Start discovery of FLIR FLIR ONE cameras or built in FLIR ONE cameras emulators
 * {@linkplain #startDiscovery(DiscoveryEventListener, DiscoveryStatus)}
 * Use a discovered Camera {@linkplain Identity} and connect to the Camera
 * {@linkplain #connect(Identity, ConnectionStatusListener)}
 * Once connected to a camera
 * {@linkplain #startStream(StreamDataListener)}
 * </pre>
 * <p/>
 * You don't *have* to specify your application to listen or USB intents but it might be beneficial for you application,
 * we are enumerating the USB devices during the discovery process which eliminates the need to listen for USB intents.
 * See the Android documentation about USB Host mode for more information
 * <p/>
 * Please note, this is <b>NOT</b> production quality code, error handling has been kept to a minimum to keep the code as clear and concise as possible
 */
class CameraHandler {

    private static final String TAG = "CameraHandler";


    public interface DiscoveryStatus {
        void started();
        void stopped();
    }

    public CameraHandler() {
    }

    /**
     * Start discovery of USB and Emulators
     */
    public void startDiscovery(DiscoveryEventListener cameraDiscoveryListener, DiscoveryStatus discoveryStatus) {
        DiscoveryFactory.getInstance().scan(cameraDiscoveryListener, CommunicationInterface.EMULATOR, CommunicationInterface.USB);
        discoveryStatus.started();
    }

    /**
     * Stop discovery of USB and Emulators
     */
    public void stopDiscovery(DiscoveryStatus discoveryStatus) {
        DiscoveryFactory.getInstance().stop(CommunicationInterface.EMULATOR, CommunicationInterface.USB);
        discoveryStatus.stopped();
    }




}
