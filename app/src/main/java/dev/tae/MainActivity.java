package dev.tae;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraManager;
import android.os.Build;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.support.annotation.RequiresApi;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.util.SparseArray;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.widget.TextView;

import com.google.android.gms.vision.CameraSource;
import com.google.android.gms.vision.Detector;
import com.google.android.gms.vision.text.TextBlock;
import com.google.android.gms.vision.text.TextRecognizer;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Observable;
import java.util.Observer;


public class MainActivity extends AppCompatActivity {


    SurfaceView surfaceView;
    CameraSource cameraSource;
    TextView textView;

    Vibrator v;

    boolean flag;
    int sens;
    Thread t = new Thread();

    private void vibrate(final int time, final int factor, final int it) {
        if (t.isAlive()) {
            //t.destroy();
        } else {
            t = new Thread(new Runnable() {
                @Override
                public void run() {
                    for (int i = 0; i <= it; i++) {
                        if (i % factor == 0) {
                            v.vibrate(time);
                        } else {
                            v.cancel();
                        }
                    }
                }
            });
            t.start();
        }
    }

    private void fps() {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.LOLLIPOP) {
            CameraManager cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
            try {
                String[] cameraIdList = cameraManager.getCameraIdList();

            } catch (CameraAccessException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        //long[] mVibratePattern = new long[]{0, 400, 200, 400};
        //v.vibrate(mVibratePattern,0);


        Log.i("VERSÃO SDK", String.valueOf(Build.VERSION.SDK_INT));

        surfaceView = (CameraBridgeViewBase) findViewById(R.id.java_camera_view);
        textView = findViewById(R.id.text_view);
        v = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);


        final TextRecognizer textRecognizer = new TextRecognizer.Builder(getApplicationContext()).build();

        cameraSource = new CameraSource.Builder(getApplicationContext(), textRecognizer)
                .setFacing(CameraSource.CAMERA_FACING_BACK)
                .setRequestedFps(1.0f)
                .setAutoFocusEnabled(true)
                .setRequestedPreviewSize(1280, 1024)
                .build();


        surfaceView.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(SurfaceHolder holder) {
                try {

                    if (ActivityCompat.checkSelfPermission(getApplicationContext(),
                            Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {

                        ActivityCompat.requestPermissions(MainActivity.this,
                                new String[]{Manifest.permission.CAMERA},
                                1001);
                        return;
                    }
                    cameraSource.start(surfaceView.getHolder());

                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            @Override
            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {

            }

            @Override
            public void surfaceDestroyed(SurfaceHolder holder) {
                cameraSource.stop();
            }
        });

        final ArrayList<String> strings = new ArrayList<String>();


        //VibrationEffect effect = VibrationEffect.createOneShot(1000,25);
        textRecognizer.setProcessor(new Detector.Processor<TextBlock>() {
            int nDetected;

            @Override
            public void release() {

            }

            @Override
            public void receiveDetections(final Detector.Detections<TextBlock> detections) {
                final SparseArray<TextBlock> items = detections.getDetectedItems();

                nDetected = items.size();

                //int[] amplitude = new int[]{0, nDetected};
                //effect = VibrationEffect.createWaveform(tempo,amplitude,-1);
                //effect = VibrationEffect.createOneShot(1000,nDetected);
                Log.i("nDetected", String.valueOf(nDetected));

                //v.vibrate(nDetected);
                //v.vibrate(VibrationEffect.createOneShot(1000,VibrationEffect.DEFAULT_AMPLITUDE));
                if (items.size() != 0) {

                    textView.post(new Runnable() {
                        @Override
                        public void run() {
                            StringBuilder stringBuilder = new StringBuilder();
                            for (int i = 0; i < items.size(); i++) {
                                TextBlock item = items.valueAt(i);
                                stringBuilder.append(item.getValue());
                                stringBuilder.append("\n");
                                Log.i("ITEMS---->", items.valueAt(i).toString());
                                Log.i("ITEM---->", item.getValue());
                            }
                            textView.setText(stringBuilder.toString());
                            strings.add(stringBuilder.toString());


                            //v.vibrate(nDetected*50);

                            if (strings.size() > 1) {
                                int dif = strings.get(strings.size() - 1).length() - (strings.get(strings.size() - 2).length());
                                Log.i("DIFERENÇA", String.valueOf(dif));

                                if (dif > 2 || dif < -2) {
                                    t = new Thread();
                                    if (nDetected >= 1 && nDetected <= 10) {
                                        vibrate(70, 1, 500);
                                    }
                                    if (nDetected > 10 && nDetected <= 15) {
                                        vibrate(70, 3, 500);
                                    }
                                    if (nDetected > 15 && nDetected <= 25) {
                                        vibrate(70, 5, 500);
                                    }
                                    if (nDetected > 25) {
                                        vibrate(70, 6, 500);
                                    }

                                }
                            }
                        }
                    });
                }
            }
        });


    }


}
