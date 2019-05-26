package dev.tae;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.os.Environment;
import android.util.Log;

import com.googlecode.leptonica.android.Pixa;
import com.googlecode.tesseract.android.TessBaseAPI;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class TessOCR2 {
    public static final String PACKAGE_NAME = "dev.tae";
    public static final String DATA_PATH = Environment
            .getExternalStorageDirectory().toString() + "/Blinder/";
    public static final String lang = "por";

    private static final String TAG = "TESSERACT";
    private AssetManager assetManager;

    private static TessBaseAPI mTess = new TessBaseAPI();

    public TessOCR2(AssetManager assetManager) {

        Log.i(TAG, DATA_PATH);

        this.assetManager = assetManager;

        String[] paths = new String[]{DATA_PATH, DATA_PATH + "tessdata/"};

        for (String path : paths) {
            File dir = new File(path);
            if (!dir.exists()) {
                if (!dir.mkdirs()) {
                    Log.v(TAG, "ERROR: Creation of directory " + path + " on sdcard failed");
                    return;
                } else {
                    Log.v(TAG, "Created directory " + path + " on sdcard");
                }
            }
        }

        if (!(new File(DATA_PATH + "tessdata/" + lang + ".traineddata")).exists()) {
            try {
                InputStream in = assetManager.open("tessdata/" + lang + ".traineddata");
                OutputStream out = new FileOutputStream(new File(DATA_PATH + "tessdata/", lang + ".traineddata"));

                byte[] buf = new byte[1024];
                int len;
                while ((len = in.read(buf)) != -1) {
                    out.write(buf, 0, len);
                }
                in.close();
                out.close();

                Log.v(TAG, "Copied " + lang + " traineddata");
            } catch (IOException e) {
                Log.e(TAG, "Was unable to copy " + lang + " traineddata " + e.toString());
            }
        }

        mTess.setVariable(TessBaseAPI.VAR_CHAR_WHITELIST, "0123456789qwertyuiopasdfghjklçzxcvbnm" +
                "QWERTYUIOPASDFGHJKLÇZXCVBNM");
        mTess.setDebug(true);
        mTess.init(DATA_PATH, lang);
        mTess.setPageSegMode(TessBaseAPI.PageSegMode.PSM_AUTO);

    }


    public String getResults(Bitmap bitmap) {
        mTess.setImage(bitmap);
        String result = mTess.getUTF8Text();
        return result;
    }

    public void onDestroy() {
        if (mTess != null)
            mTess.end();
    }

    public List<Rect> getTextRegions(Bitmap bmp) {
        mTess.setImage(bmp);
        Pixa regions = mTess.getWords();

        List<Rect> lineRects = new ArrayList<>(regions.getBoxRects());
        Log.i("Text Regions ---->", String.valueOf(lineRects.size()));
        regions.recycle();
        return lineRects;
    }
}
