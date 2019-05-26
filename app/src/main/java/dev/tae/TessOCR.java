package dev.tae;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.os.Environment;
import android.util.Log;
import android.widget.Toast;

import com.googlecode.leptonica.android.Pixa;
import com.googlecode.tesseract.android.TessBaseAPI;

import org.opencv.core.Mat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import static com.googlecode.tesseract.android.TessBaseAPI.PageSegMode.PSM_AUTO;

public class TessOCR {
    private static TessBaseAPI mTess = new TessBaseAPI();
    private Context context;
    private String datapath;

    private static final String TAG = "TextRecognitionHelper";

    private static final String TESSERACT_TRAINED_DATA_FOLDER = "/tessdata";
    private static final String TESSERACT_PATH = Environment.getExternalStorageDirectory()
            .getAbsolutePath();

    //private final String TESSERACT_PATH = context.getFilesDir().getAbsolutePath();


    public List<Rect> regions;

    private void prepareTessData() {
        try {
            File dir = context.getExternalFilesDir(TESSERACT_TRAINED_DATA_FOLDER);
            if (!dir.exists()) {
                if (!dir.mkdir()) {
                    Toast.makeText(context, "The folder " + dir.getPath() + "was not created", Toast.LENGTH_SHORT).show();
                }
            }
            String fileList[] = context.getAssets().list("tessdata");

            for (String fileName : fileList) {
                String pathToDataFile = dir + "/" + fileName;
                if (!(new File(pathToDataFile)).exists()) {
                    InputStream in = context.getAssets().open("tessdata/" + fileName);
                    OutputStream out = new FileOutputStream(pathToDataFile);
                    byte[] buff = new byte[1024];
                    int len;
                    while ((len = in.read(buff)) > 0) {
                        out.write(buff, 0, len);
                    }
                    in.close();
                    out.close();
                }
            }
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
        }
    }

    public void prepareTesseract(final String language) {

        System.out.println(TESSERACT_PATH + TESSERACT_TRAINED_DATA_FOLDER);
        try {
            prepareDirectory(TESSERACT_PATH + TESSERACT_TRAINED_DATA_FOLDER);

        } catch (Exception e) {
            e.printStackTrace();
        }

        copyTessDataFiles(TESSERACT_TRAINED_DATA_FOLDER);
        mTess.init(TESSERACT_PATH, language);
    }

    private void prepareDirectory(String path) {
        Log.i("TessPath", path);
        File dir = new File(path);
        if (!dir.exists()) {

            if (!dir.mkdirs()) {
                Log.e(TAG,
                        "ERROR: Creation of directory " + path + " failed, check does Android Manifest have permission to write to external storage.");
            }
        } else {
            Log.i(TAG, "Created directory " + path);
        }
    }

    private void copyTessDataFiles(String path) {
        try {
            String fileList[] = context.getAssets().list(path);

            for (String fileName : fileList) {
                String pathToDataFile = TESSERACT_PATH + path + "/" + fileName;
                if (!(new File(pathToDataFile)).exists()) {
                    InputStream in = context.getAssets().open(path + "/" + fileName);
                    OutputStream out = new FileOutputStream(pathToDataFile);
                    byte[] buf = new byte[1024];
                    int length;
                    while ((length = in.read(buf)) > 0) {
                        out.write(buf, 0, length);
                    }
                    in.close();
                    out.close();
                    Log.d(TAG, "Copied " + fileName + "to tessdata");
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "Unable to copy files to tessdata " + e.getMessage());
        }
    }

    public TessOCR(Context context) {
        this.context = context;


        prepareTessData();
        //datapath = "/mnt/sdcard";
        mTess.setDebug(true);
//        mTess.setVariable(TessBaseAPI.VAR_CHAR_WHITELIST, "1234567890!@#$%^&*()_+=-qwertyuiop[]}{POIU" +
        //        "YTREWQasdASDfghFGHjklJKLl;L:'\"\\|~`xcvXCVbnmBNM,./<>?");

        //mTess.setVariable(TessBaseAPI.VAR_CHAR_WHITELIST, "0123456789qwertyuiopasdfghjklçzxcvbnm" +
        //        "QWERTYUIOPASDFGHJKLÇZXCVBNMáéíóúÁÉÍÓÚ!?:");


        mTess.init(TESSERACT_PATH, "por");
        mTess.setPageSegMode(PSM_AUTO);

    }

    public List<Rect> getTextRegions1(Mat img) {
        mTess.setImage(Bitmap.createBitmap(img.cols(), img.rows(), Bitmap.Config.ARGB_8888));
        Pixa regions = mTess.getWords();
        /*
        final ResultIterator iterator = mTess.getResultIterator();

        float lastConfidence;
        int count = 0;
        String utf8;
        iterator.begin();
        do{
            utf8 = iterator.getUTF8Text(TessBaseAPI.PageIteratorLevel.RIL_WORD);
            lastConfidence = iterator.confidence(TessBaseAPI.PageIteratorLevel.RIL_WORD);

            count++;
            //lineRects.add(iterator.getBoundingRect(TessBaseAPI.PageIteratorLevel.RIL_WORD));
        }while(iterator.next(TessBaseAPI.PageIteratorLevel.RIL_WORD));
        //System.out.println(count);


        */


        //System.out.println(lineRects.size());

        regions.recycle();
        return null;
    }

    public List<Rect> getTextRegions(Bitmap bmp) {
        mTess.setImage(bmp);
        Pixa regions = mTess.getWords();

        List<Rect> lineRects = new ArrayList<>(regions.getBoxRects());
        Log.i("Text Regions ---->", String.valueOf(lineRects.size()));
        regions.recycle();
        return lineRects;
    }

    public int getTextAreas(Bitmap bmp) {
        mTess.setImage(bmp);
        Pixa regions = mTess.getWords();

        List<Rect> lineRects = new ArrayList<>(regions.getBoxRects());
        int s = regions.size();
        Log.i("Regions Size ---->", String.valueOf(regions.size()));

        regions.recycle();

        return s;
    }

    public void stopRecognition() {
        mTess.stop();
    }

    public String getOCRResult(Bitmap bitmap) {

        //mTess.setVariable(TessBaseAPI.VAR_CHAR_WHITELIST, "1234567890");

        mTess.setImage(bitmap);

        //String result = mTess.getUTF8Text();
        String result = mTess.getUTF8Text();
        return result;
    }

    public void onDestroy() {
        if (mTess != null)
            mTess.end();
    }

}
