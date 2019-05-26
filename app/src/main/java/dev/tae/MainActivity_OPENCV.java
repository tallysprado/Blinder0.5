package dev.tae;

import android.Manifest;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.os.Vibrator;
import android.speech.tts.TextToSpeech;
import android.support.annotation.NonNull;
import android.support.annotation.RequiresApi;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.util.Pair;
import android.util.SparseArray;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.text.TextBlock;
import com.google.android.gms.vision.text.TextRecognizer;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;

import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;


public class MainActivity_OPENCV extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    static {
        System.loadLibrary("opencv_java3");
    }

    ImageView imageView, imageView2;
    CameraBridgeViewBase surfaceView;

    EditText editTextArea;

    Mat rgb;
    Mat gray;

    Bitmap bmp;

    TextToSpeech tts;
    TextView textView;
    Vibrator v;

    //image holder
    Mat bwIMG, hsvIMG, lrrIMG, urrIMG, dsIMG, usIMG, cIMG, hovIMG;
    MatOfPoint2f approxCurve;


    private void initVar() {
        bwIMG = new Mat();
        dsIMG = new Mat();
        hsvIMG = new Mat();
        lrrIMG = new Mat();
        urrIMG = new Mat();
        usIMG = new Mat();
        cIMG = new Mat();
        hovIMG = new Mat();
        approxCurve = new MatOfPoint2f();



    }

    BaseLoaderCallback mBaseLoaderCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case BaseLoaderCallback.SUCCESS:
                    surfaceView.setMaxFrameSize(1280, 1024);

                    initVar();

                    surfaceView.enableView();

                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
            super.onManagerConnected(status);

        }
    };

    private static double angle(Point pt1, Point pt2, Point pt0) {
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return (dx1 * dx2 + dy1 * dy2) / Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
    }

    private void setLabel(Mat im, String label, MatOfPoint contour) {
        int fontface = Core.FONT_HERSHEY_SIMPLEX;
        double scale = 3;//0.4;,
        int thickness = 3;//1;
        int[] baseline = new int[1];
        Size text = Imgproc.getTextSize(label, fontface, scale, thickness, baseline);
        Rect r = Imgproc.boundingRect(contour);
        Point pt = new Point(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
        Imgproc.putText(im, label, pt, fontface, scale, new Scalar(255, 0, 0), thickness);
    }

    private void findRectangle(Mat src) {
        Mat blurred = src.clone();
        Imgproc.medianBlur(src, blurred, 9);

        Mat gray0 = new Mat(blurred.size(), CvType.CV_8U), gray = new Mat();

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

        List<Mat> blurredChannel = new ArrayList<Mat>();
        blurredChannel.add(blurred);
        List<Mat> gray0Channel = new ArrayList<Mat>();
        gray0Channel.add(gray0);

        MatOfPoint2f approxCurve;

        double maxArea = 0;
        int maxId = -1;

        for (int c = 0; c < 3; c++) {
            int ch[] = {c, 0};
            Core.mixChannels(blurredChannel, gray0Channel, new MatOfInt(ch));

            int thresholdLevel = 1;
            for (int t = 0; t < thresholdLevel; t++) {
                if (t == 0) {
                    Imgproc.Canny(gray0, gray, 10, 20, 3, true); // true ?
                    Imgproc.dilate(gray, gray, new Mat(), new Point(-1, -1), 1); // 1
                    // ?
                } else {
                    Imgproc.adaptiveThreshold(gray0, gray, thresholdLevel,
                            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                            Imgproc.THRESH_BINARY,
                            (src.width() + src.height()) / 200, t);
                }

                Imgproc.findContours(gray, contours, new Mat(),
                        Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

                for (MatOfPoint contour : contours) {
                    MatOfPoint2f temp = new MatOfPoint2f(contour.toArray());

                    double area = Imgproc.contourArea(contour);
                    approxCurve = new MatOfPoint2f();
                    Imgproc.approxPolyDP(temp, approxCurve,
                            Imgproc.arcLength(temp, true) * 0.02, true);

                    if (approxCurve.total() == 4 && area >= maxArea) {
                        double maxCosine = 0;

                        List<Point> curves = approxCurve.toList();
                        for (int j = 2; j < 5; j++) {

                            double cosine = Math.abs(angle(curves.get(j % 4),
                                    curves.get(j - 2), curves.get(j - 1)));
                            maxCosine = Math.max(maxCosine, cosine);
                        }

                        if (maxCosine < 0.3) {
                            maxArea = area;
                            maxId = contours.indexOf(contour);
                        }
                    }
                }
            }
        }

        if (maxId >= 0) {
            Imgproc.drawContours(src, contours, maxId, new Scalar(255, 0, 0,
                    .8), 8);

        }
    }

    private Bitmap getBmp(Mat mat) {
        Mat tmp = new Mat(mat.height(), mat.width(), CvType.CV_8U, new Scalar(4));

        try {
            Imgproc.cvtColor(mat, tmp, Imgproc.COLOR_RGBA2RGB, 4);
            bmp = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
        } catch (Exception e) {
            e.printStackTrace();
        }
        Utils.matToBitmap(tmp, bmp);

        return bmp;
    }


    @Override
    public void onCameraViewStarted(int width, int height) {
        rgb = new Mat(height, width, CvType.CV_8UC4);
        gray = new Mat(height, width, CvType.CV_8UC4);

    }

    @Override
    public void onCameraViewStopped() {
        rgb.release();
        gray.release();
    }


    private Pair<String, Integer> detection2(Mat rect) {
        int size = 0;

        bmp = getBmp(rect);

        Frame frame = new Frame.Builder().setBitmap(bmp).build();

        SparseArray items = textRecognizer.detect(frame);


        StringBuilder stringBuilder = new StringBuilder();

        String str = null;

        for (int i = 0; i < items.size(); i++) {
            TextBlock item = (TextBlock) items.valueAt(i);
            size = items.size();
            stringBuilder.append(item.getValue());
            stringBuilder.append("\n");
            //textView.setText(stringBuilder.toString());
        }
        Log.i("DETECTION2", stringBuilder.toString());
        return new Pair<String, Integer>(stringBuilder.toString(), size);
    }

    private ArrayList<String> carregaPalavras(){
        BufferedReader reader = null;
        String palavras = null;
        ArrayList<String> palavrasList = new ArrayList<String>();
        try {
            reader = new BufferedReader(
                    new InputStreamReader(getAssets().open("palavras.txt")));

            // do reading, usually loop until end of file reading

            while ((palavras = reader.readLine()) != null) {
                palavrasList.add(palavras);
            }
        } catch (IOException e) {
            //log the exception
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }
        return palavrasList;
    }
    private String corrige(String text){
        String[] splited = text.split("\\s+");
        BufferedReader reader = null;
        String palavras = null;
        try {
            reader = new BufferedReader(
                    new InputStreamReader(getAssets().open("palavras.txt")));

            // do reading, usually loop until end of file reading

            while ((palavras = reader.readLine()) != null) {
                for(int i=0; i<=splited.length;i++){
                    if(palavras.contains(splited[i])) return splited[i] = palavras;
                }
            }
        } catch (IOException e) {
            //log the exception
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }
        return splited.toString();
    }

    Rect rectR;
    double minCos = -0.1, maxCos = 0.2;
    int maxSize;
    int minSize;
    double area;
    int threshold, threshold1;
    boolean rectFlag;
    int maxContour = 500;

    private Pair<Mat,Mat> detectRect(Mat dst, Mat gray){

        Imgproc.pyrDown(gray, dsIMG, new Size(gray.cols() / 2, gray.rows() / 2));
        Imgproc.pyrUp(dsIMG, usIMG, gray.size());

        Imgproc.Canny(usIMG, bwIMG, threshold1, threshold);

        Imgproc.dilate(bwIMG, bwIMG, new Mat(), new Point(-1, 1), 1);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

        cIMG = bwIMG.clone();

        Imgproc.findContours(cIMG, contours, hovIMG, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);


        for (MatOfPoint cnt : contours) {

            MatOfPoint2f curve = new MatOfPoint2f(cnt.toArray());

            Imgproc.approxPolyDP(curve, approxCurve, 0.02 * Imgproc.arcLength(curve, true), true);

            int numberVertices = (int) approxCurve.total();

            double contourArea = Imgproc.contourArea(cnt);
            area = Math.abs(contourArea);

            Log.i("CONTOURAREA",String.valueOf(area));

            if (area<500) {
                continue;
            }

            if (numberVertices >= 4 && numberVertices <= 6) {

                List<Double> cos = new ArrayList<>();

                for (int j = 2; j < numberVertices + 1; j++) {
                    cos.add(angle(approxCurve.toArray()[j % numberVertices], approxCurve.toArray()[j - 2], approxCurve.toArray()[j - 1]));
                }

                Collections.sort(cos);

                double mincos = cos.get(0);
                double maxcos = cos.get(cos.size() - 1);

                Log.i("MINCOS", String.valueOf(mincos));
                Log.i("MAXCOS", String.valueOf(maxcos));


                if (numberVertices == 4 && mincos >= -0.1 && maxcos <= 0.3) {
                    rectR = Imgproc.boundingRect(cnt);

                    if(rectR.width>1.5*rectR.height) rectFlag = true;
                    else rectFlag = false;
                    minCos = mincos;
                    maxCos = maxcos;

                    //setLabel(dst, "X", cnt);


                }

            }
        }
        return new Pair<Mat,Mat>(dst.submat(rectR),cIMG);
    }


    Size size = new Size(10, 10);

    Pair<String, Integer> txt = null;
    int c = 8, blockSize = 5;
    String line = null;
    Mat gaussianBlur = new Mat(),mRgba = new Mat(), rect = new Mat(), grayRect = new Mat(), adaptiveThreshold = new Mat();
    Pair pairRect;

    String result;

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        rgb = inputFrame.rgba();
        gray = inputFrame.gray();
        try {
            if(flag==true) {
                rect = rgb.clone();
            }
            if(flag==false) {
                try{
                    pairRect = detectRect(rgb,gray);
                    rect = (Mat) pairRect.first;
                }catch (Exception e){
                    e.printStackTrace();
                }
            }

            //rect = detectRect(rgb);
            //detectText();
            //Log.i("VOLUMEDOWN",String.valueOf(flag));
            //rect = rgb.clone();

            Imgproc.cvtColor(rect,grayRect,Imgproc.COLOR_RGBA2GRAY);

            //final Mat resized1 = new Mat();
            //size.height = rect.height() * 4;
            //size.width = rect.width() * 4;
            //Imgproc.resize(grayResized, resized1, size);

            Imgproc.GaussianBlur(grayRect,gaussianBlur, new Size(3,3), 0);
            //Imgproc.cvtColor(gaussianBlur,grayResized,Imgproc.COLOR_RGBA2GRAY);
            Imgproc.adaptiveThreshold(gaussianBlur,adaptiveThreshold,255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY,blockSize,c);//5 e 8 ou 5 e 6
            runOnUiThread(new Runnable() {
                public void run() {

                    try {

                        Imgproc.cvtColor(adaptiveThreshold, mRgba,Imgproc.COLOR_GRAY2RGBA);
                        txt = detection2(rect);


                        result = "";
                        try {

                            String[] lines = txt.first.split("\\r?\\n|\\r");

                            for(int i=0; i<=lines.length;i++){
                                if(digits(lines[i])<=5){
                                    result = result + lines[i] +"\n";
                                }

                            }


                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                        Log.i("RECTAREA", String.valueOf(rectR.area()+"\nMINAREA: "+minArea));
                        if(txt.first!=null && rectR.area()>=minArea) {
                            imageView.setImageBitmap(getBmp(rect));
                            line = result;
                            //textView.setText(result+"\nAREA DE CONTORNO: "+area+"\nAREA DE RETANGULO: "+rectR.area()+
                            //        "\nMIN COS: "+minCos+"\nMAX COS: "+maxCos);
                            textView.setText(result);
                        }

                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }

            });
            return rgb;

        } catch (Exception e) {
            e.printStackTrace();
        }
        return rgb;
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            Log.i("OPENCV", "OpenCV carregado com sucesso.");
            mBaseLoaderCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);

        } else {
            Log.i("OPENCV", "Erro ao carregar OpenCV.");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mBaseLoaderCallBack);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (surfaceView != null) {
            surfaceView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (surfaceView != null) {
            surfaceView.disableView();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        switch (requestCode) {
            case 1001: {
                if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                        return;
                    }
                }
            }
            break;
        }
    }

    TextRecognizer textRecognizer;

    Util utils;
    Frame frame = null;
    SparseArray items = null;

    private int detection(Mat rect) {
        TextBlock it = null;
        Bitmap bmp = getBmp(rect);
        StringBuilder stringBuilder = null;

        try {
            frame = new Frame.Builder().setBitmap(bmp).build();
            items = textRecognizer.detect(frame);

            stringBuilder = new StringBuilder();
            Log.i("DETsize", String.valueOf(size));

            //Log.i("LINE p",String.valueOf("x1-> "+p[0]+" y1-> "+p[1]+" x2-> "+p[2]+" y2-> "+p[3]));
            for (int i = 0; i < items.size(); i++) {


                TextBlock item = (TextBlock) items.valueAt(i);
                stringBuilder.append(item.getValue());
                stringBuilder.append("\n");
                if (digits(item.getValue()) <= 5) {
                    textView.setText(item.getValue());
                }

                return digits(item.getValue());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        //tts.speak(stringBuilder.toString(),TextToSpeech.QUEUE_FLUSH,null);
        //textView.setText(stringBuilder.toString()+"\n"+
        //        "REGIÕES DE TEXTO -----> "+tess.getTextRegions(bmp).size());


        //textView2.setText("TESSERACT\n"+tess.getResults(bmp));

        return 0;
    }


    public int letter(String s) {
        int count = 0;
        for (int i = 0, len = s.length(); i < len; i++) {
            if (Character.isLetter(s.charAt(i))) {
                count++;
            }
        }

        return count;
    }

    public int digits(String s) {
        int count = 0;
        for (int i = 0, len = s.length(); i < len; i++) {
            if (Character.isDigit(s.charAt(i))) {
                count++;
            }
        }

        return count;
    }

    //GOOGLE ML KIT
    /*
    FirebaseVisionTextRecognizer recognizer;
    private void runTextRecognition(Bitmap bmp) throws IOException {
        bmp2 = bmp;
        FirebaseVisionImage image = FirebaseVisionImage.fromBitmap(bmp2);

        recognizer = FirebaseVision.getInstance()
                .getOnDeviceTextRecognizer();
        if(!mov) {
            recognizer.processImage(image)
                    .addOnSuccessListener(
                            new OnSuccessListener<FirebaseVisionText>() {
                                @Override
                                public void onSuccess(FirebaseVisionText texts) {

                                    processTextRecognitionResult(texts);
                                }
                            })
                    .addOnFailureListener(
                            new OnFailureListener() {
                                @Override
                                public void onFailure(@NonNull Exception e) {
                                    // Task failed with an exception

                                    e.printStackTrace();
                                }
                            });

        }else{
            recognizer.close();
        }

    }
    ArrayList<String> dLines = new ArrayList<String>();

    List<FirebaseVisionText.Line> lines;
    String lin = "";
    private void processTextRecognitionResult(FirebaseVisionText texts) {
        List<FirebaseVisionText.TextBlock> blocks = texts.getTextBlocks();
        if (blocks.size() == 0 || mov) {
            return;
        }

        for (int i = 0; i < blocks.size(); i++) {
            lines = blocks.get(i).getLines();
            if(mov) break;
            for (int j = 0; j < lines.size(); j++) {
                if(mov) break;
                String line = lines.get(j).getText();
                Log.i("GOOGLEMLKIT",line);
                Log.i("LINELEN",String.valueOf(line.length()));
                if(letter(line)>3){
                    dLines.add(line);
                    if(dLines.size()>6){
                        dLines.clear();
                        dLines.add("t");
                    }
                    StringBuilder stringBuilder = new StringBuilder();
                    stringBuilder.append(dLines.get(dLines.size()-1));
                    stringBuilder.append("\n");
                    String txt1 = dLines.get(dLines.size()-1);
                    String txt2 = "";
                    try{
                       txt2 = dLines.get(dLines.size()-2);
                    }catch (Exception e){
                        e.printStackTrace();
                    }

                    Log.i("TXT1",txt1);
                    Log.i("TXT2",txt2);
                    if(textView.getText().toString().length()>50) textView.setText("");
                    if(textView.getFreezesText()) textView.setText("");
                    if(txt1.length()>txt2.length() || digits(txt1)>5){
                        //textView.setText(txt1);

                        textView.append(txt1+"\n");

                    }else{
                        textView.append(txt2+"\n");

                    }



                    //textView.setText(stringBuilder.toString());

                    Log.i("dLines",dLines.get(dLines.size()-1));

                }else{
                    break;
                }

                /*
                List<FirebaseVisionText.Element> elements = lines.get(j).getElements();
                for (int k = 0; k < elements.size(); k++) {

                    //textView2.setText(texts.getText());
                }

            }

        }

    }

    private int compare(String str1, String str2) {
        int n = 0;
        try {
            if (str1.length() >= str2.length()) {
                for (int i = 0; i <= str1.length(); i++) {
                    if (str1.charAt(i) == '\0' || str2.charAt(i) == '\0') break;
                    if (!(str1.charAt(i) == str2.charAt(i))) {
                        n++;

                    }
                }
            } else {
                for (int i = 0; i <= str2.length(); i++) {
                    if (str1.charAt(i) == '\0' || str2.charAt(i) == '\0') break;
                    if (!(str2.charAt(i) == str1.charAt(i))) {
                        n++;
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return n;
    }
    private static Search search = new Search();
    private static Stack<String> suggestions = new Stack<String>();
    private static String[] words;
    TessOCR2 tess;
    public void loadWords() {
        String line = null; // Temp variable for storing one line at a time
        ArrayList<String> temp = new ArrayList<String>();

        try {

            BufferedReader buffReader = new BufferedReader(
                    new InputStreamReader(getAssets().open("palavras.txt")));

            while ((line = buffReader.readLine()) != null) {
                temp.add(line.trim());
            }

            buffReader.close();
            words = new String[temp.size()];
            temp.toArray(words);

        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }
    public static String autoCorrect(String userWord) {
        int result = Search.binarySearch(words, userWord);

        // First, check for an exact match
        if (result != -1) {
            System.out.println("Correct. Congratulations, you can spell!"
                    + "\n");
        }
        // Else, check if the user's word is an anagram of a dictionary word
        else {
            for (String word : words) {
                char wordStart; // First char of word
                char userWordStart; // First char of userWord
                if (!word.isEmpty()) { // If word is NOT empty
                    wordStart = word.charAt(0);
                    userWordStart = userWord.charAt(0);

                    if (userWordStart == wordStart){ // Same starting char

                        if (containsAllChars(userWord, word)){ // Same chars
                            suggestions.push(word);
                            userWord = word;
                            return word;
                        }
                    }
                }
            }

            if (suggestions.isEmpty()) {
                System.out.println("No suggestions.\n");

            }
            else {
                System.out.print("Suggestions: ");
                while (!suggeinarySearcstions.isEmpty()) {
                    System.out.print(suggestions.pop() + " ");
                }
                System.out.println("\n");
            }
        }
        return userWord;
    }
    public static boolean containsAllChars(String strOne, String strTwo) {
        Character[] one = strToCharArray(strOne);
        Character[] two = strToCharArray(strTwo);

        sort(one);
        sort(two);

        for (int i = 0; i < one.length; i++) {
            if (Search.binarySearch(two, one[i]) == -1)
                return false;
            two[i] = '0';
        }

        two = strToCharArray(strTwo);
        sort(two);

        for (int i = 0; i < two.length; i++) {
            if (Search.binarySearch(one, two[i]) == -1)
                return false;
            one[i] = '0';
        }

        return true;
    }
    public static Character[] strToCharArray(String str) {
        Character[] charArray = new Character[str.length()];
        for (int i = 0; i < str.length(); i++) {
            charArray[i] = new Character(str.charAt(i));
        }

        return charArray;
    }
    public static <E extends Comparable<E>> void sort(E[] array) {
        int n = array.length; // Get length of array

        // Insertion sort
        for (int i = 1; i < n; i++) {
            E temp = array[i]; // Save the element at index i
            int j = i - 1; // Let j be the element one index before i

            // Iterate through array
            while (j > -1 && (array[j].compareTo(temp) > 0)) {
                // Insert element at array[j] in proper place
                array[j + 1] = array[j];
                j--;
            }

            // Complete swap
            array[j + 1] = temp;
        }
    }
    */



    boolean flag = false;
    double minArea;
    @RequiresApi(api = Build.VERSION_CODES.O)
    @Override
    protected void onCreate(Bundle savedInstance) {
        super.onCreate(savedInstance);

        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        threshold = 0;
        threshold1 = 100;
        minSize = 3000;
        maxSize = 10000;


        setContentView(R.layout.activity_main);



        textRecognizer = new TextRecognizer.Builder(getApplicationContext()).build();
        tts = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status == TextToSpeech.SUCCESS) {
                    tts.setLanguage(Locale.getDefault());
                }
            }
        });

        editTextArea = (EditText) findViewById(R.id.editTextArea);


        surfaceView = findViewById(R.id.java_camera_view);
        textView = findViewById(R.id.text_view);
        imageView = findViewById(R.id.imageView);
        imageView2 = findViewById(R.id.imageView2);




        editTextArea.setText("100");
        editTextArea.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {

            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {

            }

            @Override
            public void afterTextChanged(Editable s) {
                try {
                    minArea = Double.valueOf(editTextArea.getText().toString());
                }catch (Exception e){
                    e.printStackTrace();
                }
            }
        });


        v = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);

        surfaceView.setVisibility(View.VISIBLE);
        surfaceView.setCvCameraViewListener(this);
        surfaceView.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View v) {
                if (flag == true) flag = false;
                else flag = true;
                return false;
            }
        });

        surfaceView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
               try{
                   if(tts.isSpeaking()) tts.stop();
                   else tts.speak(result,TextToSpeech.QUEUE_FLUSH,null);
               }catch (Exception e){
                   e.printStackTrace();
               }


            }
        });

        if (ActivityCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {

            ActivityCompat.requestPermissions(MainActivity_OPENCV.this,
                    new String[]{Manifest.permission.CAMERA},
                    1001);

        }
    }



    Scalar CONTOUR_COLOR = new Scalar(255);
    List<KeyPoint> listpoint;
    KeyPoint kpoint;
    Mat mask;
    int rectanx1;
    int rectany1;
    int rectanx2;
    int rectany2;
    int imgsize;
    Scalar zeos = new Scalar(0, 0, 0);

    private void detectText () {
        MatOfKeyPoint keypoint = new MatOfKeyPoint();

        mask = Mat.zeros(gray.size(), CvType.CV_8UC1);

        imgsize = gray.height() * gray.width();


        List<MatOfPoint> contour2 = new ArrayList<MatOfPoint>();
        Mat kernel = new Mat(1, 50, CvType.CV_8UC1, Scalar.all(255));
        Mat morbyte = new Mat();
        Mat hierarchy = new Mat();

        Rect rectan3;

        FeatureDetector detector = FeatureDetector
                .create(FeatureDetector.MSER);
        detector.detect(gray, keypoint);
        listpoint = keypoint.toList();

        for (int ind = 0; ind < listpoint.size(); ind++) {
            kpoint = listpoint.get(ind);
            rectanx1 = (int) (kpoint.pt.x - 0.5 * kpoint.size);
            rectany1 = (int) (kpoint.pt.y - 0.5 * kpoint.size);
            rectanx2 = (int) (kpoint.size);
            rectany2 = (int) (kpoint.size);
            if (rectanx1 <= 0)
                rectanx1 = 1;
            if (rectany1 <= 0)
                rectany1 = 1;
            if ((rectanx1 + rectanx2) > gray.width())
                rectanx2 = gray.width() - rectanx1;
            if ((rectany1 + rectany2) > gray.height())
                rectany2 = gray.height() - rectany1;
            Rect rectant = new Rect(rectanx1, rectany1, rectanx2, rectany2);
            try {
                Mat roi = new Mat(mask, rectant);
                roi.setTo(CONTOUR_COLOR);
            } catch (Exception ex) {
                Log.d("mylog", "mat roi error " + ex.getMessage());
            }
        }

        Imgproc.morphologyEx(mask, morbyte, Imgproc.MORPH_DILATE, kernel);
        Imgproc.findContours(morbyte, contour2, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        for (int ind = 0; ind < contour2.size(); ind++) {

            rectan3 = Imgproc.boundingRect(contour2.get(ind));

            if (rectan3.area() > 0.5 * imgsize || rectan3.area() < 100
                    || rectan3.width / rectan3.height < 2) {
                Mat roi = new Mat(morbyte, rectan3);
                roi.setTo(zeos);

            } else
                //Imgproc.rectangle(rgb, rectan3.br(), rectan3.tl(),
                //        CONTOUR_COLOR);
             v.vibrate(contour2.size() * 15);


        }
    }


}



