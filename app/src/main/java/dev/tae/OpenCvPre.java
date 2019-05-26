package dev.tae;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class OpenCvPre {
    ;

    public Mat preProccess(Mat inputMat) {
        Mat mat = new Mat();
        Imgproc.cvtColor(inputMat, mat, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(mat, mat, new Size(3, 3), 0);
        //Imgproc.adaptiveThreshold(mat, mat,255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 5,7);

        return mat;
    }

    public synchronized boolean isBlurredImage(Bitmap image) {
        try {
            if (image != null) {
                BitmapFactory.Options opt = new BitmapFactory.Options();
                opt.inDither = true;
                opt.inPreferredConfig = Bitmap.Config.ARGB_8888;
                int l = CvType.CV_8UC1;
                Mat matImage = new Mat();
                Utils.bitmapToMat(image, matImage);
                Mat matImageGrey = new Mat();
                Imgproc.cvtColor(matImage, matImageGrey, Imgproc.COLOR_BGR2GRAY);

                Mat dst2 = new Mat();
                Utils.bitmapToMat(image, dst2);

                Mat laplacianImage = new Mat();
                dst2.convertTo(laplacianImage, l);
                Imgproc.Laplacian(matImageGrey, laplacianImage, CvType.CV_8U);
                Mat laplacianImage8bit = new Mat();
                laplacianImage.convertTo(laplacianImage8bit, l);
                System.gc();

                Bitmap bmp = Bitmap.createBitmap(laplacianImage8bit.cols(),
                        laplacianImage8bit.rows(), Bitmap.Config.ARGB_8888);

                Utils.matToBitmap(laplacianImage8bit, bmp);

                int[] pixels = new int[bmp.getHeight() * bmp.getWidth()];
                bmp.getPixels(pixels, 0, bmp.getWidth(), 0, 0, bmp.getWidth(),
                        bmp.getHeight());
                if (bmp != null)
                    if (!bmp.isRecycled()) {
                        bmp.recycle();

                    }
                int maxLap = -16777216;

                for (int i = 0; i < pixels.length; i++) {

                    if (pixels[i] > maxLap) {
                        maxLap = pixels[i];
                    }
                }
                int soglia = -6118750;

                return maxLap < soglia || maxLap == soglia;
            } else {
                return false;
            }
        } catch (NullPointerException e) {
            return false;
        } catch (OutOfMemoryError e) {
            return false;
        }
    }
}
