package com.example.appwithtflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import com.example.appwithtflite.ml.PlantDiseaseModelKamekTerbaru
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.model.Model
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions
import org.tensorflow.lite.task.vision.classifier.Classifications
import org.tensorflow.lite.task.vision.classifier.ImageClassifier
import org.tensorflow.lite.task.vision.classifier.ImageClassifier.ImageClassifierOptions
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ImageClassifierHelper(
    private val maxResult:Int = 2,
    private val numThreads:Int = 4,
    private val modelName:String = "plant_disease_model_Kamek_terbaru.tflite",
    private val context: Context,
    private var imageClassifierListener:ClassifierListener?
) {

    private var imageClassifier:ImageClassifier?=null

    init {
        setUpImageClassifier()
    }

    private fun setUpImageClassifier(){
        val optionsBuilder = ImageClassifier.ImageClassifierOptions.builder()
            .setScoreThreshold(0.1f)
            .setMaxResults(maxResult)

        val baseOptionsBuilder = BaseOptions.builder()
            .setNumThreads(numThreads)

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        try {
            imageClassifier = ImageClassifier.createFromFileAndOptions(
                context,
                modelName,
                optionsBuilder.build()
            )
        }catch (e:Exception){
            e.printStackTrace()
            imageClassifierListener?.onError(e.message.toString())
        }
    }

    /*fun classify(imageUri: Uri):String {
        try {
            val image = imageUriToBitmap(imageUri)
            val model: PlantDiseaseModel = PlantDiseaseModel.newInstance(context)

            val imageSize = 128

            // Creates inputs for reference.
            val inputFeature0 =
                TensorBuffer.createFixedSize(intArrayOf(1, 32, 32, 3), DataType.FLOAT32)
            val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
            byteBuffer.order(ByteOrder.nativeOrder())

            val intValues = IntArray(imageSize * imageSize)
            image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
            println(intValues)
            var pixel = 0
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for (i in 0 until imageSize) {
                for (j in 0 until imageSize) {
                    val `val` = intValues[pixel++] // RGB
                    byteBuffer.putFloat(((`val` shr 16) and 0xFF) * (1f / 1))
                    byteBuffer.putFloat(((`val` shr 8) and 0xFF) * (1f / 1))
                    byteBuffer.putFloat((`val` and 0xFF) * (1f / 1))
                }
            }

            inputFeature0.loadBuffer(byteBuffer)

            // Runs model inference and gets result.
            val outputs: PlantDiseaseModel.Outputs = model.process(inputFeature0)
            val outputFeature0: TensorBuffer = outputs.getOutputFeature0AsTensorBuffer()

            val confidences = outputFeature0.floatArray
            // find the index of the class with the biggest confidence.
            var maxPos = 0
            var maxConfidence = 0f
            for (i in confidences.indices) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i]
                    maxPos = i
                }
            }
            val classes =  listOf("Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus",
            "Tomato___healthy")
            // Releases model resources if no longer used.
            model.close()

            return classes[maxPos]
        } catch (e: IOException) {
            return e.message.toString()
        }
    }
*/

    fun classify(imageUri: Uri){

        if (imageClassifier == null){
            setUpImageClassifier()
        }

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(300, 300, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(CastOp(DataType.FLOAT32))
            .build()

        val tensorImage = imageProcessor.process(
            TensorImage.fromBitmap(
                imageUriToBitmap(
                    imageUri
                )
            )
        )

        val imageProcessingOptions = ImageProcessingOptions.builder()
            .build()

        val results = imageClassifier?.classify(tensorImage, imageProcessingOptions)
        println(results)
        imageClassifierListener?.onResult(results)
    }

    private fun imageUriToBitmap(imageUri: Uri):Bitmap{
        return if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.P){
            val source = ImageDecoder.createSource(context.contentResolver, imageUri)
            ImageDecoder.decodeBitmap(source)
        }else{
            MediaStore.Images.Media.getBitmap(context.contentResolver, imageUri)
        }.copy(Bitmap.Config.ARGB_8888, true)
    }

    private fun resizeBitmap(bitmap: Bitmap): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, 300, 300, false)
    }

    /*private fun getMax(arr:FloatArray):Int{
        var max = 0
        (arr.indices).forEach{
            if(arr[it] > arr[max]){
                max = it
            }
        }
        return max
    }*/

    interface ClassifierListener{
        fun onError(error:String)

        fun onResult(
            results:List<Classifications>?
        )
    }
}