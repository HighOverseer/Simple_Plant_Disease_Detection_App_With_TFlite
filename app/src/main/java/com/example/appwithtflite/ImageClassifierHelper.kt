package com.example.appwithtflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.SystemClock
import android.provider.MediaStore
import com.example.appwithtflite.ml.AutoModel1
import com.example.appwithtflite.ml.PlantDiseaseModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions
import org.tensorflow.lite.task.vision.classifier.Classifications
import org.tensorflow.lite.task.vision.classifier.ImageClassifier
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ImageClassifierHelper(
    private val maxResult:Int = 2,
    private val numThreads:Int = 4,
    private val modelName:String = "1.tflite",
    private val context: Context,
    private var imageClassifierListener:ClassifierListener?
) {

    private var imageClassifier:ImageClassifier?=null

    init {
        //setUpImageClassifier()
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


    fun classify(imageUri: Uri){
        /*//GymGuide Project Reference
        val model = PlantDiseaseModel.newInstance(context)
        // Creates inputs for reference.
        val imageSize = 128

        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(1, imageSize, imageSize, 3), DataType.FLOAT32)
        val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(imageSize * imageSize)

        val image = imageUriToBitmap2(imageUri) ?: throw Exception("error")

        image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
        var pixel = 0
        //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
        for (i in 0 until imageSize) {
            for (j in 0 until imageSize) {
                val `val` = intValues[pixel++] // RGB
                byteBuffer.putFloat((`val` shr 16 and 0xFF) * (1f / 255f))  // Normalize to [0, 1]
                byteBuffer.putFloat((`val` shr 8 and 0xFF) * (1f / 255f))
                byteBuffer.putFloat((`val` and 0xFF) * (1f / 255f))
            }
        }
        inputFeature0.loadBuffer(byteBuffer)

        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        println("outputs : $outputs")
        println("outputFeature0 : $outputFeature0")

        val classes =  listOf(
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus",
            "Tomato___healthy")

        val sortedResult = outputFeature0.floatArray.sortedDescending()
        val classNames = sortedResult.let {
            val outputNames = mutableListOf<String>()

            for(result in sortedResult){
                val index = outputFeature0.floatArray.indexOfFirst { it == result }
                if(index == -1) continue

                outputNames.add(classes[index])
            }

            outputNames
        }

        imageClassifierListener?.onResult(classNames)*/

        //Sample Code Model Reference
        val model = PlantDiseaseModel.newInstance(context)

        val bitmap = imageUriToBitmap(imageUri)

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(128, 128, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f)) // Normalize pixel values
            .build()

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        val processedImage = imageProcessor.process(tensorImage)

        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 128, 128, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(processedImage.buffer)

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        println("outputs : $outputs")
        println("outputFeature0 : $outputFeature0")

        val classes =  listOf(
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus",
            "Tomato___healthy")

        val sortedResult = outputFeature0.floatArray.sortedDescending()
        val classNames = sortedResult.let {
            val outputNames = mutableListOf<String>()

            for(result in sortedResult){
                val index = outputFeature0.floatArray.indexOfFirst { it == result }
                if(index == -1) continue

                outputNames.add(classes[index])
            }

            outputNames
        }

        imageClassifierListener?.onResult(classNames)

        // Releases model resources if no longer used.
        model.close()
    }

    //kode untuk membandingkan cara 1 dengan cara 2
    /*fun classify(imageUri: Uri){
        //cara 1
        val model = AutoModel1.newInstance(context)

        val bitmap1 = imageUriToBitmap2(imageUri, imageSize = 224)
        val image = TensorImage.fromBitmap(bitmap1)


        val outputs = model.process(image)
        val probability = outputs.probabilityAsCategoryList
        probability.sortByDescending { it.score }
        probability.slice(0..5).forEach {
            println("class : ${it.label} , score : ${it.score}")
        }


        model.close()


        //cara 2
        if (imageClassifier == null) {
            setUpImageClassifier()
        }

        val bitmap2 = imageUriToBitmap(imageUri)

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(CastOp(DataType.UINT8))
            .build()

        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap2))

        val imageProcessingOptions = ImageProcessingOptions.builder()
            .build()

        val results = imageClassifier?.classify(tensorImage, imageProcessingOptions) ?: emptyList()
        imageClassifierListener?.onResult(
            results
        )
    }
*/
    private fun imageUriToBitmap(imageUri: Uri):Bitmap{
        return if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.P){
            val source = ImageDecoder.createSource(context.contentResolver, imageUri)
            ImageDecoder.decodeBitmap(source)
        }else{
            MediaStore.Images.Media.getBitmap(context.contentResolver, imageUri)
        }.copy(Bitmap.Config.ARGB_8888, true)
    }

    private fun imageUriToBitmap2(imageUri: Uri, imageSize:Int = 128):Bitmap?{
        return if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.P){
            val source = ImageDecoder.createSource(context.contentResolver, imageUri)
            var bitmap = ImageDecoder.decodeBitmap(source){ decoder, _, _ ->
                decoder.setTargetSampleSize(1) // shrinking by
                decoder.isMutableRequired =
                    true
            }
            bitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, false)
            bitmap
        }else{
            null
        }
    }

    private fun resizeBitmap(bitmap: Bitmap): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, 300, 300, false)
    }

    private fun imageUriToByteBuffer(context: Context, imageUri: Uri): ByteBuffer? {
        return try {
            val contentResolver = context.contentResolver
            val inputStream = contentResolver.openInputStream(imageUri)

            val byteBuffer = ByteBuffer.allocate(inputStream!!.available())
            byteBuffer.order(ByteOrder.nativeOrder())
            inputStream.read(byteBuffer.array())
            inputStream.close()

            byteBuffer
        } catch (e: IOException) {
            null // Handle errors appropriately
        }
    }

    interface ClassifierListener{
        fun onError(error:String)

        fun onResult(
            outputClassNames:List<String>
        )
    }
}