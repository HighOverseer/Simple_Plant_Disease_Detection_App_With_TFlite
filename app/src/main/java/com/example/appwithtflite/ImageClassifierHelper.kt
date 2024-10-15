package com.example.appwithtflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.media.FaceDetector.Face.CONFIDENCE_THRESHOLD
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import com.example.appwithtflite.ml.LastFloat32
import com.example.appwithtflite.ml.ModelKamekTerbaru
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.metadata.MetadataExtractor
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.classifier.ImageClassifier
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer

class ImageClassifierHelper(
    private val maxResult:Int = 2,
    private val numThreads:Int = 4,
    private val modelName:String = "1.tflite",
    private val context: Context,
    private var imageClassifierListener:ClassifierListener?
) {

    private var imageClassifier:ImageClassifier?=null
    //setUpImageClassifier()
    private val associatedAxisLabels:List<String> = FileUtil.loadLabels(context, LABEL_PATH)
    private val model = FileUtil.loadMappedFile(context, "model_kamek_terbaru.tflite")
    private val interpreter = Interpreter(model)

    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    init {
        val options = Interpreter.Options().apply{
            this.setNumThreads(4)
        }

        labels.addAll(extractNamesFromMetadata(model))
        labels.forEach {
            println("label : $it")
        }
        val inputShape = interpreter.getInputTensor(0)?.shape()
        val outputShape = interpreter.getOutputTensor(0)?.shape()

        if (inputShape != null) {
            tensorWidth = inputShape[1]
            tensorHeight = inputShape[2]

            // If in case input shape is in format of [1, 3, ..., ...]
            if (inputShape[1] == 3) {
                tensorWidth = inputShape[2]
                tensorHeight = inputShape[3]
            }
        }

        if (outputShape != null) {
            numElements = outputShape[1]
            numChannel = outputShape[2]
        }
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

        val resizedBitmap = Bitmap.createScaledBitmap(imageUriToBitmap(imageUri), tensorWidth, tensorHeight, false)

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap)

        val imageProcessor = ImageProcessor.Builder()
            .add(NormalizeOp(0f, 225f))
            .add(CastOp(DataType.FLOAT32))
            .build()

        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), DataType.FLOAT32)
        interpreter.run(imageBuffer, output.buffer)

        val bestBoxes = bestBox(output.floatArray)

        println("outputBuffer : ${output.buffer}")
        println("boxes : $bestBoxes")


    /*    val imageSize = 640
        val bitmap = imageUriToBitmap(imageUri)

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(imageSize, imageSize, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f)) // Normalize pixel values
            .build()

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val probabilityBuffer = TensorBuffer.createFixedSize(intArrayOf(1, imageSize, imageSize, 3), DataType.FLOAT32)

        interpreter.run(processedImage.buffer, probabilityBuffer.buffer)

        println(probabilityBuffer.buffer)

        val tensorProcessor = TensorProcessor.Builder()
            .add(NormalizeOp(0f, 255f))
            .build()

        val labels = TensorLabel(associatedAxisLabels, tensorProcessor.process(probabilityBuffer))
        val floatMap = labels.mapWithFloatValue
        val probability = floatMap.toList().sortedByDescending { (_, value) -> value }
        probability.forEach {
            println("class: ${it.first} score: ${it.second}")
        }*/

        //cara 1
        /*val model = ModelKamekTerbaru.newInstance(context)

        val bitmap1 = imageUriToBitmap(imageUri)
        //val image = TensorImage.fromBitmap(bitmap1)

        val imageSize = 640
        val bitmap = imageUriToBitmap(imageUri)

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(imageSize, imageSize, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f)) // Normalize pixel values
            .build()

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        val processedImage = imageProcessor.process(tensorImage)

        val outputs = model.process(processedImage)
        val tensorProcessor = TensorProcessor.Builder()
            .add(NormalizeOp(0f, 255f))
            .build()

        val labels = TensorLabel(associatedAxisLabels, tensorProcessor.process(outputs.outputAsTensorBuffer))
        val floatMap = labels.mapWithFloatValue
        val probability = floatMap.toList().sortedByDescending { (_, value) -> value }
        probability.forEach {
            println("class: ${it.first} score: ${it.second}")
        }

        //imageClassifierListener?.onResult(probability.sortedByDescending { it.score })

        model.close()*/
    }

    private fun imageUriToBitmap(imageUri: Uri):Bitmap{
        return if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.P){
            val source = ImageDecoder.createSource(context.contentResolver, imageUri)
            ImageDecoder.decodeBitmap(source)
        }else{
            MediaStore.Images.Media.getBitmap(context.contentResolver, imageUri)
        }.copy(Bitmap.Config.ARGB_8888, true)
    }

    private fun bestBox(array: FloatArray) : List<BoundingBox> {
        val boundingBoxes = mutableListOf<BoundingBox>()
        for (r in 0 until numElements) {
            val cnf = array[r * numChannel + 4]
            if (cnf > 0.1f) {
                val x1 = array[r * numChannel]
                val y1 = array[r * numChannel + 1]
                val x2 = array[r * numChannel + 2]
                val y2 = array[r * numChannel + 3]
                val cls = array[r * numChannel + 5].toInt()
                val clsName = labels[cls]
                boundingBoxes.add(
                    BoundingBox(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cnf = cnf, cls = cls, clsName = clsName
                    )
                )
            }
        }
        return boundingBoxes
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

    fun extractNamesFromMetadata(model: MappedByteBuffer): List<String> {
        try {
            val metadataExtractor = MetadataExtractor(model)
            val inputStream = metadataExtractor.getAssociatedFile("temp_meta.txt")
            val metadata = inputStream?.bufferedReader()?.use { it.readText() } ?: return emptyList()

            val regex = Regex("'names': \\{(.*?)\\}", RegexOption.DOT_MATCHES_ALL)

            val match = regex.find(metadata)
            val namesContent = match?.groups?.get(1)?.value ?: return emptyList()

            val regex2 = Regex("\"([^\"]*)\"|'([^']*)'")
            val match2 = regex2.findAll(namesContent)
            val list = match2.map { it.groupValues[1].ifEmpty { it.groupValues[2] }}.toList()

            return list
        } catch (_: Exception) {
            return emptyList()
        }
    }

    interface ClassifierListener{
        fun onError(error:String)

        fun onResult(
            outputClassNames:List<Category>
        )
    }

    companion object{
        private const val LABEL_PATH = "labels.txt"
    }
}