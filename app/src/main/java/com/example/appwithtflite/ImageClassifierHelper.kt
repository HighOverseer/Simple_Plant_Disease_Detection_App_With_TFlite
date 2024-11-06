package com.example.appwithtflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import com.example.appwithtflite.ml.ModelLocal
import com.google.flatbuffers.FlatBufferBuilder
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.schema.ExpandDimsOptions
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.support.metadata.MetadataExtractor
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.classifier.ImageClassifier
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer

class ImageClassifierHelper(
    private val maxResult: Int = 2,
    private val numThreads: Int = 4,
    private val modelName: String = "1.tflite",
    private val context: Context,
    private var imageClassifierListener: ClassifierListener?
) {

    private var imageClassifier: ImageClassifier? = null



    private fun setUpImageClassifier() {
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
        } catch (e: Exception) {
            e.printStackTrace()
            imageClassifierListener?.onError(e.message.toString())
        }
    }

    fun classify(imageUri: Uri) {

        val model = ModelLocal.newInstance(context)

        val image = imageUriToBitmap2(imageUri, 255)

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(255, 255, ResizeOp.ResizeMethod.BILINEAR))
            //.add(NormalizeOp(0f, 255f))
            .build()


        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(image)
        val processedImage = imageProcessor.process(tensorImage)

        val repeatedImageBuffer = ByteBuffer.allocateDirect(32 * 255 * 255 * 3 * 4)
        repeatedImageBuffer.order(ByteOrder.nativeOrder())

        repeat(32){
            repeatedImageBuffer.put(processedImage.buffer)
        }


        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(32, 255, 255, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(repeatedImageBuffer, intArrayOf(32, 255, 255, 3))


        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        /*
        //urutan dari label.txt
        val clasess = listOf(
            "black pod rot",
            "Hama helopelthis sp.",
            "Healty Cacao"
        )
        */

        // urutan yang benar??
        val clasess = listOf(
            "Hama helopelthis sp.",
            "Healty Cacao",
            "black pod rot"
        )

        val sortedResult = outputFeature0
            .floatArray
            .slice(0..2)
            .sortedDescending()
        val classNames = sortedResult.let {
            val outputNames = mutableListOf<String>()

            for (result in sortedResult){
                val index = outputFeature0.floatArray.indexOfFirst { it == result }

                if(index == -1) continue

                outputNames.add(clasess[index])
            }

            outputNames
        }

        imageClassifierListener?.onResult(classNames)

// Releases model resources if no longer used.
        model.close()


        /*  val resizedBitmap = Bitmap.createScaledBitmap(imageUriToBitmap(imageUri), tensorWidth, tensorHeight, false)

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
          println("boxes : $bestBoxes")*/

    }

    private fun imageUriToBitmap(imageUri: Uri): Bitmap {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(context.contentResolver, imageUri)
            ImageDecoder.decodeBitmap(source)
        } else {
            MediaStore.Images.Media.getBitmap(context.contentResolver, imageUri)
        }.copy(Bitmap.Config.ARGB_8888, true)
    }

    private fun imageUriToBitmap2(imageUri: Uri, imageSize: Int = 128): Bitmap? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(context.contentResolver, imageUri)
            var bitmap = ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.setTargetSampleSize(1) // shrinking by
                decoder.isMutableRequired =
                    true
            }
            bitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, false)
            bitmap
        } else {
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
            val metadata =
                inputStream?.bufferedReader()?.use { it.readText() } ?: return emptyList()

            val regex = Regex("'names': \\{(.*?)\\}", RegexOption.DOT_MATCHES_ALL)

            val match = regex.find(metadata)
            val namesContent = match?.groups?.get(1)?.value ?: return emptyList()

            val regex2 = Regex("\"([^\"]*)\"|'([^']*)'")
            val match2 = regex2.findAll(namesContent)
            val list = match2.map { it.groupValues[1].ifEmpty { it.groupValues[2] } }.toList()

            return list
        } catch (_: Exception) {
            return emptyList()
        }
    }

    interface ClassifierListener {
        fun onError(error: String)

        fun onResult(
            outputClassNames: List<String>
        )
    }

    companion object {
        private const val LABEL_PATH = "labels.txt"
    }
}