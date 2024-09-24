package com.example.appwithtflite

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.graphics.drawable.Drawable
import android.net.Uri
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.view.drawToBitmap
import com.bumptech.glide.load.DataSource
import com.bumptech.glide.load.engine.GlideException
import com.bumptech.glide.request.RequestListener
import com.bumptech.glide.request.target.Target
import com.example.appwithtflite.databinding.ActivityMainBinding
import org.tensorflow.lite.task.vision.classifier.Classifications
import java.text.NumberFormat


class MainActivity : AppCompatActivity(),
    ImageClassifierHelper.ClassifierListener{

    private lateinit var binding:ActivityMainBinding
    private lateinit var imageClassifierHelper:ImageClassifierHelper
    private val imageCaptureHandler = ImageCaptureHandler()

    private var imageUri:Uri? = null

    private val launcherGallery = registerForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ){ uri ->
        if (uri == null) return@registerForActivityResult
        binding.ivImage.loadImage(uri)
        imageUri = uri
    }

    private val launcherCamera = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ){ isSuccess ->
        if (!isSuccess) return@registerForActivityResult

        imageUri = imageCaptureHandler.latestImageCaptured

        binding.ivImage.loadImage(
            imageCaptureHandler.latestImageCaptured
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        imageClassifierHelper = ImageClassifierHelper(
            context = this,
            imageClassifierListener = this
        )


        binding.apply {
            btnAddCamera.setOnClickListener {
                openCamera()
            }
            btnAddGallery.setOnClickListener {
                openGallery()
            }

            btnPredict.setOnClickListener {
                predict()
            }

        }
    }

    private fun predict(){
        val uri = imageUri

        if(uri == null){
            Toast.makeText(
                this,
                getString(R.string.please_insert_the_image_file_first),
                Toast.LENGTH_SHORT
            ).show()

            return
        }

        imageClassifierHelper.classify(uri)
    }

    override fun onError(error: String) {
        runOnUiThread {
            Toast.makeText(this, error, Toast.LENGTH_SHORT).show()
        }

    }

    override fun onResult(results:List<Classifications>?) {
        runOnUiThread {
            results?.let {
                if(it.isNotEmpty() && it[0].categories.isNotEmpty()){
                    println(it)
                    val sortedCategories = it[0].categories.sortedByDescending { category ->
                        category?.score
                    }
                    val displayResult = sortedCategories.joinToString("\n") {category ->
                        "${category.label} " + NumberFormat.getPercentInstance()
                            .format(category.score).trim()
                    }
                    binding.edAddDescription.text = displayResult
                }
            }
        }

    }


    private fun openGallery(){
        launcherGallery.launch(
            PickVisualMediaRequest(
                ActivityResultContracts.PickVisualMedia.ImageOnly
            )
        )
    }

    private fun openCamera(){
        val uri = imageCaptureHandler.getImageUri(this)
        println(uri)
        launcherCamera.launch(uri)
    }
}