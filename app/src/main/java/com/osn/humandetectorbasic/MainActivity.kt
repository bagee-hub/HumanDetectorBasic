package com.osn.humandetectorbasic

import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private lateinit var objectDetector: ObjectDetectorSample
    private lateinit var imageView: ImageView
    private lateinit var nextButton: Button
    private lateinit var resultTextView: TextView

    private lateinit var imagePaths: List<String>
    private var currentImageIndex = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        nextButton = findViewById(R.id.nextButton)
        resultTextView = findViewById(R.id.resultTextView)

        // Load all images from test_images folder
        imagePaths = loadImagesFromAssets()
        
        if (imagePaths.isEmpty()) {
            resultTextView.text = getString(R.string.no_images_found)
            nextButton.isEnabled = false
            return
        }
        
        Log.d("MainActivity", "Found ${imagePaths.size} images: $imagePaths")

        objectDetector = ObjectDetectorSample(
            context = this,
            modelPath = "models/yolo11n_float32.tflite"
        )

        nextButton.setOnClickListener {
            currentImageIndex = (currentImageIndex + 1) % imagePaths.size
            processImage()
        }
        
        // Update button text to show navigation info
        updateButtonText()

        processImage()
    }

    private fun loadImagesFromAssets(): List<String> {
        val folderPath = "test_images"
        return try {
            val imageFiles = assets.list(folderPath)?.filter { fileName ->
                fileName.lowercase().endsWith(".png") || 
                fileName.lowercase().endsWith(".jpg") || 
                fileName.lowercase().endsWith(".jpeg")
            }?.sorted() ?: emptyList()
            
            imageFiles.map { "$folderPath/$it" }
        } catch (e: IOException) {
            Log.e("MainActivity", "Error loading images from assets", e)
            emptyList()
        }
    }
    
    private fun updateButtonText() {
        nextButton.text = getString(R.string.next_button_text, currentImageIndex + 1, imagePaths.size)
    }

    private fun processImage() {
        val currentImagePath = imagePaths[currentImageIndex]
        val imageName = currentImagePath.substringAfterLast("/")
        
        try {
            val bitmap = assets.open(currentImagePath).use { BitmapFactory.decodeStream(it) }
            imageView.setImageBitmap(bitmap)
            
            val humanDetected = objectDetector.detect(bitmap)
            
            val resultText = if (humanDetected) {
                getString(R.string.human_detected, imageName)
            } else {
                getString(R.string.no_human_detected, imageName)
            }
            
            if (humanDetected) {
                resultTextView.text = resultText
                resultTextView.setTextColor(ContextCompat.getColor(this, R.color.green))
                resultTextView.setBackgroundColor(ContextCompat.getColor(this, R.color.light_green))
            } else {
                resultTextView.text = resultText
                resultTextView.setTextColor(ContextCompat.getColor(this, R.color.red))
                resultTextView.setBackgroundColor(ContextCompat.getColor(this, R.color.light_red))
            }
            
            updateButtonText()
            
        } catch (e: IOException) {
            Log.e("MainActivity", "Error loading image: $currentImagePath", e)
            resultTextView.text = getString(R.string.error_loading_image, imageName)
        }
    }
}
