package com.osn.humandetectorbasic

import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
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
        imagePaths = loadImagesFromAssets("test_images")
        
        if (imagePaths.isEmpty()) {
            resultTextView.text = "No images found in test_images folder"
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

    private fun loadImagesFromAssets(folderPath: String): List<String> {
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
        nextButton.text = "Next (${currentImageIndex + 1}/${imagePaths.size})"
    }

    private fun processImage() {
        val currentImagePath = imagePaths[currentImageIndex]
        val imageName = currentImagePath.substringAfterLast("/")
        
        try {
            val bitmap = assets.open(currentImagePath).use { BitmapFactory.decodeStream(it) }
            imageView.setImageBitmap(bitmap)
            
            val humanDetected = objectDetector.detect(bitmap)
            
            val resultText = if (humanDetected) {
                "✓ Human DETECTED\n($imageName)"
            } else {
                "✗ No Human Detected\n($imageName)"
            }
            
            if (humanDetected) {
                resultTextView.text = resultText
                resultTextView.setTextColor(Color.parseColor("#4CAF50")) // Green
                resultTextView.setBackgroundColor(Color.parseColor("#E8F5E9")) // Light green
            } else {
                resultTextView.text = resultText
                resultTextView.setTextColor(Color.parseColor("#F44336")) // Red
                resultTextView.setBackgroundColor(Color.parseColor("#FFEBEE")) // Light red
            }
            
            updateButtonText()
            
        } catch (e: IOException) {
            Log.e("MainActivity", "Error loading image: $currentImagePath", e)
            resultTextView.text = "Error loading image: $imageName"
        }
    }
}
