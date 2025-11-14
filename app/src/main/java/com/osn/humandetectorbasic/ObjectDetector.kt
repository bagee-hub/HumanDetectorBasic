package com.osn.humandetectorbasic

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer

class ObjectDetectorSample(context: Context, modelPath: String) {

    private var interpreter: Interpreter
    private val imageProcessor: ImageProcessor
    private val tensorImage: TensorImage

    init {
        val model: ByteBuffer = try {
            FileUtil.loadMappedFile(context, modelPath)
        } catch (e: IOException) {
            Log.e("ObjectDetectorSample", "Error loading model", e)
            throw e
        }
        interpreter = Interpreter(model, Interpreter.Options())

        val inputTensor = interpreter.getInputTensor(0)
        val modelInputWidth = inputTensor.shape()[1]
        val modelInputHeight = inputTensor.shape()[2]
        tensorImage = TensorImage(inputTensor.dataType())

        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(modelInputHeight, modelInputWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 1f))
            .build()
    }

    fun detect(bitmap: Bitmap): Boolean {
        tensorImage.load(bitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        val outputTensor = interpreter.getOutputTensor(0)
        val outputBuffer = TensorBuffer.createFixedSize(outputTensor.shape(), outputTensor.dataType())

        interpreter.run(imageBuffer, outputBuffer.buffer.rewind())

        return postProcess(outputBuffer)
    }

    private fun postProcess(outputBuffer: TensorBuffer): Boolean {
        val outputArray = outputBuffer.floatArray
        val shape: IntArray = outputBuffer.shape
        
        // YOLO11 output shape is [1, 84, 8400] where:
        // - 84 = 4 (bbox: x, y, w, h) + 80 (class scores for COCO dataset)
        // - 8400 = number of predictions
        val numFeatures = shape[1] // 84
        val numPredictions = shape[2] // 8400
        
        val confidenceThreshold = 0.35f // Lowered threshold for better detection
        val personClassIndex = 0 // 'person' is class 0 in COCO dataset
        var humanDetected = false
        var maxPersonConfidence = 0f
        
        Log.d("ObjectDetectorSample", "Output shape: [${shape[0]}, ${shape[1]}, ${shape[2]}]")
        Log.d("ObjectDetectorSample", "Processing $numPredictions predictions with $numFeatures features each")

        // YOLO11 format: data is organized as [batch, features, predictions]
        // For each prediction, we need to access: [x, y, w, h, class0, class1, ..., class79]
        for (i in 0 until numPredictions) {
            // Get bounding box coordinates (first 4 features)
            val x = outputArray[0 * numPredictions + i]
            val y = outputArray[1 * numPredictions + i]
            val w = outputArray[2 * numPredictions + i]
            val h = outputArray[3 * numPredictions + i]
            
            // Get class scores (features 4 to 83 = 80 classes)
            var maxClassConfidence = 0f
            var bestClassIndex = -1
            
            for (classIdx in 0 until 80) {
                val confidence = outputArray[(4 + classIdx) * numPredictions + i]
                if (confidence > maxClassConfidence) {
                    maxClassConfidence = confidence
                    bestClassIndex = classIdx
                }
            }
            
            // Track max person confidence for debugging
            val personConfidence = outputArray[(4 + personClassIndex) * numPredictions + i]
            if (personConfidence > maxPersonConfidence) {
                maxPersonConfidence = personConfidence
            }
            
            // Check if the best class is 'person' and confidence is above threshold
            if (bestClassIndex == personClassIndex && maxClassConfidence >= confidenceThreshold) {
                humanDetected = true
                Log.d("ObjectDetectorSample", "Human found at prediction $i with confidence: $maxClassConfidence")
                Log.d("ObjectDetectorSample", "BBox: x=$x, y=$y, w=$w, h=$h")
                break // Exit early since we only care if at least one human is present
            }
        }

        Log.d("ObjectDetectorSample", "Max person confidence found: $maxPersonConfidence (threshold: $confidenceThreshold)")
        
        // Additional debugging: show if we're close to the threshold
        if (!humanDetected && maxPersonConfidence > 0.2f) {
            Log.d("ObjectDetectorSample", "Close call: Person confidence $maxPersonConfidence is below threshold $confidenceThreshold")
        }
        
        if (humanDetected) {
            Log.d("ObjectDetectorSample", "Result: Human DETECTED ✓")
        } else {
            Log.d("ObjectDetectorSample", "Result: No human detected ✗")
        }
        
        return humanDetected
    }
}
