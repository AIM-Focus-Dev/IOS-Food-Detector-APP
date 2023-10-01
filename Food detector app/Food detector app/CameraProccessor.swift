import Foundation
import SwiftUI
import AVFoundation
import CoreML
import CoreImage
import Vision
import UIKit


class CameraProcessor: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, ObservableObject {
    
    var previewLayer: AVCaptureVideoPreviewLayer
    var captureSession: AVCaptureSession
    var model: FoodCL?
    var onPrediction: ((String) -> Void)?
    var shouldCapture: Bool = false
    
    override init() {
        captureSession = AVCaptureSession()
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        super.init()
        
        startCaptureSession() // Start the capture session
        
        // load model
        let config = MLModelConfiguration()
        do {
            self.model = try FoodCL(configuration: config)
        } catch {
            print("Failed to load the model: \(error)")
        }
    }
    
    // Add this initializer
    convenience init(currentPrediction: Binding<String>) {
        self.init()
        
        onPrediction = { newPrediction in
            DispatchQueue.main.async {
                currentPrediction.wrappedValue = newPrediction
            }
        }
    }
    
    func startCapturing() {
        shouldCapture = true
        captureSession.startRunning()
    }

    func stopCapturing() {
        shouldCapture = false
        captureSession.stopRunning()
    }
    

    
    func setPreviewLayerFrame(frame: CGRect) {
        self.previewLayer.frame = frame
    }
    
    private func startCaptureSession() {
        // Start configuration
        captureSession.beginConfiguration()
        
        // Setup the camera input
        guard let captureDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: captureDevice)
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
            }
        } catch {
            print("Error: \(error)")
        }
        
        // Setup the video output
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        
        // Commit configuration and start the session
        captureSession.commitConfiguration()
        
        // Use background queue to start running the session
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }
    // This delegate method is called whenever a new video frame is captured
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        if !shouldCapture {
             return
         }
        
        // Retrieve the captured image buffer
        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        // Preprocess the frame to get the MLMultiArray
        guard let mlMultiArray = preprocessFrame(pixelBuffer: pixelBuffer) else {
            print("Preprocessing failed.")
            return
        }
        
        do {
            if let unwrappedModel = model {
                let modelInput = FoodCLInput(x_1: mlMultiArray)
                if let modelOutput = try? unwrappedModel.prediction(input: modelInput) {
                    // Assuming var_810 is the output MultiArray
                    let outputArray = modelOutput.var_810
                    
                    // Convert MLMultiArray to [Float]
                    let count = outputArray.count
                    let doublePtr =  outputArray.dataPointer.bindMemory(to: Float.self, capacity: count)
                    let doubleBuffer = UnsafeBufferPointer(start: doublePtr, count: count)
                    let outputArrayFloat = Array(doubleBuffer)
                    
                    // Find maximum value and its index
                    if let maxValue = outputArrayFloat.max() {
                        if let maxIndex = outputArrayFloat.firstIndex(of: maxValue) {
                            let imagePredictor = ImagePredictor()

                            // Inside your captureOutput function after finding the max index
                            do {
                                let label = try imagePredictor.getLabel(index: maxIndex)
                                onPrediction?(label)
                            } catch {
                                print("An error occurred: \(error)")
                            }

                        }
                    }
                } else {
                    print("Prediction failed.")
                }
            }
        } catch {
            print("An error occurred during inference: \(error)")
        }
    }
}





