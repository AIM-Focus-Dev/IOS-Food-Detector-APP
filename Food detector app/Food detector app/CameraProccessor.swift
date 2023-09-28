import Foundation
import SwiftUI
import AVFoundation
import CoreML
import Vision

class CameraProcessor: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, ObservableObject {
    
    var previewLayer: AVCaptureVideoPreviewLayer
    var captureSession: AVCaptureSession
    
    override init() {
        captureSession = AVCaptureSession()
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        super.init()
        
        // Start the capture session
        startCaptureSession()
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
    
    // ... (rest of the code remains unchanged)
}

struct CameraView: UIViewControllerRepresentable {
    @Binding var isScanning: Bool
    private let cameraProcessor = CameraProcessor()
    
    func makeUIViewController(context: Context) -> UIViewController {
        let viewController = UIViewController()
        let cameraProcessor = CameraProcessor()
        
        cameraProcessor.previewLayer.frame = viewController.view.bounds
        viewController.view.layer.addSublayer(cameraProcessor.previewLayer)
        
        return viewController
    }
    
    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
        // Update the preview layer when isScanning changes
        if isScanning {
            cameraProcessor.setPreviewLayerFrame(uiViewController.view.bounds)
            if cameraProcessor.previewLayer.superlayer == nil {
                uiViewController.view.layer.addSublayer(cameraProcessor.previewLayer)
            }
        } else {
            cameraProcessor.previewLayer.removeFromSuperlayer()
        }
    }
}
