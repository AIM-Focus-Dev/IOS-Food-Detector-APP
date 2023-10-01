//
//  CameraProcessorViewModel.swift
//  Food detector app
//
//  Created by Nelio Barbosa on 01/10/2023.
//

import Foundation
import SwiftUI

class CameraProcessorViewModel: ObservableObject {
    @Published var currentPrediction: String = "No prediction"

    let cameraProcessor: CameraProcessor

    init() {
        self.cameraProcessor = CameraProcessor()
        self.cameraProcessor.onPrediction = handlePrediction
    }

    private func handlePrediction(_ prediction: String) {
        DispatchQueue.main.async {
            self.currentPrediction = prediction
        }
    }
}

struct CameraView: UIViewControllerRepresentable {
    @Binding var isScanning: Bool
    var cameraProcessor: CameraProcessor
    
    func makeUIViewController(context: Context) -> UIViewController {
        let viewController = UIViewController()
        
        cameraProcessor.previewLayer.frame = viewController.view.bounds
        viewController.view.layer.addSublayer(cameraProcessor.previewLayer)
        
        return viewController
    }
    
    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
        // Update the preview layer when isScanning changes
        if isScanning {
            cameraProcessor.setPreviewLayerFrame(frame: uiViewController.view.bounds)
            if cameraProcessor.previewLayer.superlayer == nil {
                uiViewController.view.layer.addSublayer(cameraProcessor.previewLayer)
            }
        } else {
            cameraProcessor.previewLayer.removeFromSuperlayer()
        }
    }
}
