import SwiftUI
import AVKit

struct ContentView: View {
    @State private var isScanning = false
    @State private var currentPrediction = "No prediction"
    @State private var cameraProcessor: CameraProcessor? = nil
    
    var body: some View {
        VStack {
            VStack {
                Text("Food Detector")
                    .font(.largeTitle)
                    .padding(.bottom, 20)
            }
            .padding(.top, 20)
            
            ZStack {
                if isScanning {
                    if let cameraProcessor = self.cameraProcessor {
                        CameraView(isScanning: $isScanning, cameraProcessor: cameraProcessor)
                            .cornerRadius(12)
                    }
                    VStack {
                        Spacer()
                        Text("Scanning...")
                            .foregroundColor(.white)
                            .padding(.bottom, 10)
                    }
                } else {
                    Color.gray.opacity(0.5)
                        .cornerRadius(12)
                    Text("Tap to Start Scanning")
                        .foregroundColor(.white)
                }
            }
            .frame(width: 350, height: 350)
            
            Text("Prediction: \(currentPrediction)")
                .font(.headline)
                .padding(.top, 20)
            
            Spacer()
            
            Button(action: {
                // Toggle scanning
                DispatchQueue.main.async {
                    self.isScanning.toggle()
                }
            }) {
                Text(isScanning ? "Stop Scanning" : "Start Scanning")
                    .foregroundColor(.white)
                    .padding(.horizontal, 40)
                    .padding(.vertical, 15)
                    .background(Color.blue)
                    .cornerRadius(8)
            }
            .padding(.bottom, 20)
        }
        .onAppear {
            if self.cameraProcessor == nil {
                self.cameraProcessor = CameraProcessor(currentPrediction: self.$currentPrediction)
            }
        }
    }
}
