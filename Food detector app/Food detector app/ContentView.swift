import SwiftUI
import AVKit

struct ContentView: View {
    @State private var isScanning = false
    
    // Create an instance of CameraProcessor
    private let cameraProcessor = CameraProcessor()
    
    var body: some View {
        VStack {
            // Your custom icons
            Image(systemName: "scanner")
            Image(systemName: "fork.knife.circle.fill")
            
            // Placeholder for Camera Feed
            if isScanning {
                CameraPreview(captureSession: cameraProcessor.captureSession)
                    .frame(width: 300, height: 300)
                    .cornerRadius(12)
            } else {
                Color.white
                    .frame(width: 300, height: 300)
                    .cornerRadius(12)
            }
            
            Text("Press the 'start scanning' button to find this food!")
            
            // Start/Stop Scanning Button
            Button(action: {
                self.isScanning.toggle()
                
                if self.isScanning {
                    CameraView(isScanning: $isScanning)
                        .frame(width: 300, height: 300)                } else {
                    cameraProcessor.captureSession.stopRunning()
                }
                
            }, label: {
                Text(isScanning ? "Stop Scanning" : "Start Scanning")
                    .foregroundColor(.white)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(8)
            })
        }
        .padding()
        .border(Color.blue, width: 0.5)
        .frame(width: 370, height: 540)
        .cornerRadius(12, antialiased: false)
    }
}

struct CameraPreview: UIViewRepresentable {
    
    var captureSession: AVCaptureSession
    
    func makeUIView(context: Context) -> some UIView {
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        
        let cameraView = UIView(frame: .zero)
        previewLayer.frame = cameraView.layer.bounds
        cameraView.layer.addSublayer(previewLayer)
        
        return cameraView
    }
    
    func updateUIView(_ uiView: UIViewType, context: Context) {
        // No-op
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
