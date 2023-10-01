import Foundation
import CoreML

enum PredictorError: Error {
    case invalidInputTensor
    case predictionFailed
    case labelNotFound
}

class ImagePredictor {
    private var isRunning: Bool = false
    private var labels: [String] = []
    
    init() {
        // Initialize labels
        loadLabels()
    }
    
    // Function to load labels
    private func loadLabels() {
        if let filepath = Bundle.main.path(forResource: "labels", ofType: "txt") {
            do {
                let contents = try String(contentsOfFile: filepath)
                labels = contents.components(separatedBy: "\n")
            } catch {
                print("Could not read the file.")
            }
        } else {
            print("labels.txt file not found.")
        }
    }
    
    // Function to get label by index
    func getLabel(index: Int) throws -> String {
        guard index < labels.count else {
            throw PredictorError.labelNotFound
        }
        return labels[index]
    }
    
}
