//
//  ImageFramesPreprocessing.swift
//  Food detector app
//
//  Created by Nelio Barbosa on 01/10/2023.
//

import Foundation
import UIKit
import CoreML



// convert UIimage to MLMultiArray
extension UIImage {
    
    func mlMultiArray(scale preprocessScale: Double = 255,
                      rBias preprocessRBias: Double = 0,
                      gBias preprocessGBias: Double = 0,
                      bBias preprocessBBias: Double = 0) -> MLMultiArray? {
        
        guard let cgImage = self.cgImage else {
            return nil
        }
        
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo: UInt32 = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        
        guard let context = CGContext(data: &pixelData,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: bitmapInfo) else {
            return nil
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
        
        let count = width * height
        guard let mlArray = try? MLMultiArray(shape: [1, 3, NSNumber(integerLiteral: height), NSNumber(integerLiteral: width)], dataType: .double) else {
            return nil
        }
        
        for i in 0..<count {
            let r = Double(pixelData[i * 4]) / preprocessScale + preprocessRBias
            let g = Double(pixelData[i * 4 + 1]) / preprocessScale + preprocessGBias
            let b = Double(pixelData[i * 4 + 2]) / preprocessScale + preprocessBBias
            
            mlArray[i] = NSNumber(value: r)
            mlArray[count + i] = NSNumber(value: g)
            mlArray[2 * count + i] = NSNumber(value: b)
        }
        
        return mlArray
    }
}

// Force resize the image to a specific target size
func forceResizeImage(image: UIImage, targetSize: CGSize) -> UIImage {
    let rect = CGRect(x: 0, y: 0, width: targetSize.width, height: targetSize.height)
    
    UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
    image.draw(in: rect)
    let newImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    
    return newImage!
}



// Process frames, convert to UIimage
extension CameraProcessor {
    
    func pixelBufferToUIImage(pixelBuffer: CVPixelBuffer) -> UIImage? {
        // Convert CVPixelBuffer to CIImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        // Convert CIImage to UIImage
        let temporaryContext = CIContext(options: nil)
        guard let videoImage = temporaryContext.createCGImage(ciImage, from: CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))) else {
            return nil
        }
        return UIImage(cgImage: videoImage)
    }
    
    func preprocessFrame(pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        // Step 1: Convert to UIImage
        guard let uiImage = pixelBufferToUIImage(pixelBuffer: pixelBuffer) else {
            return nil
        }
        
        // Step 2: Force resize UIImage to 400x400
        let resizedImage = forceResizeImage(image: uiImage, targetSize: CGSize(width: 400, height: 400))
        
        // Step 3: Convert UIImage to MLMultiArray using your existing extension
        guard let multiArray = resizedImage.mlMultiArray(scale: 255, rBias: 0.42714671, gBias: 0.42714637, bBias: 0.42714605) else {
            return nil
        }
        
        return multiArray
    }

}
