//
//  ModelRunner.swift
//  ANCDemo
//
//  Loads the fine-tuned dns48 INT8 Core ML model ONCE (via `shared`) and
//  exposes reusable inference helpers. Stage 1 used this to verify the
//  model loads/runs correctly on-device (4 back-to-back runs on the same
//  instance confirmed a cold-start pattern: ~274ms first call, ~53-55ms
//  warm -- same shape as the Phase 3a Mac finding). Stage 2 reuses the
//  SAME model instance (via the app-launch warmup call) and the SAME
//  fit-to-64000/prediction code path for real recorded audio -- no second
//  implementation of either.
//
//  MODEL INPUT CONTRACT (verified against src/export/traceable_demucs.py,
//  src/export/to_coreml.py, and src/export/verify_onnx.py's
//  _fit_to_fixed_length -- not assumed):
//   - Sample rate: 16000 Hz, mono, Float32 PCM.
//   - Exact sample count: 64000 (4.0s @ 16kHz). The traced/converted graph
//     is ONLY valid at this exact length (TraceableDemucs hardcodes
//     length-dependent padding math for length=64000; any other length
//     silently produces wrong output).
//   - Fit convention (from verify_onnx.py::_fit_to_fixed_length, the exact
//     function used to produce this project's verified accuracy numbers):
//       - if longer than 64000 samples: CENTER-CROP to 64000.
//       - if shorter: ZERO-PAD SYMMETRICALLY (pad_left = total//2,
//         pad_right = remainder), NOT trailing-only zero-pad.
//     This is NOT the model's own internal valid_length padding (that's a
//     separate, already-baked-into-the-graph 85-sample pad inside
//     TraceableDemucs.forward) -- this is the input-fitting step that must
//     happen in Swift before the tensor ever reaches the model.
//   - Input/output tensor name: "noisy_waveform" / "enhanced_waveform",
//     shape (1, 1, 64000).
//
//  GENERATED CLASS NAME (Xcode auto-generates a Swift interface from the
//  .mlpackage at build time -- EXPECTATION, not yet confirmed against a
//  real build): for `dns48_finetuned_v5_int8.mlpackage`, Xcode's coremlc
//  tool should generate a class named `dns48_finetuned_v5_int8` with a
//  `class func load(...)` / `init(configuration:)` and an instance method
//  `prediction(noisy_waveform: MLMultiArray) throws -> dns48_finetuned_v5_int8Output`
//  (the output type name is derived the same way, `<ModelName>Output`).
//  If the actual generated symbol differs (Xcode sometimes sanitizes
//  leading digits/underscores in the class name -- e.g. a leading digit
//  in the model name can get prefixed or altered), Xcode's autocomplete
//  in this same file will show the real name -- fix the two references
//  below (`dns48_finetuned_v5_int8` type name) to match. This is flagged
//  explicitly here so a mismatch shows up as an obvious two-line fix, not
//  a mysterious compile error.

import Foundation
import AVFoundation
import CoreML

enum ModelRunnerError: Error, CustomStringConvertible {
    case audioFileNotFound
    case audioFormatConversionFailed
    case pcmBufferReadFailed
    case modelLoadFailed(String)
    case predictionFailed(String)
    case outputExtractionFailed

    var description: String {
        switch self {
        case .audioFileNotFound: return "bundled test audio file not found in app bundle"
        case .audioFormatConversionFailed: return "failed to convert audio to 16kHz mono Float32"
        case .pcmBufferReadFailed: return "failed to read PCM samples from audio buffer"
        case .modelLoadFailed(let msg): return "model load failed: \(msg)"
        case .predictionFailed(let msg): return "prediction failed: \(msg)"
        case .outputExtractionFailed: return "failed to extract output samples from model prediction"
        }
    }
}

enum ModelRunner {

    // Verified constants -- see file header. Do not change without
    // re-verifying against src/export/traceable_demucs.py.
    static let expectedSampleRate: Double = 16000
    static let expectedSampleCount: Int = 64000

    /// The model is loaded exactly once and reused for every call (warmup
    /// AND every real recording) -- this is what makes the launch warmup
    /// meaningful: it's the SAME instance the record flow uses, not a
    /// throwaway one. Backed by a lazy static, which Swift itself
    /// initializes thread-safely exactly once.
    private static var sharedModel: dns48_finetuned_v5_int8?

    private static func modelInstance() throws -> dns48_finetuned_v5_int8 {
        if let existing = sharedModel {
            return existing
        }
        let model = try loadModel()
        sharedModel = model
        return model
    }

    /// Result of one inference call: the enhanced samples and the
    /// wall-clock time (ms) for JUST the prediction() call, measured in
    /// Swift via CFAbsoluteTimeGetCurrent -- not the Mac-side benchmark,
    /// not model load time, not audio I/O time.
    struct InferenceResult {
        let outputSamples: [Float]
        let inferenceTimeMs: Double
    }

    /// Silent launch warmup (Stage 2, task 1): loads the model (if not
    /// already loaded) and runs ONE throwaway prediction call on the
    /// bundled Stage 1 test file, discarding the output. This absorbs the
    /// cold-start cost (measured ~274ms on-device) before the user ever
    /// taps record, so the FIRST real recording's measured inference time
    /// reflects the warm ~53-55ms number, not the cold one. No UI is tied
    /// to this -- call it fire-and-forget from a background Task at launch.
    /// Reuses the Stage 1 bundled file rather than a synthetic buffer,
    /// since it's already verified to load/fit/convert correctly end to
    /// end -- a synthetic buffer would be one more untested code path for
    /// no benefit (the model doesn't care about content, only shape, for
    /// warmup purposes).
    static func warmup() {
        do {
            let inputSamples = try loadAndFitTestAudio()
            let inputArray = try makeMLMultiArray(from: inputSamples)
            let model = try modelInstance()
            _ = try runPrediction(model: model, input: inputArray)
            print("ModelRunner: silent warmup complete.")
        } catch {
            print("ModelRunner: warmup failed (non-fatal, first real recording will just be cold) -- \(error)")
        }
    }

    /// Reusable Stage 2 entrypoint: fits already-loaded samples (from a
    /// live recording, NOT the bundled test file) to the model's exact
    /// input contract, runs inference on the shared model instance, and
    /// returns both the output and the measured time for THIS call only.
    static func process(rawSamples: [Float]) throws -> InferenceResult {
        let fitted = fitToFixedLength(rawSamples, length: expectedSampleCount)
        let inputArray = try makeMLMultiArray(from: fitted)
        let model = try modelInstance()

        let startTime = CFAbsoluteTimeGetCurrent()
        let outputSamples = try runPrediction(model: model, input: inputArray)
        let elapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

        return InferenceResult(outputSamples: outputSamples, inferenceTimeMs: elapsedMs)
    }

    /// Stage 1's original standalone verification: load bundled test
    /// audio, run 4 back-to-back prediction() calls on the same model
    /// instance, log each run's timing plus a summary, write Run 1's
    /// output to a temp .wav. Kept as a manually-invokable diagnostic
    /// (no longer wired to app launch -- Stage 2 uses `warmup()` +
    /// `process(rawSamples:)` instead) in case the cold/warm pattern ever
    /// needs re-checking after a model/device change.
    static func runVerification() {
        print("=== ModelRunner Stage 1 verification starting ===")
        do {
            let inputSamples = try loadAndFitTestAudio()
            print("Input sample count (after fit-to-64000): \(inputSamples.count)")

            let model = try modelInstance()
            let inputArray = try makeMLMultiArray(from: inputSamples)

            let runCount = 4
            var runTimesMs: [Double] = []
            var firstOutputSamples: [Float] = []

            for runIndex in 1...runCount {
                let startTime = CFAbsoluteTimeGetCurrent()
                let outputSamples = try runPrediction(model: model, input: inputArray)
                let elapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
                runTimesMs.append(elapsedMs)

                print("Run \(runIndex): output sample count = \(outputSamples.count)")
                print(String(format: "Run \(runIndex): inference time = %.2f ms", elapsedMs))

                if runIndex == 1 {
                    firstOutputSamples = outputSamples
                }
            }

            print("=== Inference time summary (all \(runCount) runs, same model instance, same input) ===")
            for (index, time) in runTimesMs.enumerated() {
                print(String(format: "  Run %d: %.2f ms", index + 1, time))
            }

            let outputURL = try writeOutputWav(samples: firstOutputSamples, sampleRate: expectedSampleRate)
            print("Enhanced audio (Run 1 output) written to: \(outputURL.path)")

            print("SUCCESS: ModelRunner Stage 1 verification completed.")
        } catch {
            print("FAILURE: ModelRunner Stage 1 verification failed -- \(error)")
        }
    }

    // MARK: - Audio loading + fitting

    /// Loads the bundled test .wav via AVFoundation, converts to 16kHz mono
    /// Float32, then fits to exactly `expectedSampleCount` samples using the
    /// SAME center-crop / symmetric-zero-pad convention verified in
    /// src/export/verify_onnx.py::_fit_to_fixed_length.
    static func loadAndFitTestAudio() throws -> [Float] {
        guard let url = Bundle.main.url(forResource: "test_gunshot_noisy", withExtension: "wav") else {
            throw ModelRunnerError.audioFileNotFound
        }

        let file = try AVAudioFile(forReading: url)
        let sourceFormat = file.processingFormat

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: expectedSampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw ModelRunnerError.audioFormatConversionFailed
        }

        let rawSamples: [Float]
        if sourceFormat.sampleRate == expectedSampleRate
            && sourceFormat.channelCount == 1
            && sourceFormat.commonFormat == .pcmFormatFloat32 {
            rawSamples = try readAllSamples(file: file, format: sourceFormat)
        } else {
            guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
                throw ModelRunnerError.audioFormatConversionFailed
            }
            rawSamples = try convertAndReadAllSamples(file: file, sourceFormat: sourceFormat, targetFormat: targetFormat, converter: converter)
        }

        return fitToFixedLength(rawSamples, length: expectedSampleCount)
    }

    private static func readAllSamples(file: AVAudioFile, format: AVAudioFormat) throws -> [Float] {
        let frameCount = AVAudioFrameCount(file.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw ModelRunnerError.pcmBufferReadFailed
        }
        try file.read(into: buffer)
        return floatArray(from: buffer)
    }

    private static func convertAndReadAllSamples(
        file: AVAudioFile,
        sourceFormat: AVAudioFormat,
        targetFormat: AVAudioFormat,
        converter: AVAudioConverter
    ) throws -> [Float] {
        let sourceFrameCount = AVAudioFrameCount(file.length)
        guard let sourceBuffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: sourceFrameCount) else {
            throw ModelRunnerError.pcmBufferReadFailed
        }
        try file.read(into: sourceBuffer)

        let ratio = targetFormat.sampleRate / sourceFormat.sampleRate
        let outputFrameCapacity = AVAudioFrameCount(Double(sourceFrameCount) * ratio) + 1024
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputFrameCapacity) else {
            throw ModelRunnerError.pcmBufferReadFailed
        }

        var error: NSError?
        var suppliedSource = false
        converter.convert(to: outputBuffer, error: &error) { _, outStatus in
            if suppliedSource {
                outStatus.pointee = .noDataNow
                return nil
            }
            suppliedSource = true
            outStatus.pointee = .haveData
            return sourceBuffer
        }
        if let error = error {
            throw ModelRunnerError.predictionFailed("AVAudioConverter error: \(error)")
        }

        return floatArray(from: outputBuffer)
    }

    private static func floatArray(from buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channelData = buffer.floatChannelData else { return [] }
        let frameLength = Int(buffer.frameLength)
        return Array(UnsafeBufferPointer(start: channelData[0], count: frameLength))
    }

    /// Center-crop if longer, symmetric zero-pad if shorter. Mirrors
    /// verify_onnx.py::_fit_to_fixed_length exactly.
    static func fitToFixedLength(_ samples: [Float], length: Int) -> [Float] {
        let n = samples.count
        if n == length {
            return samples
        }
        if n > length {
            let start = (n - length) / 2
            return Array(samples[start..<(start + length)])
        }
        let padTotal = length - n
        let padLeft = padTotal / 2
        let padRight = padTotal - padLeft
        return [Float](repeating: 0, count: padLeft) + samples + [Float](repeating: 0, count: padRight)
    }

    // MARK: - Model loading + prediction

    static func loadModel() throws -> dns48_finetuned_v5_int8 {
        do {
            let config = MLModelConfiguration()
            return try dns48_finetuned_v5_int8(configuration: config)
        } catch {
            throw ModelRunnerError.modelLoadFailed("\(error)")
        }
    }

    static func makeMLMultiArray(from samples: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, 1, NSNumber(value: samples.count)], dataType: .float32)
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: samples.count)
        for i in 0..<samples.count {
            pointer[i] = samples[i]
        }
        return array
    }

    static func runPrediction(model: dns48_finetuned_v5_int8, input: MLMultiArray) throws -> [Float] {
        let output: dns48_finetuned_v5_int8Output
        do {
            output = try model.prediction(noisy_waveform: input)
        } catch {
            throw ModelRunnerError.predictionFailed("\(error)")
        }

        let outArray = output.enhanced_waveform
        let count = outArray.count
        guard let pointer = try? UnsafeBufferPointer<Float>(outArray) else {
            throw ModelRunnerError.outputExtractionFailed
        }
        return Array(pointer.prefix(count))
    }

    // MARK: - Output writing

    static func writeOutputWav(samples: [Float], sampleRate: Double, filename: String = "enhanced_output.wav") throws -> URL {
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false) else {
            throw ModelRunnerError.audioFormatConversionFailed
        }
        let tempDir = FileManager.default.temporaryDirectory
        let outputURL = tempDir.appendingPathComponent(filename)

        let outFile = try AVAudioFile(forWriting: outputURL, settings: format.settings)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else {
            throw ModelRunnerError.pcmBufferReadFailed
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        if let channelData = buffer.floatChannelData {
            for i in 0..<samples.count {
                channelData[0][i] = samples[i]
            }
        }
        try outFile.write(from: buffer)
        return outputURL
    }
}

private extension UnsafeBufferPointer where Element == Float {
    init(_ multiArray: MLMultiArray) throws {
        guard multiArray.dataType == .float32 else {
            throw ModelRunnerError.outputExtractionFailed
        }
        let count = multiArray.count
        let pointer = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        self.init(start: pointer, count: count)
    }
}
