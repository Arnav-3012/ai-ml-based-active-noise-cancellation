//
//  ContentView.swift
//  ANCDemo
//
//  Stage 2: record -> infer -> play-locally flow, with Raw/Enhanced A/B
//  playback so the audience can hear the before/after. Stage 3 adds a
//  "Clean & Share" button that presents iOS's Share Sheet
//  (UIActivityViewController) with the enhanced file, surfacing AirDrop
//  automatically -- no AirDrop-specific code needed, iOS handles that once
//  the Share Sheet is shown with a file URL.
//

import SwiftUI
import UIKit

enum DemoStatus: String {
    case ready = "Ready"
    case recording = "Recording"
    case processing = "Processing"
    case done = "Done"
    case error = "Error"
}

struct ContentView: View {
    @StateObject private var recorder = AudioRecorder()
    @StateObject private var player = AudioPlayer()

    @State private var status: DemoStatus = .ready
    @State private var lastInferenceTimeMs: Double?
    @State private var lastErrorMessage: String?

    // Raw/Enhanced A/B playback + Stage 3 share target -- both point at
    // real files on disk (raw = AudioRecorder's own recording file,
    // enhanced = ModelRunner's freshly-written output), not re-derived or
    // copied.
    @State private var rawAudioURL: URL?
    @State private var enhancedAudioURL: URL?
    @State private var isShareSheetPresented = false

    private var deviceModelName: String {
        UIDevice.current.name
    }

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            Text(deviceModelName)
                .font(.headline)
                .foregroundStyle(.secondary)

            statusView

            recordButton

            if recorder.isRecording {
                Text(String(format: "%.1fs", recorder.recordingDuration))
                    .font(.title3.monospacedDigit())
                    .foregroundStyle(.red)
            }

            if let timeMs = lastInferenceTimeMs {
                Text(String(format: "Inference time: %.2f ms", timeMs))
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            if status == .done {
                playbackAndShareControls
            }

            if let errorMessage = lastErrorMessage {
                Text(errorMessage)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }

            Spacer()
        }
        .padding()
        .task {
            // Silent launch warmup (Stage 2, task 1): absorbs the model's
            // cold-start cost before the user can interact, so the first
            // real recording's measured inference time is the warm number.
            // No UI is tied to this -- runs invisibly in the background.
            await Task.detached(priority: .userInitiated) {
                ModelRunner.warmup()
            }.value
        }
        .sheet(isPresented: $isShareSheetPresented) {
            if let enhancedAudioURL {
                ShareSheet(items: [enhancedAudioURL])
            }
        }
    }

    private var playbackAndShareControls: some View {
        VStack(spacing: 12) {
            // Order matches the logical listening flow: hear the noisy
            // original first, then the cleaned result, then share it.
            playbackButton(
                source: .raw,
                url: rawAudioURL,
                idleLabel: "Original (Noisy)",
                idleIcon: "waveform",
                tint: .gray
            )

            playbackButton(
                source: .enhanced,
                url: enhancedAudioURL,
                idleLabel: "Cleaned (AI)",
                idleIcon: "sparkles",
                tint: .accentColor
            )

            Button {
                isShareSheetPresented = true
            } label: {
                Label("Clean & Share", systemImage: "square.and.arrow.up")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(.green)
            .disabled(enhancedAudioURL == nil)
        }
    }

    /// Single playback button that visually distinguishes raw vs. enhanced
    /// via icon/tint, flips to a "Stop" state with a playing indicator
    /// while its own source is active, and disables itself while the
    /// *other* source is playing (tapping it always stops the other first
    /// regardless, as a second line of defense against overlap).
    private func playbackButton(
        source: AudioSource,
        url: URL?,
        idleLabel: String,
        idleIcon: String,
        tint: Color
    ) -> some View {
        let isThisPlaying = player.playingSource == source
        let isOtherPlaying = player.isPlaying && !isThisPlaying

        return Button {
            if isThisPlaying {
                player.stop()
            } else if let url {
                player.play(url: url, source: source)
            }
        } label: {
            Label {
                Text(isThisPlaying ? "Stop" : idleLabel)
            } icon: {
                if isThisPlaying {
                    Image(systemName: "stop.fill")
                } else {
                    Image(systemName: idleIcon)
                }
            }
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .tint(isThisPlaying ? .red : tint)
        .disabled(url == nil || isOtherPlaying)
    }

    private var statusView: some View {
        Text(status.rawValue)
            .font(.title2.bold())
            .foregroundStyle(statusColor)
    }

    private var statusColor: Color {
        switch status {
        case .ready: return .primary
        case .recording: return .red
        case .processing: return .orange
        case .done: return .green
        case .error: return .red
        }
    }

    private var recordButton: some View {
        Button(action: handleRecordButtonTapped) {
            ZStack {
                Circle()
                    .fill(recorder.isRecording ? Color.red : Color.blue)
                    .frame(width: 100, height: 100)
                if recorder.isRecording {
                    // Pulsing indicator while actively recording, so it's
                    // obvious to an audience the app is listening.
                    Circle()
                        .stroke(Color.red.opacity(0.5), lineWidth: 4)
                        .frame(width: 120, height: 120)
                        .scaleEffect(recorder.isRecording ? 1.2 : 1.0)
                        .opacity(recorder.isRecording ? 0 : 1)
                        .animation(
                            .easeOut(duration: 1.0).repeatForever(autoreverses: false),
                            value: recorder.isRecording
                        )
                }
                Image(systemName: recorder.isRecording ? "stop.fill" : "mic.fill")
                    .font(.system(size: 36))
                    .foregroundStyle(.white)
            }
        }
        .disabled(status == .processing)
    }

    private func handleRecordButtonTapped() {
        if recorder.isRecording {
            stopRecordingAndProcess()
        } else {
            startRecording()
        }
    }

    private func startRecording() {
        lastErrorMessage = nil
        lastInferenceTimeMs = nil
        rawAudioURL = nil
        enhancedAudioURL = nil
        Task {
            do {
                try await recorder.requestPermissionAndStart()
                status = .recording
            } catch {
                status = .error
                lastErrorMessage = "\(error)"
                print("ContentView: failed to start recording -- \(error)")
            }
        }
    }

    private func stopRecordingAndProcess() {
        status = .processing
        Task {
            do {
                let rawSamples = try recorder.stopAndReadSamples()
                print("ContentView: recorded \(rawSamples.count) raw samples.")
                rawAudioURL = recorder.recordingURL

                // Reuses ModelRunner's already-warm shared model instance
                // and the SAME fitToFixedLength/prediction path as Stage 1
                // and the launch warmup -- not a new instance, not a second
                // implementation.
                let result = try ModelRunner.process(rawSamples: rawSamples)
                print(String(format: "ContentView: recording inference time = %.2f ms", result.inferenceTimeMs))

                // Named for readability once shared/AirDropped -- e.g.
                // "cleaned_speech_2026-09-12_14-30-05.wav" -- rather than
                // the raw recording's UUID-based temp name, so it's
                // recognizable when it lands on the receiving device.
                let outputURL = try ModelRunner.writeOutputWav(
                    samples: result.outputSamples,
                    sampleRate: ModelRunner.expectedSampleRate,
                    filename: Self.sharableFilename()
                )
                print("ContentView: enhanced output written to \(outputURL.path)")

                lastInferenceTimeMs = result.inferenceTimeMs
                enhancedAudioURL = outputURL
                status = .done

                // Automatic playback -- no extra button, happens the
                // moment processing finishes. Raw/Enhanced A/B buttons and
                // Clean & Share become available once status == .done.
                player.play(url: outputURL, source: .enhanced)
            } catch {
                status = .error
                lastErrorMessage = "\(error)"
                print("ContentView: record -> infer -> play flow failed -- \(error)")
            }
        }
    }

    /// "cleaned_speech_<timestamp>.wav" -- readable on the receiving
    /// device after AirDrop/share, unlike a raw UUID filename.
    private static func sharableFilename() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        return "cleaned_speech_\(formatter.string(from: Date())).wav"
    }
}

#Preview {
    ContentView()
}
