//
//  AudioPlayer.swift
//  ANCDemo
//
//  AVAudioPlayer wrapper for Stage 2's Raw/Enhanced A/B playback. Tracks
//  which source (if any) is currently active so the UI can show per-button
//  playing state, and always stops any in-flight playback before starting
//  a new one -- two AVAudioPlayer instances can otherwise mix and play
//  simultaneously, which is what made Raw/Enhanced feel "confusing."

import Foundation
import AVFoundation
import Combine

enum AudioSource: Equatable {
    case raw
    case enhanced
}

@MainActor
final class AudioPlayer: NSObject, ObservableObject {
    @Published var playingSource: AudioSource?

    var isPlaying: Bool { playingSource != nil }

    private var player: AVAudioPlayer?

    func play(url: URL, source: AudioSource) {
        // Always stop whatever is currently playing first -- prevents two
        // AVAudioPlayer instances from mixing and producing garbled audio.
        stop()

        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
            try session.setActive(true)

            let newPlayer = try AVAudioPlayer(contentsOf: url)
            newPlayer.delegate = self
            player = newPlayer
            if newPlayer.play() {
                playingSource = source
            } else {
                playingSource = nil
            }
        } catch {
            print("AudioPlayer: playback failed -- \(error)")
            playingSource = nil
        }
    }

    func stop() {
        player?.stop()
        player = nil
        playingSource = nil
    }
}

extension AudioPlayer: @preconcurrency AVAudioPlayerDelegate {
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        playingSource = nil
    }
}
