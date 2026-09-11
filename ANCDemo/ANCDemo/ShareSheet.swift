//
//  ShareSheet.swift
//  ANCDemo
//
//  Stage 3: reusable SwiftUI wrapper around UIActivityViewController.
//  SwiftUI has no native share sheet component, so this is the standard
//  UIViewControllerRepresentable bridge -- presenting it with a file URL
//  surfaces AirDrop (plus Messages, Mail, Save to Files, etc.) as share
//  targets automatically; no AirDrop-specific code is needed.

import SwiftUI
import UIKit

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {
        // No dynamic updates needed -- the share sheet is presented once
        // per `items` set via ContentView's .sheet(isPresented:).
    }
}
