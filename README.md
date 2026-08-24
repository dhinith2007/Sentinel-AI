# Sentinel-AI: Distributed Edge-to-Cloud Driver Telemetry

"Sentinel" = Watcher + Protector

This repository contains the full architecture for a real-time driver monitoring system, heavily inspired by modern EV monitoring capabilities. It seamlessly synchronizes edge-based AI with cloud processing and centralized monitoring.

## System Architecture

The project is structured into three continuous functional layers:

1. **`/edge` (Phase 1):** The core intelligence running locally (e.g., Raspberry Pi). Detects driver drowsiness via Eye Aspect Ratio (EAR) and unauthorized phone usage using OpenCV & YOLO. Sends instant alerts via GPIO and syncs events to the cloud.
2. **`/backend` (Phase 2):** Fast cloud pipeline API built with FastAPI. It handles routing incoming Edge hardware logs, storing metadata in MongoDB, and buffering raw violation snapshots in S3.
3. **`/frontend` (Phase 3):** The next-generation Command Center. Built with Next.js and Tailwind CSS with a deep dark/neon UI. Visualizes edge telemetry live via dynamic grids and real-time alerts.
4. **`/docs`:** Architecture diagrams and further reading material.

## Getting Started
*(Further installation instructions will be provided in specific phase implementations).*
