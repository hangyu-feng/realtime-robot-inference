# realtime-robot-inference

Real-time cloud inference service for a robot fleet: robots stream 640×480 JPEG frames at 30 fps, the cloud batches them through CPU-only ONNX models (MobileNetV3-small classification, MiDaS-small depth), and streams control commands back with bounded freshness and fair sharing across robots.

Work in progress.
