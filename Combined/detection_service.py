
import grpc
from concurrent import futures
import time
import os
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO

# Import generated proto files
import videopipeline_pb2
import videopipeline_pb2_grpc

class DetectionServicer(videopipeline_pb2_grpc.DetectionServiceServicer):
    def __init__(self, model_variant='yolov8n'):
        self.model = YOLO(f'{model_variant}.pt')
        print(f"Detection service initialized with {model_variant}")

    def DetectObjects(self, request, context):
        start_time = time.time()

        # Decode image
        image = Image.open(io.BytesIO(request.image_data))

        # Run detection
        results = self.model(image, verbose=False)

        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, class_id in enumerate(boxes.cls):
                    if int(class_id) == 0 and boxes.conf[i] >= request.confidence_threshold:
                        box = boxes.xyxy[i]
                        detection = videopipeline_pb2.BoundingBox(
                            x1=float(box[0]),
                            y1=float(box[1]),
                            x2=float(box[2]),
                            y2=float(box[3]),
                            confidence=float(boxes.conf[i]),
                            class_id=int(class_id)
                        )
                        detections.append(detection)

        inference_time = (time.time() - start_time) * 1000  # ms

        return videopipeline_pb2.DetectionResponse(
            frame_id=request.frame_id,
            detections=detections,
            inference_time_ms=inference_time
        )

    def DetectObjectsBatch(self, request, context):
        for req in request.requests:
            response = self.DetectObjects(req, context)
            yield response

def serve():
    model_variant = os.environ.get('MODEL_VARIANT', 'yolov8n')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    videopipeline_pb2_grpc.add_DetectionServiceServicer_to_server(
        DetectionServicer(model_variant), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    print(f"Detection service started on port 50051 with {model_variant}")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
