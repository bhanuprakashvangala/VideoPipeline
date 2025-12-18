"""
T4: Agentic Framework with Real Multimodal LLM
Uses Qwen3/Gemma3/LLaMA3 via NRP API for intelligent decision-making
"""

import torch
import numpy as np
import cv2
import os
import sys
import time
import json
import requests
import base64
from io import BytesIO
from PIL import Image
from collections import deque, defaultdict
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# LLM API Configuration (from Adaptive_LLM_Inference)
# ============================================================================
LLM_CONFIG = {
    "base_url": "https://ellm.nrp-nautilus.io/v1",
    "api_key": "2KrDQlp6jRDIOuxqndLcZ2gaSiucYMQs",
    "models": {
        "qwen3": {"api_name": "qwen3", "params": "235B"},
        "gemma3": {"api_name": "gemma3", "params": "27B"},
        "llama3": {"api_name": "llama3-sdsc", "params": "70B"}
    }
}

# ============================================================================
# OC-SORT Tracker
# ============================================================================
class OCSort:
    def __init__(self, det_thresh=0.3, max_age=30, min_hits=3, iou_threshold=0.3):
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.track_id_count = 1

    def update(self, detections):
        self.frame_count += 1
        if torch.is_tensor(detections):
            detections = detections.cpu().numpy()
        if len(detections) > 0:
            detections = detections[detections[:, 4] >= self.det_thresh]

        for tracker in self.trackers:
            tracker['age'] += 1

        if len(detections) > 0 and len(self.trackers) > 0:
            iou_matrix = np.zeros((len(detections), len(self.trackers)))
            for d, det in enumerate(detections):
                for t, tracker in enumerate(self.trackers):
                    iou_matrix[d, t] = self._iou(det[:4], tracker['bbox'])

            det_indices, track_indices = linear_sum_assignment(-iou_matrix)
            matched = [(d, t) for d, t in zip(det_indices, track_indices) if iou_matrix[d, t] >= self.iou_threshold]
            matched_dets = set(m[0] for m in matched)

            for det_idx, track_idx in matched:
                self.trackers[track_idx]['bbox'] = detections[det_idx][:4]
                self.trackers[track_idx]['score'] = detections[det_idx][4]
                self.trackers[track_idx]['age'] = 0
                self.trackers[track_idx]['hits'] += 1

            for d in range(len(detections)):
                if d not in matched_dets:
                    self.trackers.append({'id': self.track_id_count, 'bbox': detections[d][:4],
                                         'score': detections[d][4], 'age': 0, 'hits': 1})
                    self.track_id_count += 1
        elif len(detections) > 0:
            for det in detections:
                self.trackers.append({'id': self.track_id_count, 'bbox': det[:4],
                                     'score': det[4], 'age': 0, 'hits': 1})
                self.track_id_count += 1

        self.trackers = [t for t in self.trackers if t['age'] <= self.max_age]
        results = []
        for tracker in self.trackers:
            if tracker['hits'] >= self.min_hits or self.frame_count <= self.min_hits:
                results.append([tracker['bbox'][0], tracker['bbox'][1], tracker['bbox'][2],
                               tracker['bbox'][3], tracker['id'], tracker['score']])
        return results

    def _iou(self, box1, box2):
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)

# ============================================================================
# LLM Client
# ============================================================================
class LLMClient:
    """Client for NRP LLM API"""

    def __init__(self, model_name="gemma3"):
        self.base_url = LLM_CONFIG["base_url"]
        self.api_key = LLM_CONFIG["api_key"]
        self.model_name = model_name
        self.api_model = LLM_CONFIG["models"][model_name]["api_name"]
        self.params = LLM_CONFIG["models"][model_name]["params"]
        print(f"[LLM] Initialized {model_name} ({self.params})")

    def query(self, prompt, max_tokens=150):
        """Send query to LLM and get response"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.api_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "stream": False
        }

        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            latency = (time.time() - start_time) * 1000

            data = response.json()
            text = data['choices'][0]['message']['content']

            return {
                "success": True,
                "response": text,
                "latency_ms": latency,
                "model": self.model_name
            }
        except Exception as e:
            return {
                "success": False,
                "response": f"Error: {str(e)}",
                "latency_ms": 0,
                "model": self.model_name
            }

# ============================================================================
# T4: Agentic Framework with LLM
# ============================================================================
@dataclass
class SLA:
    max_latency_ms: float = 500.0
    min_throughput_fps: float = 20.0
    min_accuracy: float = 0.75
    max_cost: float = 20.0

class AgenticLLMOptimizer:
    """
    Agentic Optimizer using Real LLM for Observe-Think-Act
    """
    def __init__(self, sla, llm_model="gemma3"):
        self.sla = sla
        self.llm = LLMClient(llm_model)
        self.load_history = deque(maxlen=10)
        self.latency_history = deque(maxlen=10)
        self.current_config = {
            'variant': 'yolov8n',
            'batch_size': 1,
            'replicas': 1,
            'cost': 1.0
        }
        self.decisions = []
        self.llm_responses = []
        self.last_llm_response = ""
        self.last_action = 'initialize'
        self.call_llm_every = 10  # Call LLM every N frames to save API calls
        self.frame_counter = 0

    def observe(self, latency_ms, num_detections, num_tracks, avg_confidence):
        """OBSERVE: Collect runtime metrics"""
        load = num_detections / 20.0
        self.load_history.append(load)
        self.latency_history.append(latency_ms)

        return {
            'latency_ms': latency_ms,
            'load': load,
            'num_detections': num_detections,
            'num_tracks': num_tracks,
            'avg_confidence': avg_confidence,
            'avg_latency': np.mean(list(self.latency_history)) if self.latency_history else latency_ms,
            'avg_load': np.mean(list(self.load_history)) if self.load_history else load
        }

    def think_with_llm(self, observation, frame_idx):
        """THINK: Use LLM to analyze and decide"""
        self.frame_counter += 1

        # Only call LLM every N frames to save API quota
        if self.frame_counter % self.call_llm_every != 1 and self.last_llm_response:
            # Use cached response
            return self._parse_llm_decision(self.last_llm_response, observation)

        # Build prompt for LLM
        prompt = f"""You are an AI video pipeline optimizer. Analyze these metrics and decide the action.

CURRENT METRICS:
- Frame: {frame_idx}
- Detections: {observation['num_detections']}
- Tracks: {observation['num_tracks']}
- Latency: {observation['latency_ms']:.1f}ms
- Load: {observation['load']:.2f}
- Avg Confidence: {observation['avg_confidence']:.2f}
- SLA Max Latency: {self.sla.max_latency_ms}ms

AVAILABLE ACTIONS:
1. MAINTAIN - Keep current config (system stable)
2. SCALE_UP - Increase replicas (high load)
3. SCALE_DOWN - Decrease replicas (low load)
4. SWITCH_MODEL - Change to faster/better model

Respond in exactly this format:
ACTION: <action_name>
REASON: <brief reason>
CONFIG: <yolov8n|yolov8s>, batch=<1|2|4>, replicas=<1-4>"""

        result = self.llm.query(prompt, max_tokens=100)

        if result["success"]:
            self.last_llm_response = result["response"]
            self.llm_responses.append({
                "frame": frame_idx,
                "response": result["response"],
                "latency": result["latency_ms"]
            })
            return self._parse_llm_decision(result["response"], observation)
        else:
            # Fallback to rule-based
            return self._rule_based_decision(observation)

    def _parse_llm_decision(self, response, observation):
        """Parse LLM response into decision"""
        lines = response.strip().split('\n')
        action = "MAINTAIN"
        reason = "LLM analysis"
        config_str = ""

        for line in lines:
            if line.startswith("ACTION:"):
                action = line.replace("ACTION:", "").strip()
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()
            elif line.startswith("CONFIG:"):
                config_str = line.replace("CONFIG:", "").strip()

        # Parse config if provided
        new_config = None
        if "yolov8" in config_str.lower():
            new_config = self._parse_config_string(config_str)

        return {
            'action': action,
            'reason': reason,
            'new_config': new_config,
            'llm_response': response,
            'sla_violated': observation['latency_ms'] > self.sla.max_latency_ms
        }

    def _parse_config_string(self, config_str):
        """Parse config string like 'yolov8n, batch=2, replicas=2'"""
        config = {'variant': 'yolov8n', 'batch_size': 1, 'replicas': 1, 'cost': 1.0}

        if 'yolov8s' in config_str.lower():
            config['variant'] = 'yolov8s'
            config['cost'] = 2.0
        elif 'yolov8m' in config_str.lower():
            config['variant'] = 'yolov8m'
            config['cost'] = 3.0

        if 'batch=' in config_str:
            try:
                batch = int(config_str.split('batch=')[1].split(',')[0].split()[0])
                config['batch_size'] = min(max(batch, 1), 8)
            except:
                pass

        if 'replicas=' in config_str:
            try:
                replicas = int(config_str.split('replicas=')[1].split(',')[0].split()[0])
                config['replicas'] = min(max(replicas, 1), 4)
                config['cost'] *= config['replicas']
            except:
                pass

        return config

    def _rule_based_decision(self, observation):
        """Fallback rule-based decision"""
        if observation['latency_ms'] > self.sla.max_latency_ms:
            return {
                'action': 'SCALE_UP',
                'reason': 'SLA violated - high latency',
                'new_config': {'variant': 'yolov8n', 'batch_size': 2, 'replicas': 2, 'cost': 2.0},
                'llm_response': '[Rule-based fallback]',
                'sla_violated': True
            }
        elif observation['load'] < 0.3:
            return {
                'action': 'SCALE_DOWN',
                'reason': 'Low load - reduce resources',
                'new_config': {'variant': 'yolov8n', 'batch_size': 1, 'replicas': 1, 'cost': 1.0},
                'llm_response': '[Rule-based fallback]',
                'sla_violated': False
            }
        else:
            return {
                'action': 'MAINTAIN',
                'reason': 'System stable',
                'new_config': None,
                'llm_response': '[Rule-based fallback]',
                'sla_violated': False
            }

    def act(self, analysis):
        """ACT: Execute configuration changes"""
        if analysis['new_config']:
            self.current_config = analysis['new_config']
            self.decisions.append(analysis)
        self.last_action = analysis['action']
        return analysis['new_config']

    def run_cycle(self, frame_idx, latency_ms, num_detections, num_tracks, avg_confidence):
        """Run one Observe-Think-Act cycle with LLM"""
        obs = self.observe(latency_ms, num_detections, num_tracks, avg_confidence)
        analysis = self.think_with_llm(obs, frame_idx)
        new_config = self.act(analysis)

        return {
            'observation': obs,
            'analysis': analysis,
            'new_config': new_config,
            'current_config': self.current_config
        }

# ============================================================================
# Video Generator with LLM
# ============================================================================
def create_agentic_llm_video():
    print("=" * 70)
    print("T4: AGENTIC FRAMEWORK WITH REAL LLM")
    print("Using Gemma3 (27B) via NRP API")
    print("=" * 70)

    # Setup paths
    sequence_folder = '../Dataset/MOT17/train/MOT17-04-DPM/img1'
    if not os.path.exists(sequence_folder):
        sequence_folder = 'Dataset/MOT17/train/MOT17-04-DPM/img1'
    if not os.path.exists(sequence_folder):
        sequence_folder = '/app/Dataset/MOT17/train/MOT17-04-DPM/img1'

    print(f"Loading images from: {sequence_folder}")
    all_images = sorted([f for f in os.listdir(sequence_folder) if f.endswith('.jpg')])
    NUM_FRAMES = min(50, len(all_images))  # Reduced for API limits
    image_files = [os.path.join(sequence_folder, f) for f in all_images[:NUM_FRAMES]]

    # Load model
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')

    # Initialize components
    tracker = OCSort()
    agent = AgenticLLMOptimizer(SLA(), llm_model="gemma3")

    # Get video dimensions
    first_frame = cv2.imread(image_files[0])
    H, W = first_frame.shape[:2]

    # Create video writer
    output_path = 't4_agentic_llm_demo.mp4'
    dashboard_width = 550
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 5, (W + dashboard_width, H))

    print(f"Processing {NUM_FRAMES} frames...")
    print(f"Output: {output_path}")
    print("-" * 70)

    # Colors
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]

    # Processing loop
    for frame_idx, img_path in enumerate(image_files):
        start_time = time.time()

        # Read frame
        frame = cv2.imread(img_path)

        # Run detection
        results = model(frame, verbose=False)

        # Extract detections
        frame_dets = []
        confidences = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, class_id in enumerate(boxes.cls):
                    if int(class_id) == 0 and boxes.conf[i] >= 0.3:
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i])
                        frame_dets.append([box[0], box[1], box[2], box[3], conf])
                        confidences.append(conf)

        # Update tracker
        if frame_dets:
            det_tensor = torch.tensor(np.array(frame_dets))
            tracks = tracker.update(det_tensor)
        else:
            tracks = []

        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        avg_conf = np.mean(confidences) if confidences else 0.0

        # Run agentic cycle with LLM
        cycle_result = agent.run_cycle(
            frame_idx, latency_ms, len(frame_dets), len(tracks), avg_conf
        )

        # Draw tracks on frame
        for track in tracks:
            x1, y1, x2, y2, track_id, conf = track
            color = COLORS[int(track_id) % len(COLORS)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f'ID:{int(track_id)}'
            cv2.putText(frame, label, (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Create dashboard
        dashboard = np.zeros((H, dashboard_width, 3), dtype=np.uint8)
        dashboard[:] = (30, 30, 30)

        y_offset = 30
        line_height = 22

        # Title
        cv2.putText(dashboard, "T4: AGENTIC + LLM", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 25
        cv2.putText(dashboard, f"Model: Gemma3 (27B)", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y_offset += line_height + 5

        cv2.line(dashboard, (10, y_offset), (dashboard_width-10, y_offset), (80, 80, 80), 1)
        y_offset += 15

        # OBSERVE
        cv2.putText(dashboard, "OBSERVE", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += line_height

        obs = cycle_result['observation']
        metrics = [
            f"Frame: {frame_idx+1}/{NUM_FRAMES}",
            f"Detections: {obs['num_detections']}",
            f"Tracks: {obs['num_tracks']}",
            f"Latency: {obs['latency_ms']:.0f}ms",
            f"Load: {obs['load']:.2f}",
            f"Avg Conf: {obs['avg_confidence']:.2f}"
        ]
        for m in metrics:
            cv2.putText(dashboard, m, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y_offset += 18

        y_offset += 5
        cv2.line(dashboard, (10, y_offset), (dashboard_width-10, y_offset), (80, 80, 80), 1)
        y_offset += 15

        # THINK (LLM)
        cv2.putText(dashboard, "THINK (LLM)", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += line_height

        analysis = cycle_result['analysis']

        # Show LLM response (wrapped)
        llm_resp = analysis.get('llm_response', '')[:200]
        cv2.putText(dashboard, "LLM Response:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y_offset += 16

        # Wrap text
        words = llm_resp.split()
        line = ""
        for word in words:
            if len(line + word) < 45:
                line += word + " "
            else:
                cv2.putText(dashboard, line.strip(), (25, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 255), 1)
                y_offset += 14
                line = word + " "
                if y_offset > 300:
                    break
        if line and y_offset <= 300:
            cv2.putText(dashboard, line.strip(), (25, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 255), 1)
            y_offset += 14

        y_offset += 10
        cv2.line(dashboard, (10, y_offset), (dashboard_width-10, y_offset), (80, 80, 80), 1)
        y_offset += 15

        # ACT
        cv2.putText(dashboard, "ACT", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height

        action = analysis.get('action', 'MAINTAIN')
        reason = analysis.get('reason', '')[:50]

        # Action color
        if action == 'MAINTAIN':
            action_color = (0, 255, 0)
        elif action in ['SCALE_UP', 'SWITCH_MODEL']:
            action_color = (0, 165, 255)
        else:
            action_color = (0, 255, 255)

        cv2.putText(dashboard, f"Action: {action}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, action_color, 2)
        y_offset += line_height
        cv2.putText(dashboard, f"Reason: {reason}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y_offset += line_height + 5

        cv2.line(dashboard, (10, y_offset), (dashboard_width-10, y_offset), (80, 80, 80), 1)
        y_offset += 15

        # CONFIG
        cv2.putText(dashboard, "CURRENT CONFIG", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        y_offset += line_height

        config = cycle_result['current_config']
        config_items = [
            f"Model: {config['variant']}",
            f"Batch: {config['batch_size']}",
            f"Replicas: {config['replicas']}",
            f"Cost: {config['cost']:.1f}"
        ]
        for c in config_items:
            cv2.putText(dashboard, c, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y_offset += 18

        y_offset += 5
        cv2.line(dashboard, (10, y_offset), (dashboard_width-10, y_offset), (80, 80, 80), 1)
        y_offset += 15

        # Stats
        cv2.putText(dashboard, "STATISTICS", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 200), 2)
        y_offset += line_height
        cv2.putText(dashboard, f"LLM Calls: {len(agent.llm_responses)}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_offset += 18
        cv2.putText(dashboard, f"Config Changes: {len(agent.decisions)}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Architecture at bottom
        y_offset = H - 80
        cv2.line(dashboard, (10, y_offset), (dashboard_width-10, y_offset), (80, 80, 80), 1)
        y_offset += 20

        # Draw OBSERVE -> THINK -> ACT boxes
        box_w, box_h = 100, 35
        gap = 30
        start_x = 30

        phases = [('OBSERVE', (0, 255, 255)), ('THINK+LLM', (255, 255, 0)), ('ACT', (0, 255, 0))]
        for i, (phase, color) in enumerate(phases):
            x = start_x + i * (box_w + gap)
            cv2.rectangle(dashboard, (x, y_offset), (x + box_w, y_offset + box_h), color, 2)
            text_size = cv2.getTextSize(phase, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = x + (box_w - text_size[0]) // 2
            cv2.putText(dashboard, phase, (text_x, y_offset + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if i < 2:
                cv2.arrowedLine(dashboard, (x + box_w + 5, y_offset + box_h//2),
                               (x + box_w + gap - 5, y_offset + box_h//2),
                               (100, 100, 100), 1, tipLength=0.3)

        # Combine
        combined = np.hstack([frame, dashboard])
        out.write(combined)

        # Progress
        if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
            print(f"Frame {frame_idx+1}/{NUM_FRAMES} - Action: {action} - LLM calls: {len(agent.llm_responses)}")

    out.release()

    # Save LLM responses
    with open('t4_llm_responses.json', 'w') as f:
        json.dump(agent.llm_responses, f, indent=2)

    print("-" * 70)
    print(f"Video saved: {output_path}")
    print(f"Total LLM calls: {len(agent.llm_responses)}")
    print(f"Total config changes: {len(agent.decisions)}")
    print(f"LLM responses saved: t4_llm_responses.json")
    print("=" * 70)

    return output_path

if __name__ == "__main__":
    create_agentic_llm_video()
