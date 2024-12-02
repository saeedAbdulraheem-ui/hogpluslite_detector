import cv2
import face_recognition
import imutils
import pickle
import numpy as np
import tflite_runtime.interpreter as tflite
from imutils.video import VideoStream
from imutils.video import FPS
import time
import statistics
import matplotlib.pyplot as plt

# Load encodings for facial recognition
encodingsP = "encodings.pickle"
print("[INFO] Loading encodings and face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# Load TFLite model and labels for object detection
MODEL_PATH = "model.tflite"
LABEL_PATH = "labels.txt"
MIN_CONF_THRESH = 0.5
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5
with open(LABEL_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize video stream
contrast_val=0
sharpness_val=0
vs = VideoStream(src=0, usePiCamera=True, contrast=contrast_val, brightness=55, drc_strength='high').start()
# vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# Define video writer to save the video at the end
output_path = "output.mp4"  # Save as MP4
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use MP4 codec
fps_rate = 0.52  # fps rate we roughly get with the pi
frame_width = 600  # Width of resized frame
# calculate this using the dimensions from oKDo 5mp camera documentation
frame_height = int(1944 * (frame_width / 2592))
# use standard 1080p resolution
video_writer = cv2.VideoWriter(output_path, fourcc, fps_rate, (frame_width, frame_height))

# Time tracking
start_time = time.time()
max_duration = 20  # Run the program for 20 seconds
detections = {}

def record_detection(name, confidence, current_time):
    """Track detection timestamps and confidence."""
    if name not in detections:
        detections[name] = {"start": current_time, "end": current_time, "confidences": [confidence]}
    else:
        detections[name]["end"] = current_time
        detections[name]["confidences"].append(confidence)

def plot_timeline(detections):
    """Plot a timeline of detections with confidence scores over time."""
    plt.figure(figsize=(12, 6))
    
    for name, times in detections.items():
        # Extract timestamps and confidence scores
        timestamps = np.linspace(times["start"] - start_time, times["end"] - start_time, len(times["confidences"]))
        confidences = times["confidences"]
        
        # Plot confidence scores over time
        plt.plot(timestamps, confidences, marker="o", label=name)
    plt.xlabel("Time (s)")
    plt.ylabel("Confidence (%)")
    plt.title("Detection Timeline with Confidence Scores Over Time")
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig("timeline_with_confidence.png")  # Save as a PNG file
    # plt.show()  # Uncomment if running interactively


print("[INFO] Starting video stream...")
while True:
    current_time = time.time()
    elapsed_time = current_time - start_time

    # Stop the program after 20 seconds
    if elapsed_time > max_duration:
        break

    frame = vs.read()
    frame = imutils.resize(frame, width=frame_width)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # save the current frame for writing later
    video_writer.write(frame)
    # Normalize image for object detection
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Object detectionx
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Face recognition
    boxes_faces = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, boxes_faces)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        confidence = 0.0
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            face_distances = face_recognition.face_distance(data["encodings"], encoding)
            fd_mean = statistics.mean([fd if fd < 0.5 else 1 for fd in face_distances])
            confidence = (1 - fd_mean) * 100
            name = max(counts, key=counts.get)
            names.append(f"{name}: {confidence:.2f}%")
        else:
            names.append(name)
        record_detection(name, confidence, current_time)

    # Draw object detection boxes and track timestamps
    imH, imW, _ = frame.shape
    for i in range(len(scores)):
        if MIN_CONF_THRESH < scores[i] <= 1.0:
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            object_name = labels[int(classes[i])]
            confidence = scores[i] * 100
            label = f"{object_name}: {int(confidence)}%"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 0), 2)
            record_detection(object_name, confidence, current_time)

    # Draw facial recognition boxes
    for ((top, right, bottom, left), name) in zip(boxes_faces, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Facial Recognition and Object Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if (key == ord("q")):
        break
    if (key == ord("c") and contrast_val) < 81:
        contrast_val+=20
        print("new contrast value: {}".format(contrast_val))
        vs.contrast = contrast_val
    if (key == ord("f") and contrast_val > 20):
        contrast_val-=20
        print("new contrast value: {}".format(contrast_val))
        vs.contrast = contrast_val
    if (key == ord("u") and sharpness_val < 100):
        sharpness_val+=50
        print("new contrast value: {}".format(sharpness_val))
        vs.sharpness = sharpness_val
    if key == ord("i") and sharpness_val > -100:
        sharpness_val-=50
        print("new contrast value: {}".format(sharpness_val))
        vs.sharpness = sharpness_val
    fps.update()

fps.stop()
cv2.destroyAllWindows()
vs.stop()

# Release video writer (save the vid)
video_writer.release()

# Plot the timeline after exiting
plot_timeline(detections)
