import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from PIL import Image
import tensorflow as tf
from yad2k.models.keras_yolo import yolo_head
from yad2k.utils.utils import draw_boxes, scale_boxes, read_classes, read_anchors, preprocess_image

app = Flask(__name__)

# 全局变量
current_video_path = None
current_confidence_threshold = 0.3
stop_video = False  # 新增：停止标志
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route('/outputs/<path:filename>')
def serve_outputs(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


# 加载YOLO
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
model_image_size = (608, 608)
yolo_model = tf.keras.models.load_model("model_data/", compile=False)


# ---- 图片检测 ----
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    box_scores = box_confidence * box_class_probs
    box_classes = tf.math.argmax(box_scores, axis=-1)
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1)
    filtering_mask = box_class_scores >= threshold

    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')
    nms_indices = tf.image.non_max_suppression(
        boxes, scores, max_boxes, iou_threshold=iou_threshold)
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)

    return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs

    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(
        boxes, box_confidence, box_class_probs, threshold=score_threshold)

    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = yolo_non_max_suppression(
        scores, boxes, classes, max_boxes=max_boxes, iou_threshold=iou_threshold)

    return scores, boxes, classes


def yolo_boxes_to_corners(box_xy, box_wh):
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])


# ---- 视频流 ----
def generate_frames():
    global current_video_path, current_confidence_threshold, stop_video

    stop_video = False
    if current_video_path and os.path.exists(current_video_path):
        camera = cv2.VideoCapture(current_video_path)
    else:
        camera = cv2.VideoCapture(0)

    try:
        while True:
            if stop_video:
                break
            success, frame = camera.read()
            if not success:
                if current_video_path:
                    camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            else:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_rgb)
                resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
                image_data = np.array(resized_image, dtype='float32') / 255.
                image_data = np.expand_dims(image_data, 0)

                try:
                    yolo_model_outputs = yolo_model(image_data)
                    yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
                    scores, boxes, classes = yolo_eval(
                        yolo_outputs, [frame.shape[0], frame.shape[1]],
                        max_boxes=10, score_threshold=current_confidence_threshold, iou_threshold=0.5
                    )
                    out_img = draw_boxes(image, boxes, classes, class_names, scores)
                    out_img_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"检测错误: {e}")
                    out_img_bgr = frame

                ret, buffer = cv2.imencode('.jpg', out_img_bgr)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 未发送文件
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        # 无文件
        if not file.filename:
            return jsonify({'error': 'No selected file'})
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image, image_data = preprocess_image(filepath, model_image_size=model_image_size)
            yolo_model_outputs = yolo_model(image_data)
            yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))

            confidence_threshold = float(request.form.get('confidence_threshold', 0.3))
            scores, boxes, classes = yolo_eval(yolo_outputs, [image.size[1], image.size[0]],
                                               max_boxes=10, score_threshold=confidence_threshold, iou_threshold=0.5)

            out_img = draw_boxes(image, boxes, classes, class_names, scores)
            output_filename = 'detected_' + filename
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            cv2.imwrite(output_path, out_img)

            detections = []
            for i, c in enumerate(classes):
                detections.append({
                    'class': class_names[c],
                    'score': float(scores[i]),
                    'box': boxes[i].numpy().tolist()
                })

            return jsonify({'success': True, 'detections': detections, 'output_image': output_filename})

    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.post('/upload_video')
def upload_video():
    global current_video_path, current_confidence_threshold, stop_video
    stop_video = False
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'})
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        current_video_path = filepath
        current_confidence_threshold = float(request.form.get('confidence_threshold', 0.3))
        return jsonify({'success': True, 'message': '视频上传成功'})
    return jsonify({'error': 'Invalid file type'})


@app.post('/camera_seek')
def use_camera():
    global current_video_path, stop_video
    stop_video = False
    current_video_path = None
    return jsonify({'success': True, 'message': '已切换回摄像头'})


# ---- 停止视频检测 ----
@app.post('/stop_video')
def stop_video_route():
    global stop_video
    stop_video = True
    return jsonify({'success': True, 'message': '视频检测已停止'})


if __name__ == '__main__':
    app.run(debug=True)
