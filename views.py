import cv2
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.views import View

from .src.yolox.yolox_cam import yolox_inference
from .src.yolox.yolox_cam import draw_yolox_predictions

# Create your views here.

# ストリーミング画像・映像を表示するview
class IndexView(View):
    print("ここは、IndexView")
    def get(self, request):
        return render(request, 'base.html', {})

# ストリーミング画像を定期的に返却するview
def video_feed_view():
    print("ここは、Ivideo_feed_view")
    return lambda _: StreamingHttpResponse(generate_frame(), content_type='multipart/x-mixed-replace; boundary=frame')

# フレーム生成・返却する処理
def generate_frame():
    print("ここは、generate_frame")
    capture = cv2.VideoCapture(0)  # USBカメラから

    while True:
        # print("常時実行中！")
        if not capture.isOpened():
            print("Capture is not opened.")
            break
        # カメラからフレーム画像を取得
        ret, frame = capture.read()
        if not ret:
            print("Failed to read frame.")
            break

        # 画像判定とかの処理を入れるならここ
        # YOLO推論を入れる
        bboxes, bbclasses, scores = yolox_inference(frame, model, test_size)

        # 取得したクラス番号を、カテゴリ名に変更する
        get_classes = np.array(bbclasses)

        # frameに、YOLOX検出矩形を表示する
        frame  = draw_yolox_predictions(frame, bboxes, scores, bbclasses, confthre, COCO_CLASSES)



        # フレーム画像バイナリに変換
        ret, jpeg = cv2.imencode('.jpg', frame)
        byte_frame = jpeg.tobytes()
        # フレーム画像のバイナリデータをユーザーに送付する
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + byte_frame + b'\r\n\r\n')
    capture.release()
