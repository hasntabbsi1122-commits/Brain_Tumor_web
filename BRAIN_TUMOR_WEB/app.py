from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import io
import base64
from torchvision import transforms as T

app = Flask(__name__, static_folder='static')
CORS(app)


class HybridEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights, resnet18, ResNet18_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.vgg_features = vgg.features
        self.vgg_block1 = self.vgg_features[:5]
        self.vgg_block2 = self.vgg_features[5:10]
        self.vgg_block3 = self.vgg_features[10:17]
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.res_layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.res_pool = resnet.maxpool
        self.res_layer1 = resnet.layer1
        self.res_layer2 = resnet.layer2
        self.res_layer3 = resnet.layer3
        self.res_layer4 = resnet.layer4
        self.fuse1 = nn.Conv2d(64 + 64, 64, 1)
        self.fuse2 = nn.Conv2d(128 + 128, 128, 1)
        self.fuse3 = nn.Conv2d(256 + 256, 256, 1)

    def forward(self, x):
        import torch.nn.functional as F
        v1 = self.vgg_block1(x); v2 = self.vgg_block2(v1); v3 = self.vgg_block3(v2)
        r0 = self.res_layer0(x); r1 = self.res_layer1(self.res_pool(r0))
        r2 = self.res_layer2(r1); r3 = self.res_layer3(r2); r4 = self.res_layer4(r3)
        r1_r = F.interpolate(r1, size=v1.shape[2:], mode='bilinear', align_corners=False)
        r2_r = F.interpolate(r2, size=v2.shape[2:], mode='bilinear', align_corners=False)
        r3_r = F.interpolate(r3, size=v3.shape[2:], mode='bilinear', align_corners=False)
        f1 = self.fuse1(torch.cat([v1, r1_r], 1))
        f2 = self.fuse2(torch.cat([v2, r2_r], 1))
        f3 = self.fuse3(torch.cat([v3, r3_r], 1))
        return f1, f2, f3, r3, r4


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)


class MultiTaskHybridUNetPP(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.encoder = HybridEncoder()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(256, num_classes)
        )
        self.bottleneck_conv = ConvBlock(512 + num_classes, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, 2); self.conv1 = ConvBlock(256 + 256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2); self.conv2 = ConvBlock(128 + 256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2);  self.conv3 = ConvBlock(64 + 128, 64)
        self.up4 = nn.ConvTranspose2d(64, 64, 2, 2);   self.conv4 = ConvBlock(64 + 64, 64)
        self.up5 = nn.ConvTranspose2d(64, 32, 2, 2);   self.conv5 = ConvBlock(32, 32)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(inplace=True), nn.Conv2d(16, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        import torch.nn.functional as F
        f1, f2, f3, r3, r4 = self.encoder(x)
        cls_f = self.avgpool(r4); cls_f = torch.flatten(cls_f, 1); logits = self.classifier(cls_f)
        cls_s = torch.softmax(logits, dim=1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, r4.size(2), r4.size(3))
        btlnck = self.bottleneck_conv(torch.cat([r4, cls_s], 1))
        d1 = self.conv1(torch.cat([F.interpolate(self.up1(btlnck), size=r3.shape[2:]), r3], 1))
        d2 = self.conv2(torch.cat([F.interpolate(self.up2(d1), size=f3.shape[2:]), f3], 1))
        d3 = self.conv3(torch.cat([F.interpolate(self.up3(d2), size=f2.shape[2:]), f2], 1))
        d4 = self.conv4(torch.cat([F.interpolate(self.up4(d3), size=f1.shape[2:]), f1], 1))
        d5 = self.conv5(F.interpolate(self.up5(d4), size=x.shape[2:]))
        return self.final_conv(d5), logits



print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskHybridUNetPP(num_classes=4).to(device)
MODEL_PATH = model_path = r"C:\Users\KTS\brain_tumor_website\final_best_hybrid_unetpp_model.pth"

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f" Model loaded successfully on {device}!")
except Exception as e:
    print(f" ERROR: {e}")
    exit(1)

class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']



def preprocess_image(image_input):
    if isinstance(image_input, bytes):
        image_input = io.BytesIO(image_input)
    img = Image.open(image_input).convert('RGB')
    img = np.array(img)
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(yuv[:, :, 0])
    img = cv2.resize(cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB), (224, 224))
    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_t).unsqueeze(0)



def create_overlay(image_bytes, binary_mask):
    """
    image_bytes  : raw bytes of the uploaded MRI image
    binary_mask  : 2D numpy array of 0s and 1s, shape (224, 224)
    returns      : base64 PNG string of red overlay on original MRI
    """
    
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_np = np.array(img)
    img_resized = cv2.resize(img_np, (224, 224))
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

   
    kernel = np.ones((3, 3), np.uint8)
    smooth_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    
    colored = np.zeros_like(img_bgr)
    colored[smooth_mask == 1] = (0, 0, 255)  

    
    overlay = img_bgr.copy()
    tumor_pixels = smooth_mask == 1
    if tumor_pixels.any():
        blended = cv2.addWeighted(img_bgr, 0.55, colored, 0.45, 0)
        overlay[tumor_pixels] = blended[tumor_pixels]

       
        contours, _ = cv2.findContours(
            smooth_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)

   
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    buffered = io.BytesIO()
    Image.fromarray(overlay_rgb).save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()



@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/detection.html')
def detection():
    return send_from_directory('static', 'detection.html')

@app.route('/consult.html')
def consult():
    return send_from_directory('static', 'consult.html')

@app.route('/ask_ai.html')
def ask_ai():
    return send_from_directory('static', 'ask_ai.html')

@app.route('/about.html')
def about():
    return send_from_directory('static', 'about.html')

@app.route('/atyle1.css')
def css():
    return send_from_directory('static', 'atyle1.css')



@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})

        file = request.files['image']
        image_bytes = file.read()  

        
        img_tensor = preprocess_image(image_bytes).to(device)
        with torch.no_grad():
            mask_pred, logits = model(img_tensor)

       
        probs = torch.softmax(logits, dim=1)[0]
        class_id = torch.argmax(probs).item()
        confidence = float(probs[class_id])

        
        mask_np = mask_pred[0, 0].cpu().numpy()
        binary_mask = (mask_np > 0.5).astype(np.uint8)

       
        mask_255 = (binary_mask * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(mask_255).save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode()

        
        overlay_b64 = create_overlay(image_bytes, binary_mask)

        return jsonify({
            'success': True,
            'prediction': {
                'class': class_names[class_id],
                'confidence': round(confidence * 100, 2) 
            },
            'segmentation_mask': mask_b64,  
            'overlay_image': overlay_b64      
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)