# 📱 Mobile Quickstart Guide

> Turn your old Android/iOS phone into an AI Camera node in less than 5 minutes.

**Document Version:** v0.1.0
**Last Updated:** 2026-07-25
**Compatible with:** OpenEyes-Live v0.1.x

---

## 🛠️ Choose Your Setup Mode

| Mode | Target Hardware | Ease of Use | Performance | Network Required |
| :--- | :--- | :--- | :--- | :--- |
| **Option A: IP Camera Stream** | Any phone (Android/iOS) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | WiFi (local) |
| **Option B: Termux Local Run** | Android 7.0+ (Snapdragon 660+) | ⭐⭐⭐ | ⭐⭐⭐ | Optional |
| **Option C: Split Pipeline** | Phone + PC/Mac | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | WiFi (local) |

### Which One Should You Choose?

| Your Situation | Recommendation |
| :--- | :--- |
| You have a PC/Mac and an old phone | **Option A** — better performance, easier setup |
| You only have the old phone (no PC) | **Option B** — fully standalone |
| You want video on phone, audio on PC | **Option C** — split video/audio pipelines |
| Your old phone has >4GB RAM (SD 845+) | Option B works well |
| Your old phone has ≤3GB RAM (SD 660) | **Option A** recommended |

---

## 📡 Option A: IP Camera Mode (Recommended)

In this setup, your old phone acts purely as a camera sensor over local WiFi.

### Step 1: Prepare Your Old Phone

1. Connect your old phone and main PC to the **same WiFi network**.
2. Download an IP Camera app on your old phone:

| Platform | App | Notes |
| :--- | :--- | :--- |
| **Android** | [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) | Free, open source, most reliable |
| **iOS** | [iVCam](https://apps.apple.com/app/ivcam/id1054495415) | Free version available |

3. Open the app and tap **"Start Server"**.
4. Note down the URL shown on your phone screen: `http://192.168.1.100:8080/video`
5. **Optional:** Set resolution to **720p**, FPS to **10-15fps**

### Step 2: Run OpenEyes-Live on Main Machine

```bash
# Video only (base config)
openeyes watch --source "http://192.168.1.100:8080/video" --engines encoder+llm

# Video + Audio (full multimodal)
openeyes watch --source "http://192.168.1.100:8080/video" --audio --engines encoder+vad+asr+llm

# Split: video on PC, audio on phone
openeyes watch --source "http://192.168.1.100:8080/video" --audio --split

# Proactive Alert Mode
openeyes watch --source "http://192.168.1.100:8080/video" --mode proactive --alert "water|fire|intruder"
📱 Option B: Standalone Android Mode (Termux)
Run OpenEyes-Live directly on your old Android phone.

⚠️ Important Notes
Note	Details
Technical preview	Works on many devices, but some may have issues
Performance varies	SD 660: slow but usable; SD 845+: good
Battery drain	Expect 15-25% per hour
Memory	<4GB RAM may experience OOM crashes
Step 1: Install Termux
⚠️ Do NOT install from Google Play Store — download from F-Droid

Step 2: Environment Setup
bash
# 1. Update packages
pkg update && pkg upgrade -y

# 2. Install dependencies
pkg install python clang ffmpeg libjpeg-turbo git cmake -y

# 3. Clone repository
git clone https://github.com/vfvincentwong2026/OpenEyes-Live.git
cd OpenEyes-Live

# 4. Install
pip install -r requirements.txt
pip install -e .
Step 3: Grant Camera Permission & Run
bash
# Request camera access
termux-setup-storage

# Video only (lightweight)
openeyes watch --source camera --engines encoder+llm

# Video + Audio (full) — requires 4GB+ RAM
openeyes watch --source camera --audio --engines encoder+vad+asr+llm
🔌 Option C: Split Pipeline Mode
Split video and audio processing across devices for optimal performance.

bash
# Device 1 (Phone): Capture video stream
openeyes capture --source camera --output rtsp://phone.local:8554

# Device 2 (PC): Process video from phone
openeyes watch --source rtsp://phone.local:8554 --engines encoder+llm

# Device 3 (Optional): Process audio separately
openeyes listen --source microphone --engines vad+asr+llm
📋 Engine Combination Reference
Use Case	Recommended Engines	Size
Basic Video Description	encoder+llm	~600MB
Video + Memory	encoder+llm+memory	~650MB
Video + Proactive Alerts	encoder+llm+memory	~650MB
Audio Only (Transcription)	vad+asr	~236MB
Audio + Speaker ID	vad+asr+speaker	~266MB
Full Multimodal	encoder+vad+asr+llm+memory	~900MB
⚠️ Troubleshooting
Problem	Solution
Camera not opening	termux-setup-storage and restart
IP Webcam connection failed	Check IP address, same WiFi?
Slow inference (< 3 t/s)	Use fewer engines, e.g., encoder+llm only
OOM crash	Single pipeline only: video OR audio
Model download slow (China)	Use --mirror=modelscope
Thermal throttling	Reduce FPS to 5-10, remove phone case